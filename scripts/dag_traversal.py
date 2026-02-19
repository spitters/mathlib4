#!/usr/bin/env python3
"""
Reusable parallel DAG traversal for Lean import graphs.

Parses the import DAG from .lean source files and parallelizes an action
over a forward or backward traversal.  Each module is submitted to the
thread pool the moment its last in-DAG dependency finishes, giving true
maximum parallelism.

CLI usage:
    dag_traversal.py --forward 'lake build {}'
    dag_traversal.py --backward --module 'my_script {}'
    dag_traversal.py --forward -j4 'echo {}'

Library usage:
    from dag_traversal import DAG, traverse_dag

    dag = DAG.from_directories(Path("."))
    results = traverse_dag(dag, my_action, direction="backward")
"""

import argparse
import atexit
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Generic, TypeVar

T = TypeVar("T")

# Set by traverse_dag on KeyboardInterrupt.  Action callbacks can check this
# to bail out early instead of spawning new subprocesses.
shutdown_event = Event()

# ANSI escape codes
CLEAR_LINE = "\033[2K"
HIDE_CURSOR = "\033[?25l"
SHOW_CURSOR = "\033[?25h"


class ShutdownError(Exception):
    """Raised when a shutdown has been requested (e.g. Ctrl-C)."""


# Thread-safe registry of in-flight file modifications.  On abnormal exit
# (e.g. Ctrl-C), any file whose worker was mid-build gets reverted.
_inflight_lock = Lock()
_inflight: dict[Path, str] = {}  # filepath -> original content


def inflight_register(path: Path, content: str):
    """Register original file content so it can be reverted on abnormal exit."""
    with _inflight_lock:
        _inflight[path] = content


def inflight_unregister(path: Path):
    """Unregister a file after processing completes normally."""
    with _inflight_lock:
        _inflight.pop(path, None)


def _revert_inflight():
    with _inflight_lock:
        for path, content in _inflight.items():
            try:
                path.write_text(content)
            except Exception:
                pass
        _inflight.clear()


atexit.register(_revert_inflight)


def _kill_builds():
    """Kill all active lake build process groups."""
    with _active_builds_lock:
        for proc in _active_builds:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass


def force_exit(code: int = 1):
    """Kill active builds, revert in-flight files, and force-exit.

    Uses os._exit to bypass ThreadPoolExecutor's atexit handler, which
    would block joining worker threads that may still be running builds.
    """
    _kill_builds()
    _revert_inflight()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


# Active lake build subprocesses, tracked so force_exit can kill them.
_active_builds_lock = Lock()
_active_builds: set[subprocess.Popen] = set()


def lake_build(
    module_name: str, project_dir: Path, timeout: int = 600,
) -> tuple[bool, str]:
    """Run lake build for a module. Returns (success, output).

    Checks shutdown_event before spawning and polls it every 0.5s during
    the build, killing the subprocess and raising ShutdownError if set.
    Each build runs in its own process group (start_new_session) so the
    entire tree (lake + lean children) can be killed cleanly.
    """
    if shutdown_event.is_set():
        raise ShutdownError("shutdown requested")
    proc = subprocess.Popen(
        ["lake", "build", module_name],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=project_dir,
        start_new_session=True,
    )
    with _active_builds_lock:
        _active_builds.add(proc)
    try:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False, "Build timed out"
            try:
                stdout, _ = proc.communicate(timeout=min(remaining, 0.5))
                return proc.returncode == 0, stdout
            except subprocess.TimeoutExpired:
                if shutdown_event.is_set():
                    raise ShutdownError("shutdown requested")
    finally:
        with _active_builds_lock:
            _active_builds.discard(proc)
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()
            proc.wait()


@dataclass
class ModuleInfo:
    """Information about a single module in the DAG."""

    name: str
    filepath: Path
    imports: list[str] = field(default_factory=list)
    importers: list[str] = field(default_factory=list)


@dataclass
class TraversalResult(Generic[T]):
    """Result of processing a single module."""

    module_name: str
    filepath: Path
    result: T | None = None
    error: Exception | None = None


def _parse_imports(filepath: Path) -> list[str]:
    """Parse import statements from a .lean file.

    Follows the same approach as scripts/count-trans-deps.py: read lines
    until /-! (module docstring), matching import statements along the way.
    """
    imports = []
    with open(filepath, "r") as f:
        for line in f:
            if "/-!" in line:
                break
            match = re.match(r"^(?:public\s+)?import\s+(\S+)", line)
            if match:
                imports.append(match.group(1))
    return imports


def _filepath_to_module(filepath: str) -> str:
    """Convert a file path like Mathlib/Foo/Bar.lean to Mathlib.Foo.Bar."""
    return filepath.replace("/", ".").removesuffix(".lean")


class DAG:
    """The import DAG for a set of Lean modules."""

    def __init__(
        self,
        modules: dict[str, ModuleInfo] | None = None,
        project_root: Path | None = None,
    ):
        self.modules: dict[str, ModuleInfo] = modules or {}
        self.project_root: Path | None = project_root

    @staticmethod
    def from_directories(
        project_root: Path,
        directories: list[str] | None = None,
    ) -> "DAG":
        """Build DAG by parsing imports from .lean files in directories."""
        if directories is None:
            directories = ["Mathlib", "MathlibTest", "Archive", "Counterexamples"]

        modules: dict[str, ModuleInfo] = {}

        for directory in directories:
            dir_path = project_root / directory
            if not dir_path.exists():
                continue
            for root, _, files in os.walk(dir_path):
                for fname in files:
                    if not fname.endswith(".lean"):
                        continue
                    full_path = Path(root) / fname
                    rel_path = full_path.relative_to(project_root)
                    module_name = _filepath_to_module(str(rel_path))
                    imports = _parse_imports(full_path)
                    modules[module_name] = ModuleInfo(
                        name=module_name,
                        filepath=rel_path,
                        imports=[i for i in imports if i != module_name],
                    )

        # Build reverse edges (importers)
        for name, info in modules.items():
            for imp in info.imports:
                if imp in modules:
                    modules[imp].importers.append(name)

        return DAG(modules, project_root.resolve())

    def subset(self, module_names: set[str]) -> "DAG":
        """Return a sub-DAG containing only the specified modules."""
        new_modules: dict[str, ModuleInfo] = {}
        for name in module_names:
            if name not in self.modules:
                continue
            info = self.modules[name]
            new_modules[name] = ModuleInfo(
                name=name,
                filepath=info.filepath,
                imports=[i for i in info.imports if i in module_names],
                importers=[i for i in info.importers if i in module_names],
            )
        return DAG(new_modules, self.project_root)

    def forward_cone(self, seeds: set[str]) -> set[str]:
        """Return *seeds* plus all modules that transitively import any seed."""
        from collections import deque

        result: set[str] = set()
        queue: deque[str] = deque(s for s in seeds if s in self.modules)
        while queue:
            name = queue.popleft()
            if name in result:
                continue
            result.add(name)
            for imp in self.modules[name].importers:
                if imp in self.modules and imp not in result:
                    queue.append(imp)
        return result

    def levels_backward(self) -> list[list[str]]:
        """Compute levels for backward (downstream-first) traversal.

        Useful for display/reporting. The traverse_dag function does not use
        this -- it schedules dynamically for maximum parallelism.

        Level 0 = modules not imported by any other module in the DAG.
        Level n+1 = modules whose importers are all at level <= n.
        """
        remaining_importers = {
            name: len(info.importers) for name, info in self.modules.items()
        }
        remaining = set(self.modules.keys())
        levels: list[list[str]] = []

        while remaining:
            level = sorted(m for m in remaining if remaining_importers[m] == 0)
            if not level:
                levels.append(sorted(remaining))
                break
            levels.append(level)
            for m in level:
                remaining.discard(m)
                for imp in self.modules[m].imports:
                    if imp in remaining_importers:
                        remaining_importers[imp] -= 1

        return levels


def traverse_dag(
    dag: DAG,
    action: Callable[[str, Path], T],
    direction: str = "backward",
    max_workers: int | None = None,
    module_callback: Callable[[str, T | None, Exception | None], None] | None = None,
    progress_callback: Callable[[int, int], None] | None = None,
    stop_on_failure: bool = False,
    skip: set[str] | None = None,
) -> list[TraversalResult[T]]:
    """Process modules in DAG order with maximum parallelism.

    Each module is submitted to the thread pool the instant all its
    in-DAG dependencies have finished.

    Args:
        dag: The DAG to traverse.
        action: Callable(module_name, filepath) -> result.  Raise to signal
                failure.
        direction: "backward" (downstream first) or "forward" (upstream first).
        max_workers: Thread pool size (default: 2 * cpu_count).
        module_callback: Called after each module as (module_name, result, error).
        progress_callback: Called after each module as (completed, total).
        stop_on_failure: If True, when a module fails (action raises), its
                         successors in the DAG are skipped.
        skip: Set of module names to mark as completed without running the
              action.  Useful after an initial build (e.g. a full ``lake build``)
              identifies which modules are already clean.

    Returns:
        List of TraversalResult for all processed modules.
    """
    all_names = set(dag.modules.keys())

    if direction == "backward":
        # Process modules that nothing depends on first.
        # A module's "dependencies" are its importers (things that import it);
        # once all importers are done, this module can run.
        deps_of = {
            name: [m for m in info.importers if m in all_names]
            for name, info in dag.modules.items()
        }
        successors_of = {
            name: [m for m in info.imports if m in all_names]
            for name, info in dag.modules.items()
        }
    elif direction == "forward":
        # Process modules with no imports first.
        deps_of = {
            name: [m for m in info.imports if m in all_names]
            for name, info in dag.modules.items()
        }
        successors_of = {
            name: [m for m in info.importers if m in all_names]
            for name, info in dag.modules.items()
        }
    else:
        raise ValueError(f"Unknown direction: {direction!r}")

    if max_workers is None:
        max_workers = (os.cpu_count() or 4) * 2

    total = len(dag.modules)
    if total == 0:
        return []

    lock = Lock()
    done_event = Event()
    remaining_deps: dict[str, int] = {
        name: len(deps_of[name]) for name in dag.modules
    }
    all_results: list[TraversalResult[T]] = []
    completed_count = 0
    skipped: set[str] = set()

    def resolve(rel: Path) -> Path:
        if dag.project_root:
            return dag.project_root / rel
        return rel

    def mark_skipped(name: str):
        """Recursively mark a module and all its successors as skipped."""
        if name in skipped:
            return
        skipped.add(name)
        for succ in successors_of.get(name, []):
            if succ in remaining_deps:
                mark_skipped(succ)

    def on_done(future: Future, module_name: str):
        nonlocal completed_count

        filepath = resolve(dag.modules[module_name].filepath)
        try:
            result = future.result()
            tr = TraversalResult(module_name, filepath, result, None)
        except Exception as e:
            tr = TraversalResult(module_name, filepath, None, e)

        failed = tr.error is not None

        ready: list[str] = []
        with lock:
            all_results.append(tr)
            completed_count += 1
            cc = completed_count

            if stop_on_failure and failed:
                # Don't unlock successors — mark them as skipped
                for succ in successors_of.get(module_name, []):
                    if succ in remaining_deps:
                        mark_skipped(succ)
            else:
                # Decrement dep counts and collect newly ready modules.
                # Skip-set modules that become ready are completed inline
                # (no action) and their successors are cascaded.
                cascade = [module_name]
                while cascade:
                    current = cascade.pop()
                    for succ in successors_of.get(current, []):
                        if succ in remaining_deps and succ not in skipped:
                            remaining_deps[succ] -= 1
                            if remaining_deps[succ] == 0:
                                if succ in skip_set:
                                    fp = resolve(dag.modules[succ].filepath)
                                    all_results.append(TraversalResult(succ, fp))
                                    completed_count += 1
                                    cascade.append(succ)
                                else:
                                    ready.append(succ)

            cc = completed_count
            if cc + len(skipped) >= total:
                done_event.set()

        if module_callback:
            module_callback(module_name, tr.result, tr.error)
        if progress_callback:
            progress_callback(cc, total - len(skipped))

        # Submit newly ready modules
        for name in ready:
            submit(name)

    def submit(name: str):
        info = dag.modules[name]
        fut = executor.submit(action, name, resolve(info.filepath))
        fut.add_done_callback(lambda f, n=name: on_done(f, n))

    # Process skip modules before starting the executor.  We iterate in
    # topological order (BFS from zero-dep seeds) so that each skipped
    # module's successors have their dep counts correctly decremented.
    skip_set = skip or set()

    if skip_set:
        from collections import deque

        skip_queue: deque[str] = deque(
            name for name, count in remaining_deps.items()
            if count == 0 and name in skip_set
        )
        while skip_queue:
            name = skip_queue.popleft()
            filepath = resolve(dag.modules[name].filepath)
            all_results.append(TraversalResult(name, filepath))
            completed_count += 1
            for succ in successors_of.get(name, []):
                if succ in remaining_deps:
                    remaining_deps[succ] -= 1
                    if remaining_deps[succ] == 0 and succ in skip_set:
                        skip_queue.append(succ)

    executor = ThreadPoolExecutor(max_workers=max_workers)
    try:
        # Seed with all modules that have zero deps (excluding already-skipped)
        seeds = [
            name for name, count in remaining_deps.items()
            if count == 0 and name not in skip_set
        ]
        if not seeds and completed_count < total:
            raise ValueError(
                f"No modules with zero dependencies -- possible cycle in DAG "
                f"({total} modules)"
            )
        if not seeds:
            done_event.set()
        for name in seeds:
            submit(name)

        # Wait until all modules are processed or skipped.
        # We cannot use executor.shutdown(wait=True) here because on_done
        # callbacks submit new futures — shutdown would reject them.
        # Use a timeout loop so the main thread can receive KeyboardInterrupt
        # (Event.wait() without timeout blocks signal delivery on Linux).
        while not done_event.wait(timeout=0.5):
            pass
        executor.shutdown(wait=True)
    except KeyboardInterrupt:
        shutdown_event.set()
        executor.shutdown(wait=False, cancel_futures=True)
        raise

    unprocessed = total - completed_count - len(skipped)
    if unprocessed > 0:
        unreached = [
            name
            for name, c in remaining_deps.items()
            if c > 0 and name not in skipped
        ]
        raise ValueError(
            f"Traversal incomplete: {completed_count}/{total} modules processed, "
            f"{len(skipped)} skipped, {unprocessed} unreached (possible cycle). "
            f"Examples: {unreached[:5]}"
        )

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


class _CliDisplay:
    """ANSI progress display for CLI mode."""

    def __init__(self):
        self.lock = Lock()
        self.completed = 0
        self.total = 0
        self.ok = 0
        self.fail = 0
        self.messages: list[str] = []
        self.displayed_lines = 0
        self.started = False

    def start(self, total: int):
        self.total = total
        self.started = True
        print(HIDE_CURSOR, end="", flush=True)
        self._redraw()

    def stop(self):
        if self.started:
            # Clear the status line
            if self.displayed_lines > 0:
                print(f"\033[{self.displayed_lines}A", end="")
                for _ in range(self.displayed_lines):
                    print(CLEAR_LINE)
                print(f"\033[{self.displayed_lines}A", end="", flush=True)
            print(SHOW_CURSOR, end="", flush=True)
            self.started = False

    def _redraw(self):
        if not self.started:
            return
        if self.displayed_lines > 0:
            print(f"\033[{self.displayed_lines}A", end="")
        for msg in self.messages:
            print(CLEAR_LINE + msg)
        self.messages.clear()

        line = (
            CLEAR_LINE
            + f"[{self.completed}/{self.total}]  ok: {self.ok}  fail: {self.fail}"
        )
        print(line, flush=True)
        self.displayed_lines = 1

    def on_progress(self, completed: int, total: int):
        with self.lock:
            self.completed = completed
            self.total = total
            self._redraw()

    def on_module(self, module_name: str, result, error: Exception | None):
        with self.lock:
            if error:
                self.fail += 1
                self.messages.append(f"  \u2717 {module_name}: {error}")
            else:
                self.ok += 1
                self.messages.append(f"  \u2713 {module_name}")
            self._redraw()


class _CommandFailed(Exception):
    """Raised when a shell command exits with non-zero status."""

    def __init__(self, exit_code: int, output: str = ""):
        self.exit_code = exit_code
        self.output = output
        super().__init__(f"exit {exit_code}")


def _cli_main():
    parser = argparse.ArgumentParser(
        description="Run a command on each Lean module in import-DAG order.",
        epilog="Use {} in the command as a placeholder for the file path "
        "(or module name with --module).",
    )

    direction = parser.add_mutually_exclusive_group(required=True)
    direction.add_argument(
        "--forward",
        action="store_const",
        const="forward",
        dest="direction",
        help="Process upstream (roots) first",
    )
    direction.add_argument(
        "--backward",
        action="store_const",
        const="backward",
        dest="direction",
        help="Process downstream (leaves) first",
    )

    parser.add_argument(
        "command",
        help="Command template; {} is replaced with filepath or module name",
    )
    parser.add_argument(
        "--module",
        action="store_true",
        help="Replace {} with module name (Mathlib.Foo.Bar) instead of file path",
    )
    parser.add_argument(
        "-j",
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers (default: 2 * cpu count)",
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("."),
        help="Project root directory (default: cwd)",
    )
    parser.add_argument(
        "--directories",
        nargs="+",
        default=None,
        help="Directories to scan (default: Mathlib MathlibTest Archive Counterexamples)",
    )

    args = parser.parse_args()

    print("Building import DAG...", end="", flush=True)
    dag = DAG.from_directories(args.dir, args.directories)
    print(f" {len(dag.modules)} modules")

    use_module = args.module
    cmd_template = args.command

    def action(module_name: str, filepath: Path) -> int:
        replacement = module_name if use_module else str(filepath)
        cmd = cmd_template.replace("{}", shlex.quote(replacement))
        result = subprocess.run(
            cmd, shell=True, cwd=dag.project_root,
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise _CommandFailed(result.returncode, result.stdout + result.stderr)
        return result.returncode

    display = _CliDisplay()
    display.start(len(dag.modules))

    try:
        results = traverse_dag(
            dag,
            action,
            direction=args.direction,
            max_workers=args.max_workers,
            module_callback=display.on_module,
            progress_callback=display.on_progress,
            stop_on_failure=True,
        )
    except KeyboardInterrupt:
        display.stop()
        print("Interrupted.")
        sys.exit(1)
    finally:
        display.stop()

    # Summary
    ok = sum(1 for r in results if r.error is None)
    failed = sum(1 for r in results if r.error is not None)
    skipped = len(dag.modules) - len(results)

    print(f"Done: {ok} ok, {failed} failed, {skipped} skipped")

    if failed:
        print(f"\nFailed modules:")
        for r in results:
            if r.error:
                print(f"  {r.module_name}: {r.error}")
                if isinstance(r.error, _CommandFailed) and r.error.output.strip():
                    for line in r.error.output.strip().splitlines():
                        print(f"    {line}")

    if skipped:
        processed = {r.module_name for r in results}
        skipped_names = sorted(set(dag.modules.keys()) - processed)
        print(f"\nSkipped modules ({skipped}):")
        for name in skipped_names[:20]:
            print(f"  {name}")
        if len(skipped_names) > 20:
            print(f"  ... and {len(skipped_names) - 20} more")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    _cli_main()
