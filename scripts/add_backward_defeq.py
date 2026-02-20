#!/usr/bin/env python3
"""
Add `set_option backward.isDefEq.respectTransparency false in` before failing declarations.

Traverses the import DAG forward (roots first) so that each module is only
built after all its imports are clean.  No "discovery" builds needed.
"""

import argparse
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dag_traversal import (
    DAG,
    Display,
    TraversalResult,
    force_exit,
    inflight_register,
    inflight_unregister,
    lake_build,
    traverse_dag,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

SET_OPTION_LINE = "set_option backward.isDefEq.respectTransparency false in\n"


@dataclass
class FileResult:
    """Result of processing one file."""

    fixed: int = 0
    already_present: int = 0


class _AddDisplay(Display):
    """Progress display for the add script."""

    def __init__(self):
        super().__init__()
        self.total_fixed = 0
        self.total_failed = 0

    def _status_line(self) -> str:
        return (
            f"[{self.completed}/{self.total}]  "
            f"Working on: {self.inflight}  "
            f"Fixed: {self.total_fixed}  Failed: {self.total_failed}"
        )

    def on_module(self, module_name: str, result: FileResult | None, error: Exception | None):
        with self.lock:
            if result:
                self.total_fixed += result.fixed
                if result.fixed > 0:
                    sym = "+"
                else:
                    sym = "·"
                detail = f"+{result.fixed}" if result.fixed else ""
                if result.already_present:
                    detail += f" already {result.already_present}"
                self.messages.append(f"  {sym} {module_name} {detail}")
            elif error:
                self.total_failed += 1
                self.messages.append(f"  ! {module_name}: {error}")
            self._redraw()


def is_inside_block_comment(lines: list[str], line_idx: int) -> bool:
    """Check if line_idx is inside a /- ... -/ block comment."""
    for i in range(line_idx - 1, -1, -1):
        line = lines[i]
        if "/-" in line:
            start_pos = line.find("/-")
            end_pos = line.find("-/", start_pos + 2)
            if end_pos == -1:
                return True
            return False
        if "-/" in line:
            return False
    return False


def find_declaration_start(lines: list[str], error_line: int) -> int:
    """Find the start of the declaration containing the error.

    Walks backwards from error_line (1-indexed) to the first blank line
    that isn't inside a block comment.

    TODO: This heuristic is imperfect — it can misidentify the declaration
    boundary when there's no blank line between consecutive declarations,
    or when multi-line attributes bridge two declarations. Ideas for a
    better rule welcome.
    """
    idx = error_line - 1  # convert to 0-indexed

    while idx > 0:
        idx -= 1
        if lines[idx].strip() == "":
            if not is_inside_block_comment(lines, idx):
                return idx + 1

    return 0


# lake build outputs "error: filepath:line:col: message"
LAKE_ERROR_PATTERN = re.compile(r"^error: (.+?\.lean):(\d+):\d+:", re.MULTILINE)


def parse_errors_in_file(build_output: str, filepath: Path) -> list[int]:
    """Parse error line numbers for a specific file from build output."""
    filepath_str = str(filepath)
    error_lines: list[int] = []
    for match in LAKE_ERROR_PATTERN.finditer(build_output):
        if match.group(1) == filepath_str:
            line_num = int(match.group(2))
            if line_num not in error_lines:
                error_lines.append(line_num)
    error_lines.sort()
    return error_lines


def parse_error_modules(build_output: str) -> set[str]:
    """Parse module names of all files with errors from build output."""
    modules: set[str] = set()
    for match in LAKE_ERROR_PATTERN.finditer(build_output):
        filepath = match.group(1)
        module = filepath.replace("/", ".").removesuffix(".lean")
        modules.add(module)
    return modules


def count_errors_per_module(build_output: str) -> dict[str, int]:
    """Count errors per module from build output."""
    counts: dict[str, int] = {}
    for match in LAKE_ERROR_PATTERN.finditer(build_output):
        module = match.group(1).replace("/", ".").removesuffix(".lean")
        counts[module] = counts.get(module, 0) + 1
    return counts


PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]")


def initial_build() -> tuple[set[str], str]:
    """Run a full lake build to discover which modules have errors.

    Returns (error_modules, build_output).
    """
    print("Running initial build...", flush=True)
    proc = subprocess.Popen(
        ["lake", "build"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PROJECT_DIR,
    )
    output_lines: list[str] = []
    for line in proc.stdout:
        output_lines.append(line)
        m = PROGRESS_RE.search(line)
        if m:
            print(f"\r  [{m.group(1)}/{m.group(2)}]", end="", flush=True)
    proc.wait()
    print()  # newline after progress

    output = "".join(output_lines)
    error_mods = parse_error_modules(output)
    print(f"  {len(error_mods)} modules with errors")
    return error_mods, output


class UnfixableError(Exception):
    """Raised when errors remain that can't be fixed by set_option insertion."""

    def __init__(self, msg: str, output: str = ""):
        self.output = output
        super().__init__(msg)


def make_process_module(timeout: int) -> Callable:
    """Create the per-module action callback."""

    def process_module(module_name: str, filepath: Path) -> FileResult:
        rel_path = filepath.relative_to(PROJECT_DIR)

        # Initial build
        ok, output = lake_build(module_name, PROJECT_DIR, timeout)
        if ok:
            return FileResult()

        error_lines = parse_errors_in_file(output, rel_path)
        if not error_lines:
            # Errors are in other files, not ours — treat as success
            return FileResult()

        original_text = filepath.read_text()
        lines = original_text.splitlines(keepends=True)
        fixed = 0
        already_present = 0

        inflight_register(filepath, original_text)

        try:
            # Process errors first-to-last. After each insertion, line numbers
            # shift by 1, so we track the cumulative offset.
            offset = 0
            for error_line in error_lines:
                adjusted_line = error_line + offset
                decl_start = find_declaration_start(lines, adjusted_line)

                # Check if set_option already present
                if decl_start > 0 and SET_OPTION_LINE.strip() in lines[decl_start - 1]:
                    already_present += 1
                    continue
                if lines[decl_start].strip().startswith(SET_OPTION_LINE.strip()):
                    already_present += 1
                    continue

                # Insert set_option
                lines = lines[:decl_start] + [SET_OPTION_LINE] + lines[decl_start:]
                offset += 1

                # Write and test
                filepath.write_text("".join(lines))
                ok, output = lake_build(module_name, PROJECT_DIR, timeout)

                if ok:
                    fixed += 1
                    # No more errors — done
                    return FileResult(fixed=fixed, already_present=already_present)

                # Check if this error is gone
                new_errors = parse_errors_in_file(output, rel_path)
                # The error was at error_line; after insertion it would be at error_line+offset
                if adjusted_line + 1 in new_errors:
                    # Fix didn't help — revert this insertion
                    lines = lines[:decl_start] + lines[decl_start + 1 :]
                    offset -= 1
                    filepath.write_text("".join(lines))
                    raise UnfixableError(
                        f"set_option didn't fix error at line {error_line}",
                        output,
                    )

                fixed += 1
        except UnfixableError:
            raise
        except BaseException:
            filepath.write_text(original_text)
            raise
        finally:
            inflight_unregister(filepath)

        return FileResult(fixed=fixed, already_present=already_present)

    return process_module


def print_summary(
    results: list[TraversalResult],
    dag: DAG,
    duration: float,
):
    total_fixed = 0
    files_fixed = 0
    files_failed = 0

    for tr in results:
        if tr.error:
            files_failed += 1
        elif tr.result and tr.result.fixed > 0:
            files_fixed += 1
            total_fixed += tr.result.fixed

    skipped = len(dag.modules) - len(results)

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Modules processed:  {len(results)}")
    print(f"  Modules fixed:      {files_fixed}")
    print(f"  Modules failed:     {files_failed}")
    print(f"  Modules skipped:    {skipped}")
    print(f"  set_options added:  {total_fixed}")
    print(f"  Duration:           {duration:.0f}s")

    if files_failed:
        print()
        print("Failed modules:")
        for tr in results:
            if tr.error:
                print(f"  {tr.module_name}: {tr.error}")
                if isinstance(tr.error, UnfixableError) and tr.error.output.strip():
                    for line in tr.error.output.strip().splitlines()[-10:]:
                        print(f"    {line}")

    if skipped:
        processed = {tr.module_name for tr in results}
        skipped_names = sorted(set(dag.modules.keys()) - processed)
        print(f"\nSkipped modules ({skipped}):")
        for name in skipped_names[:20]:
            print(f"  {name}")
        if len(skipped_names) > 20:
            print(f"  ... and {len(skipped_names) - 20} more")


def main():
    parser = argparse.ArgumentParser(
        description="Add set_option backward.isDefEq.respectTransparency false in "
        "before failing declarations."
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Max parallel workers (default: cpu_count)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Build timeout per module in seconds (default: 600)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Only process these files (paths relative to project root)",
    )
    parser.add_argument(
        "--no-initial",
        action="store_true",
        help="Skip the initial build (build every module individually)",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Build DAG
    print("Building import DAG...", flush=True)
    dag = DAG.from_directories(PROJECT_DIR)
    print(f"  {len(dag.modules)} modules parsed")

    # Filter to requested files
    if args.files:
        requested = set()
        for f in args.files:
            mod = f.replace("/", ".").removesuffix(".lean")
            requested.add(mod)
        dag = dag.subset(requested)
        print(f"  Filtered to {len(dag.modules)} modules")

    # Initial build: run a full build to find which modules have errors,
    # then skip everything outside their forward cone in the DAG.
    skip: set[str] | None = None
    if not args.no_initial and not args.files:
        error_mods, build_output = initial_build()
        if not error_mods:
            print("No errors found, nothing to do.")
            return
        cone = dag.forward_cone(error_mods)
        skip = set(dag.modules.keys()) - cone
        print(f"  {len(cone)} modules in forward cone, skipping {len(skip)}")

    # Weight = LOC × error count, so expensive files get critical-path
    # priority in the DAG traversal scheduler.
    weights: dict[str, int] | None = None
    if not args.no_initial and not args.files:
        error_counts = count_errors_per_module(build_output)
        weights = {}
        for name in error_counts:
            if name in dag.modules:
                fp = dag.project_root / dag.modules[name].filepath
                weights[name] = len(fp.read_text().splitlines()) * error_counts[name]

    # Traverse forward
    display = _AddDisplay()
    action = make_process_module(args.timeout)

    display.start(len(dag.modules))
    try:
        results = traverse_dag(
            dag,
            action,
            direction="forward",
            max_workers=args.max_workers,
            progress_callback=display.on_progress,
            module_callback=display.on_module,
            stop_on_failure=True,
            skip=skip,
            weights=weights,
        )
    except KeyboardInterrupt:
        display.stop()
        print("\nInterrupted. Press Ctrl-C again if the process doesn't exit.", flush=True)
        force_exit(1)
    finally:
        display.stop()

    duration = time.time() - start_time
    print_summary(results, dag, duration)

    failed = any(tr.error for tr in results)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
