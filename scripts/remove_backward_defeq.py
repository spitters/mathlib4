#!/usr/bin/env python3
"""
Remove unnecessary `set_option backward.isDefEq.respectTransparency false in` from Mathlib.

Tries removing each occurrence (that isn't followed by a comment), testing
whether the file still builds. Processes files in reverse import-DAG order
(downstream first) to minimize unnecessary rebuilds.
"""

import argparse
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from dag_traversal import (
    DAG,
    Display,
    force_exit,
    inflight_register,
    inflight_unregister,
    lake_build,
    traverse_dag,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent

# Patterns
SET_OPTION_REMOVABLE = re.compile(
    r"^\s*set_option backward\.isDefEq\.respectTransparency false in\s*$"
)
SET_OPTION_WITH_COMMENT = re.compile(
    r"^\s*set_option backward\.isDefEq\.respectTransparency false in\s+--"
)
LAKEFILE_OPTION = re.compile(
    r"^\s*⟨`backward\.isDefEq\.respectTransparency,\s*false⟩,?\s*\n",
    re.MULTILINE,
)


@dataclass
class FileResult:
    """Result of processing one file."""

    removable: int = 0  # lines we attempted to remove
    removed: int = 0  # successfully removed
    kept: int = 0  # had to keep (build failed)
    skipped: int = 0  # with trailing comments


@dataclass
class Summary:
    """Aggregated results."""

    total_files: int = 0
    files_fully_cleaned: int = 0
    files_partially_cleaned: int = 0
    files_unchanged: int = 0
    files_errored: int = 0
    total_removed: int = 0
    total_kept: int = 0
    total_skipped: int = 0
    duration: float = 0.0


class _RemoveDisplay(Display):
    """Progress display for the remove script."""

    def __init__(self):
        super().__init__()
        self.total_removed = 0
        self.total_kept = 0

    def _status_line(self) -> str:
        return (
            f"[{self.completed}/{self.total}]  "
            f"Working on: {self.inflight}  "
            f"Removed: {self.total_removed}  Kept: {self.total_kept}"
        )

    def on_module(self, module_name: str, result: FileResult | None, error: Exception | None):
        with self.lock:
            if result:
                self.total_removed += result.removed
                self.total_kept += result.kept
                if result.removed > 0 and result.kept == 0:
                    sym = "\u2713"
                elif result.removed > 0:
                    sym = "~"
                else:
                    sym = "\u00b7"
                parts = []
                if result.removed:
                    parts.append(f"-{result.removed}")
                if result.kept:
                    parts.append(f"kept {result.kept}")
                if result.skipped:
                    parts.append(f"skipped {result.skipped}")
                detail = " ".join(parts)
                self.messages.append(f"  {sym} {module_name} {detail}")
            elif error:
                self.messages.append(f"  ! {module_name}: {error}")
            self._redraw()


def handle_lakefile() -> bool:
    """Check and remove the option from lakefile.lean. Returns True if changed."""
    lakefile = PROJECT_DIR / "lakefile.lean"
    content = lakefile.read_text()
    if "backward.isDefEq.respectTransparency" not in content:
        return False
    new_content = LAKEFILE_OPTION.sub("", content)
    if new_content != content:
        lakefile.write_text(new_content)
        print("Removed backward.isDefEq.respectTransparency from lakefile.lean")
        return True
    return False


def scan_files(dag: DAG) -> dict[str, list[int]]:
    """Find files with removable set_option lines.

    Returns dict of module_name -> list of 0-indexed line numbers.
    """
    results: dict[str, list[int]] = {}
    for name, info in dag.modules.items():
        filepath = dag.project_root / info.filepath
        if not filepath.exists():
            continue
        lines = filepath.read_text().splitlines(keepends=True)
        removable = []
        for i, line in enumerate(lines):
            if SET_OPTION_WITH_COMMENT.match(line):
                continue
            if SET_OPTION_REMOVABLE.match(line):
                removable.append(i)
        if removable:
            results[name] = removable
    return results


def count_skipped(filepath: Path) -> int:
    """Count set_option lines with trailing comments."""
    count = 0
    for line in filepath.read_text().splitlines():
        if SET_OPTION_WITH_COMMENT.match(line):
            count += 1
    return count


PROGRESS_RE = re.compile(r"\[(\d+)/(\d+)\]")


def initial_build():
    """Run a full lake build so all .oleans are fresh.

    Without this, every worker would redundantly rebuild shared
    upstream modules.
    """
    print("Running initial build...", flush=True)
    proc = subprocess.Popen(
        ["lake", "build"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=PROJECT_DIR,
    )
    for line in proc.stdout:
        m = PROGRESS_RE.search(line)
        if m:
            print(f"\r  [{m.group(1)}/{m.group(2)}]", end="", flush=True)
    proc.wait()
    print()  # newline after progress
    if proc.returncode != 0:
        print("  (initial build had errors — continuing anyway)")


def make_process_file(
    removable_map: dict[str, list[int]],
    timeout: int,
) -> Callable:
    """Create the per-file action callback."""

    def process_file(module_name: str, filepath: Path) -> FileResult:
        abs_path = filepath
        removable_lines = removable_map.get(module_name, [])
        skipped = count_skipped(abs_path)

        if not removable_lines:
            return FileResult(skipped=skipped)

        original_text = abs_path.read_text()
        original = original_text.splitlines(keepends=True)

        inflight_register(abs_path, original_text)

        try:
            # Phase 1: try removing ALL at once
            removable_set = set(removable_lines)
            new_lines = [line for i, line in enumerate(original) if i not in removable_set]
            abs_path.write_text("".join(new_lines))

            ok, _ = lake_build(module_name, PROJECT_DIR, timeout)
            if ok:
                return FileResult(
                    removable=len(removable_lines),
                    removed=len(removable_lines),
                    skipped=skipped,
                )

            # Phase 2: revert, then one-at-a-time bottom-to-top
            abs_path.write_text(original_text)
            current = list(original)
            removed = 0
            kept = 0

            for line_idx in reversed(removable_lines):
                test = current[:line_idx] + current[line_idx + 1 :]
                abs_path.write_text("".join(test))

                ok, _ = lake_build(module_name, PROJECT_DIR, timeout)
                if ok:
                    current = test
                    removed += 1
                else:
                    abs_path.write_text("".join(current))
                    kept += 1

            return FileResult(
                removable=len(removable_lines),
                removed=removed,
                kept=kept,
                skipped=skipped,
            )
        except BaseException:
            abs_path.write_text(original_text)
            raise
        finally:
            inflight_unregister(abs_path)

    return process_file


def print_summary(summary: Summary):
    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Files processed:        {summary.total_files}")
    print(f"  Fully cleaned:          {summary.files_fully_cleaned}")
    print(f"  Partially cleaned:      {summary.files_partially_cleaned}")
    print(f"  Unchanged:              {summary.files_unchanged}")
    print(f"  Errors:                 {summary.files_errored}")
    print(f"  Lines removed:          {summary.total_removed}")
    print(f"  Lines kept:             {summary.total_kept}")
    print(f"  Lines skipped (comment):{summary.total_skipped}")
    print(f"  Duration:               {summary.duration:.0f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Remove set_option backward.isDefEq.respectTransparency false in"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without modifying files or building",
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
        help="Skip the initial lake build (assumes .oleans are already fresh)",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Step 1: lakefile
    if not args.dry_run:
        handle_lakefile()

    # Step 2: build DAG
    print("Building import DAG...", flush=True)
    full_dag = DAG.from_directories(PROJECT_DIR)
    print(f"  {len(full_dag.modules)} modules parsed")

    # Step 3: scan for removable lines
    print("Scanning for removable set_option lines...", flush=True)
    removable_map = scan_files(full_dag)

    if args.files:
        # Filter to requested files
        requested = set()
        for f in args.files:
            mod = f.replace("/", ".").removesuffix(".lean")
            requested.add(mod)
        removable_map = {k: v for k, v in removable_map.items() if k in requested}

    total_removable = sum(len(v) for v in removable_map.values())
    print(f"  {len(removable_map)} files with {total_removable} removable lines")

    if not removable_map:
        print("Nothing to do.")
        return

    # Step 4: initial build to ensure all .oleans are fresh
    if not args.dry_run and not args.no_initial:
        initial_build()

    # Step 5: traverse full DAG, skipping modules without removable lines.
    # We use the full DAG (not a subset) to ensure the backward traversal
    # respects all import edges.  Without this, two modules in the target
    # set that are connected through intermediate non-target modules could
    # be processed concurrently, causing `lake build` to race on rebuilding
    # shared upstream .oleans.
    target_modules = set(removable_map.keys())
    skip_modules = set(full_dag.modules.keys()) - target_modules

    # Weight = LOC × number of removable lines, so expensive files get
    # critical-path priority in the DAG traversal scheduler.
    weights = {}
    for name, removable_lines in removable_map.items():
        fp = full_dag.project_root / full_dag.modules[name].filepath
        loc = len(fp.read_text().splitlines())
        weights[name] = loc * len(removable_lines)

    if args.dry_run:
        sub_dag = full_dag.subset(target_modules)
        levels = sub_dag.levels_backward()
        print(f"\nBackward traversal: {len(levels)} levels")
        for i, level in enumerate(levels):
            count = sum(len(removable_map.get(m, [])) for m in level)
            print(f"  Level {i}: {len(level)} files, {count} removable lines")
        print(f"\nTotal: {len(removable_map)} files, {total_removable} lines")
        print("(dry run — no changes made)")
        return

    display = _RemoveDisplay()
    action = make_process_file(removable_map, args.timeout)

    display.start(len(full_dag.modules))
    try:
        results = traverse_dag(
            full_dag,
            action,
            direction="backward",
            max_workers=args.max_workers,
            progress_callback=display.on_progress,
            module_callback=display.on_module,
            skip=skip_modules,
            weights=weights,
        )
    except KeyboardInterrupt:
        display.stop()
        print("\nInterrupted. Press Ctrl-C again if the process doesn't exit.", flush=True)
        force_exit(1)
    finally:
        display.stop()

    # Step 6: summarize (only count target modules, not skipped ones)
    target_results = [tr for tr in results if tr.module_name in target_modules]
    summary = Summary(total_files=len(target_results), duration=time.time() - start_time)

    for tr in target_results:
        r: FileResult | None = tr.result
        if tr.error:
            summary.files_errored += 1
            continue
        if r is None:
            continue
        summary.total_removed += r.removed
        summary.total_kept += r.kept
        summary.total_skipped += r.skipped
        if r.removed > 0 and r.kept == 0:
            summary.files_fully_cleaned += 1
        elif r.removed > 0:
            summary.files_partially_cleaned += 1
        else:
            summary.files_unchanged += 1

    print_summary(summary)


if __name__ == "__main__":
    main()
