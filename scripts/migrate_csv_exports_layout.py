#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import re
import shutil
from typing import Optional, Tuple


def _infer_base_name(filename: str) -> Optional[str]:
    """Infer the model base name from an exported CSV filename.

    Supports:
    - <base>_results.csv
    - <base>_metrics.csv
    - <base>_foldN_results.csv

    Returns the inferred <base> or None if not recognized.
    """

    if not filename.endswith(".csv"):
        return None

    if filename.endswith("_results.csv"):
        base = filename[: -len("_results.csv")]
        # fold results: <base>_foldN_results.csv
        m = re.match(r"^(?P<base>.+)_fold\d+$", base)
        if m:
            return m.group("base")
        return base

    if filename.endswith("_metrics.csv"):
        base = filename[: -len("_metrics.csv")]
        # fold metrics: <base>_foldN_metrics.csv
        m = re.match(r"^(?P<base>.+)_fold\d+$", base)
        if m:
            return m.group("base")
        return base

    return None


def _move_file(src: str, dest: str) -> Tuple[bool, str]:
    """Move src -> dest, creating parent directories.

    Returns (moved, message).
    """

    os.makedirs(os.path.dirname(dest), exist_ok=True)

    if os.path.abspath(src) == os.path.abspath(dest):
        return False, "skip (already in place)"

    if os.path.exists(dest):
        return False, f"skip (dest exists: {dest})"

    shutil.move(src, dest)
    return True, "moved"


def migrate(csv_exports_dir: str, dry_run: bool = False) -> int:
    if not os.path.isdir(csv_exports_dir):
        raise FileNotFoundError(f"csv_exports dir not found: {csv_exports_dir}")

    moved_count = 0
    skipped_count = 0

    for entry in sorted(os.listdir(csv_exports_dir)):
        src_path = os.path.join(csv_exports_dir, entry)

        if not os.path.isfile(src_path):
            continue

        base_name = _infer_base_name(entry)
        if base_name is None:
            skipped_count += 1
            continue

        dest_dir = os.path.join(csv_exports_dir, base_name)
        dest_path = os.path.join(dest_dir, entry)

        if dry_run:
            print(f"DRY-RUN: {src_path} -> {dest_path}")
            moved_count += 1
            continue

        moved, msg = _move_file(src_path, dest_path)
        if moved:
            moved_count += 1
            print(f"OK: {entry} -> {base_name}/ ({msg})")
        else:
            skipped_count += 1
            print(f"SKIP: {entry} ({msg})")

    print()
    print(f"Done. moved={moved_count} skipped={skipped_count}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Move flat csv_exports/*.csv files into csv_exports/<model>/ subfolders. "
            "Useful for reorganizing already-exported experiments without rerunning."
        )
    )
    parser.add_argument(
        "path",
        help=(
            "Path to an experiment directory that contains csv_exports/ (e.g. results/experiments/<exp>/), "
            "or directly to csv_exports/"
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without moving files")

    args = parser.parse_args()

    path = os.path.abspath(args.path)
    csv_exports_dir = path
    if os.path.basename(csv_exports_dir) != "csv_exports":
        csv_exports_dir = os.path.join(path, "csv_exports")

    return migrate(csv_exports_dir=csv_exports_dir, dry_run=args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
