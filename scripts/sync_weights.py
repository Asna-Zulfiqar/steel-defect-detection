"""Sync/copy the best trained checkpoint into the canonical models/ directory.

Usage:
    python scripts/sync_weights.py

This looks for `best.pt` under any `runs/**/weights/` path and copies the first
match to `models/best.pt`. If `models/` does not exist it will be created.
"""
from pathlib import Path
import shutil
import sys


def find_best_in_runs(root: Path) -> Path | None:
    matches = list(root.glob("**/weights/best.pt"))
    return matches[0] if matches else None


def main():
    repo_root = Path(__file__).resolve().parent.parent
    runs_dir = repo_root / "runs"
    dest_dir = repo_root / "models"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "best.pt"

    if not runs_dir.exists():
        print(f"No runs/ directory found at {runs_dir}. Nothing to do.")
        sys.exit(1)

    best = find_best_in_runs(runs_dir)
    if best is None:
        print("No best.pt found under runs/**/weights/. Make sure training completed.")
        sys.exit(1)

    shutil.copy(best, dest)
    print(f"Copied {best} -> {dest}")


if __name__ == "__main__":
    main()

