from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.submission import freeze_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("experiments"))
    parser.add_argument("--winner-run-id", required=True)
    parser.add_argument("--comparator-run-id", required=True)
    args = parser.parse_args()
    print(freeze_selection(args.artifact_root, args.winner_run_id, args.comparator_run_id))


if __name__ == "__main__":
    main()
