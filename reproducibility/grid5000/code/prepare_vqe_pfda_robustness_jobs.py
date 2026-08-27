#!/usr/bin/env python3
"""Write the immutable, load-balanced PFDA-BS robustness work list."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from vqe_pfda_robustness_protocol import jobs, serializable_protocol


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    protocol_path = root / "protocol.json"
    if not protocol_path.exists():
        protocol_path.write_text(
            json.dumps(serializable_protocol(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    rows = jobs()
    jobs_path = root / "jobs.tsv"
    with jobs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("variant", "molecule", "molecule_slug", "seed", "evaluations"),
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} jobs to {jobs_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
