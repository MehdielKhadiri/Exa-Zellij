#!/usr/bin/env python3
"""Validate and summarize the PFDA-BS equal-evaluation VQE campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


METHODS = ("pfda-bs", "cobyla", "l-bfgs-b", "spsa", "pso", "sobol")
DEFAULT_MOLECULES = ("HeH+", "H3+", "H4")
CHEMICAL_ACCURACY_HA = 1.5936e-3


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else math.nan
        return value, value
    cuts = statistics.quantiles(values, n=4, method="inclusive")
    return float(cuts[0]), float(cuts[2])


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--molecules", nargs="+", default=list(DEFAULT_MOLECULES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument(
        "--pfda-single-seed",
        action="store_true",
        help="Expect deterministic PFDA-BS only for the first requested seed.",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    molecules = tuple(args.molecules)
    seeds = tuple(args.seeds)
    summaries = []
    for path in sorted(root.glob("runs/*/*/seed_*/summary.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["_path"] = str(path.relative_to(root))
        summaries.append(row)

    expected_seeds = {
        method: (seeds[:1] if args.pfda_single_seed and method == "pfda-bs" else seeds)
        for method in METHODS
    }
    expected = sum(len(expected_seeds[method]) for method in METHODS) * len(molecules)
    if not summaries:
        raise SystemExit(f"no summaries found under {root}")

    seen: dict[tuple[str, str, int], dict] = {}
    errors: list[str] = []
    for row in summaries:
        key = (str(row["method"]), str(row["molecule"]), int(row["seed"]))
        if key in seen:
            errors.append(f"duplicate run {key}: {row['_path']}")
        seen[key] = row
        matched = int(row["matched_evaluations"])
        used = int(row["evaluations_used"])
        if used > matched:
            errors.append(f"budget exceeded for {key}: {used}>{matched}")
        if row["method"] == "pfda-bs" and used != matched:
            errors.append(f"PFDA count mismatch for {key}: {used}!={matched}")

    missing = [
        (method, molecule, seed)
        for method in METHODS
        for molecule in molecules
        for seed in expected_seeds[method]
        if (method, molecule, seed) not in seen
    ]

    long_rows: list[dict[str, object]] = []
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in summaries:
        matched = int(row["matched_evaluations"])
        for checkpoint_text, energy in row["checkpoints"].items():
            checkpoint = int(checkpoint_text)
            item = {
                "molecule": row["molecule"],
                "method": row["method"],
                "seed": row["seed"],
                "dimension": row["dimension"],
                "n_qubits": row["n_qubits"],
                "matched_evaluations": matched,
                "checkpoint": checkpoint,
                "checkpoint_fraction": checkpoint / matched,
                "evaluations_used": row["evaluations_used"],
                "energy": float(energy),
                "fci_energy": row["fci_energy"],
                "absolute_error": abs(float(energy) - float(row["fci_energy"])),
                "wall_time_s": row["wall_time_s"],
            }
            long_rows.append(item)
            groups[(str(row["molecule"]), str(row["method"]), checkpoint)].append(item)

    aggregate_rows: list[dict[str, object]] = []
    for (molecule, method, checkpoint), items in sorted(groups.items()):
        energies = [float(item["energy"]) for item in items]
        errors_ha = [float(item["absolute_error"]) for item in items]
        q1, q3 = quartiles(errors_ha)
        aggregate_rows.append(
            {
                "molecule": molecule,
                "method": method,
                "checkpoint": checkpoint,
                "runs": len(items),
                "median_energy": median(energies),
                "median_abs_error_ha": median(errors_ha),
                "q1_abs_error_ha": q1,
                "q3_abs_error_ha": q3,
                "chemical_accuracy_runs": sum(
                    error <= CHEMICAL_ACCURACY_HA for error in errors_ha
                ),
            }
        )

    pairwise_rows: list[dict[str, object]] = []
    for molecule in molecules:
        pfda = {
            int(row["seed"]): float(row["best_energy"])
            for row in summaries
            if row["molecule"] == molecule and row["method"] == "pfda-bs"
        }
        for method in METHODS[1:]:
            rival = {
                int(row["seed"]): float(row["best_energy"])
                for row in summaries
                if row["molecule"] == molecule and row["method"] == method
            }
            common = sorted(set(pfda) & set(rival))
            tolerance = 1e-7
            wins = sum(pfda[seed] < rival[seed] - tolerance for seed in common)
            ties = sum(abs(pfda[seed] - rival[seed]) <= tolerance for seed in common)
            losses = len(common) - wins - ties
            pairwise_rows.append(
                {
                    "molecule": molecule,
                    "baseline": method,
                    "paired_seeds": len(common),
                    "pfda_wins": wins,
                    "ties": ties,
                    "pfda_losses": losses,
                }
            )

    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_csv(analysis / "runs_long.csv", list(long_rows[0]), long_rows)
    write_csv(analysis / "aggregate.csv", list(aggregate_rows[0]), aggregate_rows)
    write_csv(analysis / "pairwise.csv", list(pairwise_rows[0]), pairwise_rows)

    report = [
        "# PFDA-BS equal-evaluation VQE campaign",
        "",
        f"- Recorded runs: {len(summaries)}/{expected}",
        f"- Missing runs: {len(missing)}",
        f"- Validation errors: {len(errors)}",
        "- PFDA-BS is configured structurally; comparator caps equal its calculated call count.",
        (
            "- PFDA-BS uses one run because this execution path is deterministic; "
            "stochastic baselines use every requested seed."
            if args.pfda_single_seed
            else "- Every method uses every requested seed."
        ),
        "",
        "## Final median absolute error (Hartree)",
        "",
        "| Molecule | Method | Calls | Runs | Median | IQR | Chemical accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    final_by_molecule: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in aggregate_rows:
        max_checkpoint = max(
            int(candidate["checkpoint"])
            for candidate in aggregate_rows
            if candidate["molecule"] == row["molecule"]
            and candidate["method"] == row["method"]
        )
        if int(row["checkpoint"]) == max_checkpoint:
            final_by_molecule[str(row["molecule"])].append(row)
    for molecule in molecules:
        for row in sorted(
            final_by_molecule[molecule], key=lambda value: float(value["median_abs_error_ha"])
        ):
            report.append(
                f"| {molecule} | {row['method']} | {row['checkpoint']} | {row['runs']} | "
                f"{float(row['median_abs_error_ha']):.6g} | "
                f"[{float(row['q1_abs_error_ha']):.6g}, {float(row['q3_abs_error_ha']):.6g}] | "
                f"{row['chemical_accuracy_runs']}/{row['runs']} |"
            )
    if missing:
        report.extend(["", "## Missing", "", *[f"- {item}" for item in missing]])
    if errors:
        report.extend(["", "## Validation errors", "", *[f"- {item}" for item in errors]])
    (analysis / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    status = {
        "recorded": len(summaries),
        "expected": expected,
        "missing": len(missing),
        "validation_errors": len(errors),
    }
    (analysis / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, sort_keys=True))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
