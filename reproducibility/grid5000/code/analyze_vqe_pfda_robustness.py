#!/usr/bin/env python3
"""Validate and summarize the PFDA-BS hyperparameter robustness campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from vqe_pfda_robustness_protocol import MOLECULES, SEEDS, VARIANTS, VARIANT_BY_NAME


CHEMICAL_ACCURACY_HA = 1.5936e-3
ERROR_FLOOR = 1e-12


def median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else math.nan


def quartiles(values: list[float]) -> tuple[float, float]:
    if len(values) < 2:
        value = values[0] if values else math.nan
        return value, value
    q = statistics.quantiles(values, n=4, method="inclusive")
    return float(q[0]), float(q[2])


def error_at(trace: list[dict[str, object]], fci: float, calls: int) -> float:
    if len(trace) < calls:
        return math.nan
    return abs(float(trace[calls - 1]["best_value"]) - fci)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()

    records: list[dict[str, object]] = []
    validation_errors: list[str] = []
    seen: set[tuple[str, str, int]] = set()
    for summary_path in sorted(root.glob("runs/*/*/seed_*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        variant = summary_path.parts[-4]
        molecule = str(summary["molecule"])
        seed = int(summary["seed"])
        key = (variant, molecule, seed)
        if key in seen:
            validation_errors.append(f"duplicate: {key}")
            continue
        seen.add(key)
        if variant not in VARIANT_BY_NAME:
            validation_errors.append(f"unknown variant: {variant}")
            continue
        trace_path = summary_path.parent / "trace.json"
        if not trace_path.exists():
            validation_errors.append(f"missing trace: {trace_path.relative_to(root)}")
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        used = int(summary["evaluations_used"])
        expected = VARIANT_BY_NAME[variant].calls(int(summary["dimension"]))
        if used != expected or len(trace) != expected:
            validation_errors.append(
                f"count mismatch {key}: used={used}, trace={len(trace)}, expected={expected}"
            )
        records.append(
            {
                "variant": variant,
                "molecule": molecule,
                "seed": seed,
                "dimension": int(summary["dimension"]),
                "n_qubits": int(summary["n_qubits"]),
                "evaluations": used,
                "best_energy": float(summary["best_energy"]),
                "fci_energy": float(summary["fci_energy"]),
                "final_abs_error_ha": float(summary["absolute_error"]),
                "wall_time_s": float(summary["wall_time_s"]),
                "trace": trace,
            }
        )

    expected_keys = {
        (variant.name, molecule, seed)
        for variant in VARIANTS
        for molecule in MOLECULES
        for seed in SEEDS
    }
    missing = sorted(expected_keys - seen)
    if not records:
        raise SystemExit(f"no robustness summaries found under {root}")

    # The fixed-call checkpoint is determined from the declared protocol, not
    # from whatever subset happens to have completed.
    common_calls = {
        molecule: min(
            variant.calls(
                next(
                    int(record["dimension"])
                    for record in records
                    if record["molecule"] == molecule
                )
            )
            for variant in VARIANTS
        )
        for molecule in MOLECULES
        if any(record["molecule"] == molecule for record in records)
    }

    long_rows: list[dict[str, object]] = []
    for record in records:
        trace = record.pop("trace")
        molecule = str(record["molecule"])
        calls = common_calls[molecule]
        fixed_error = error_at(trace, float(record["fci_energy"]), calls)
        running_log_errors = [
            math.log10(abs(float(item["best_value"]) - float(record["fci_energy"])) + ERROR_FLOOR)
            for item in trace[:calls]
        ]
        variant = VARIANT_BY_NAME[str(record["variant"])]
        long_rows.append(
            {
                **record,
                "factor": variant.factor,
                "level": variant.level,
                "reference_variant": variant.reference,
                "common_calls": calls,
                "fixed_call_abs_error_ha": fixed_error,
                "mean_log10_error_to_common": sum(running_log_errors) / len(running_log_errors),
            }
        )

    aggregate_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in long_rows:
        grouped[(str(row["variant"]), str(row["molecule"]))].append(row)
    for (variant_name, molecule), items in sorted(grouped.items()):
        final_errors = [float(item["final_abs_error_ha"]) for item in items]
        fixed_errors = [float(item["fixed_call_abs_error_ha"]) for item in items]
        q1, q3 = quartiles(fixed_errors)
        variant = VARIANT_BY_NAME[variant_name]
        aggregate_rows.append(
            {
                "variant": variant_name,
                "factor": variant.factor,
                "level": variant.level,
                "molecule": molecule,
                "runs": len(items),
                "native_evaluations": int(items[0]["evaluations"]),
                "common_calls": int(items[0]["common_calls"]),
                "median_fixed_call_error_ha": median(fixed_errors),
                "q1_fixed_call_error_ha": q1,
                "q3_fixed_call_error_ha": q3,
                "median_native_final_error_ha": median(final_errors),
                "chemical_accuracy_at_common": sum(
                    value <= CHEMICAL_ACCURACY_HA for value in fixed_errors
                ),
                "chemical_accuracy_native_final": sum(
                    value <= CHEMICAL_ACCURACY_HA for value in final_errors
                ),
                "median_wall_time_s": median([float(item["wall_time_s"]) for item in items]),
            }
        )

    by_key = {
        (str(row["variant"]), str(row["molecule"]), int(row["seed"])): row
        for row in long_rows
    }
    effect_rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        if variant.name == "default":
            continue
        deltas: list[float] = []
        wins = ties = losses = 0
        for molecule in MOLECULES:
            for seed in SEEDS:
                current = by_key.get((variant.name, molecule, seed))
                reference = by_key.get((variant.reference, molecule, seed))
                if current is None or reference is None:
                    continue
                current_error = float(current["fixed_call_abs_error_ha"])
                reference_error = float(reference["fixed_call_abs_error_ha"])
                delta = math.log10(current_error + ERROR_FLOOR) - math.log10(reference_error + ERROR_FLOOR)
                deltas.append(delta)
                if delta < -1e-9:
                    wins += 1
                elif delta > 1e-9:
                    losses += 1
                else:
                    ties += 1
        med_delta = median(deltas)
        effect_rows.append(
            {
                "variant": variant.name,
                "factor": variant.factor,
                "level": variant.level,
                "reference_variant": variant.reference,
                "paired_conditions": len(deltas),
                "median_delta_log10_error": med_delta,
                "median_error_ratio": 10.0 ** med_delta if math.isfinite(med_delta) else math.nan,
                "better": wins,
                "ties": ties,
                "worse": losses,
            }
        )

    analysis = root / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_csv(analysis / "runs.csv", long_rows)
    write_csv(analysis / "aggregate.csv", aggregate_rows)
    write_csv(analysis / "paired_effects.csv", effect_rows)

    report = [
        "# PFDA-BS hyperparameter robustness",
        "",
        f"- Complete runs: {len(records)}/{len(expected_keys)}",
        f"- Missing runs: {len(missing)}",
        f"- Validation errors: {len(validation_errors)}",
        "- Canonical hypersphere geometry is held fixed; changing geometry would define another FBD algorithm.",
        "- Native-final quality is reported together with fixed-call quality to separate tuning effects from extra objective evaluations.",
        "",
        "## Paired fixed-call effects",
        "",
        "A ratio below 1 favors the variant; a ratio above 1 favors its declared reference.",
        "",
        "| Factor | Level | Reference | Pairs | Median error ratio | Better / tie / worse |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in effect_rows:
        report.append(
            f"| {row['factor']} | {row['level']} | {row['reference_variant']} | "
            f"{row['paired_conditions']} | {float(row['median_error_ratio']):.4g} | "
            f"{row['better']} / {row['ties']} / {row['worse']} |"
        )
    if missing:
        report.extend(["", "## Missing runs", "", *[f"- {item}" for item in missing]])
    if validation_errors:
        report.extend(
            ["", "## Validation errors", "", *[f"- {item}" for item in validation_errors]]
        )
    (analysis / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    status = {
        "protocol": "PFDA-BS-hyperparameter-robustness-v1",
        "recorded": len(records),
        "expected": len(expected_keys),
        "missing": len(missing),
        "validation_errors": len(validation_errors),
        "common_calls": common_calls,
    }
    (analysis / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(status, sort_keys=True))
    if validation_errors:
        return 2
    return 0 if not missing else 3


if __name__ == "__main__":
    raise SystemExit(main())
