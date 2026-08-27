#!/usr/bin/env python3
"""Verify the curated Grid'5000 and LUMI-G result artifacts.

This script uses only the Python standard library.  It recomputes the central
paper-facing counts and checks internal numerical consistency without running
the expensive VQE objective again.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TIE_TOLERANCE_HA = 1.0e-7


def read_csv(relative: str) -> list[dict[str, str]]:
    path = ROOT / relative
    if not path.is_file():
        raise AssertionError(f"missing artifact: {relative}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def compare(candidate: float, reference: float) -> str:
    if candidate < reference - TIE_TOLERANCE_HA:
        return "win"
    if candidate > reference + TIE_TOLERANCE_HA:
        return "loss"
    return "tie"


def verify_black_box() -> dict[str, int]:
    rows = read_csv("grid5000/results/black_box_equal_call/runs_long.csv")
    molecules = ("HeH+", "H3+", "H4", "H5", "HF", "NeH+", "OH-", "LiH")
    baselines = ("cobyla", "l-bfgs-b", "spsa", "pso", "sobol")
    final: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if row["molecule"] not in molecules:
            continue
        if not math.isclose(float(row["checkpoint_fraction"]), 1.0):
            continue
        final[(row["molecule"], row["method"])].append(float(row["absolute_error"]))

    expected_methods = ("pfda-bs",) + baselines
    for molecule in molecules:
        for method in expected_methods:
            if not final[(molecule, method)]:
                raise AssertionError(f"missing final black-box result: {molecule}/{method}")

    outcomes = {"wins": 0, "ties": 0, "losses": 0}
    by_baseline: dict[str, dict[str, int]] = {}
    for baseline in baselines:
        counts = {"wins": 0, "ties": 0, "losses": 0}
        for molecule in molecules:
            pfda = sorted(final[(molecule, "pfda-bs")])[len(final[(molecule, "pfda-bs")]) // 2]
            values = sorted(final[(molecule, baseline)])
            middle = len(values) // 2
            baseline_median = values[middle] if len(values) % 2 else 0.5 * (values[middle - 1] + values[middle])
            result = compare(pfda, baseline_median)
            key = {"win": "wins", "tie": "ties", "loss": "losses"}[result]
            counts[key] += 1
            outcomes[key] += 1
        by_baseline[baseline] = counts

    assert outcomes == {"wins": 25, "ties": 14, "losses": 1}, outcomes
    expected = read_csv("grid5000/results/black_box_equal_call/paper_outcomes.csv")
    expected_total = next(row for row in expected if row["comparator"] == "total")
    assert int(expected_total["pfda_wins"]) == outcomes["wins"]
    assert int(expected_total["ties"]) == outcomes["ties"]
    assert int(expected_total["pfda_losses"]) == outcomes["losses"]
    return outcomes


def verify_knowledge_driven() -> dict[str, int]:
    rows = read_csv("grid5000/results/knowledge_driven/paper_final_errors.csv")
    assert len(rows) == 8
    outcomes = {"wins": 0, "ties": 0, "losses": 0}
    for row in rows:
        pfda = float(row["pfda_bs"])
        for key in ("ps_adam", "qng_diag", "excitation_solve"):
            result = compare(pfda, float(row[key]))
            key = {"win": "wins", "tie": "ties", "loss": "losses"}[result]
            outcomes[key] += 1
    assert outcomes == {"wins": 8, "ties": 15, "losses": 1}, outcomes

    raw = read_csv("grid5000/results/knowledge_driven/quantum_aware_summary.csv")
    raw_index = {(row["molecule"], row["method"]): float(row["absolute_error_ha"]) for row in raw}
    method_columns = {
        "ps-adam": "ps_adam",
        "qng-diag": "qng_diag",
        "excitation-solve": "excitation_solve",
    }
    # Every locally retained raw condition must agree with the rounded paper table.
    for (molecule, method), value in raw_index.items():
        paper = next(row for row in rows if row["molecule"] == molecule)
        rounded = float(paper[method_columns[method]])
        assert math.isclose(value, rounded, rel_tol=2e-6, abs_tol=5e-10), (
            molecule, method, value, rounded
        )
    return outcomes


def verify_robustness() -> dict[str, int]:
    status_path = ROOT / "local/results/robustness/status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status["expected"] == status["recorded"] == 195
    assert status["missing"] == status["validation_errors"] == 0

    runs = read_csv("local/results/robustness/runs.csv")
    effects = read_csv("local/results/robustness/paired_effects.csv")
    assert len(runs) == 195
    assert len(effects) == 12
    plateau = sum(0.9893 - 1e-12 <= float(row["median_error_ratio"]) <= 1.0 + 1e-12 for row in effects)
    assert plateau == 11
    depth_one = next(row for row in effects if row["variant"] == "depth_1")
    assert (int(depth_one["better"]), int(depth_one["ties"]), int(depth_one["worse"])) == (15, 0, 0)
    assert math.isclose(float(depth_one["median_error_ratio"]), 0.4597911024593828)
    no_ils = next(row for row in effects if row["variant"] == "ils_0")
    assert int(no_ils["worse"]) == 5
    return {"runs": len(runs), "perturbations": len(effects), "plateau": plateau}


def verify_lumi_scaling() -> dict[str, float]:
    rows = read_csv("lumi_g/results/strong_scaling.csv")
    assert len(rows) == 4
    for row in rows:
        reference_time = float(row["reference_time_s"])
        target_time = float(row["target_time_s"])
        reported_speedup = float(row["speedup"])
        ratio = reference_time / target_time
        assert math.isclose(ratio, reported_speedup, rel_tol=2.5e-3)
        scale = int(row["target_nodes"]) / int(row["reference_nodes"])
        efficiency = 100.0 * reported_speedup / scale
        assert math.isclose(efficiency, float(row["efficiency_percent"]), abs_tol=0.05)
        assert int(row["target_gpus"]) == 8 * int(row["target_nodes"])
    endpoint = next(row for row in rows if row["configuration"] == "beta10k_iter100")
    assert int(endpoint["target_gpus"]) == 8000
    assert math.isclose(float(endpoint["efficiency_percent"]), 60.98)
    return {"endpoint_gpus": 8000, "endpoint_efficiency_percent": 60.98}


def verify_lumi_quality() -> dict[str, int]:
    rows = read_csv("lumi_g/results/solution_quality.csv")
    assert len(rows) == 9
    for row in rows:
        best = float(row["best_energy_ha"])
        fci = float(row["fci_energy_ha"])
        absolute = float(row["absolute_error_ha"])
        relative = float(row["relative_error"])
        assert math.isclose(abs(best - fci), absolute, rel_tol=1.5e-3, abs_tol=5e-9)
        assert math.isclose(absolute / abs(fci), relative, rel_tol=1.5e-3)
    return {"molecules": len(rows)}


def verify_lumi_raw_evidence() -> dict[str, int]:
    evidence = read_csv("lumi_g/raw_evidence/raw_log_evidence.csv")
    assert len(evidence) == 16
    indexed = {row["evidence_id"]: row for row in evidence}
    assert len(indexed) == len(evidence)
    for row in evidence:
        digest = row["source_sha256"]
        assert len(digest) == 64 and all(char in "0123456789abcdef" for char in digest)
        assert int(row["gpus"]) == 8 * int(row["nodes"])
        assert row["paper_support"] in {"exact", "energy_only", "scheduler_only"}

    scaling = {
        row["configuration"]: row
        for row in read_csv("lumi_g/results/strong_scaling.csv")
    }
    time_checks = {
        "scaling_beta1k_iter1_ref": ("beta1k_iter1", "reference_time_s"),
        "scaling_beta1k_iter1_target": ("beta1k_iter1", "target_time_s"),
        "scaling_beta10k_iter10_ref": ("beta10k_iter10", "reference_time_s"),
        "scaling_beta10k_iter10_target": ("beta10k_iter10", "target_time_s"),
        "scaling_beta10k_iter100_ref": ("beta10k_iter100", "reference_time_s"),
    }
    for evidence_id, (configuration, field) in time_checks.items():
        observed = float(indexed[evidence_id]["internal_time_s"])
        published = float(scaling[configuration][field])
        assert math.isclose(observed, published, abs_tol=0.01)

    scheduler_endpoint = indexed["scaling_beta10k_iter100_target_scheduler"]
    assert scheduler_endpoint["paper_support"] == "scheduler_only"
    assert int(scheduler_endpoint["nodes"]) == 1000
    assert int(scheduler_endpoint["gpus"]) == 8000

    quality = {
        row["molecule"]: row
        for row in read_csv("lumi_g/results/solution_quality.csv")
    }
    exact_quality = ("HeH+", "H3+", "H4", "BH3", "CH4")
    for molecule in exact_quality:
        observed = indexed[f"quality_{molecule}"]
        published = quality[molecule]
        assert math.isclose(
            float(observed["best_energy_ha"]),
            float(published["best_energy_ha"]),
            abs_tol=5e-11,
        )
        assert math.isclose(
            float(observed["internal_time_s"]),
            float(published["wall_time_s"]),
            abs_tol=1.0,
        )

    neh = indexed["quality_NeH+"]
    assert neh["paper_support"] == "energy_only"
    assert math.isclose(
        float(neh["best_energy_ha"]),
        float(quality["NeH+"]["best_energy_ha"]),
        abs_tol=5e-10,
    )
    lih = indexed["quality_LiH"]
    assert lih["paper_support"] == "energy_only"
    assert math.isclose(
        float(lih["best_energy_ha"]),
        float(quality["LiH"]["best_energy_ha"]),
        abs_tol=5e-11,
    )
    scheduler_quality = (
        "quality_OH-_scheduler",
        "quality_HF_scheduler",
        "quality_LiH_scheduler",
    )
    for evidence_id in scheduler_quality:
        assert indexed[evidence_id]["paper_support"] == "scheduler_only"
    return {
        "indexed_evidence_records": len(evidence),
        "unique_source_files": len({row["source_sha256"] for row in evidence}),
        "paper_scaling_times_backed": len(time_checks),
        "paper_quality_energies_backed": len(exact_quality) + 2,
    }


def verify_checksums() -> int:
    manifest = ROOT / "SHA256SUMS"
    if not manifest.is_file():
        raise AssertionError("missing reproducibility/SHA256SUMS")
    checked = 0
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = ROOT / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == digest, f"checksum mismatch: {relative}"
        checked += 1
    return checked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-checksums", action="store_true")
    parser.add_argument(
        "--scopes",
        default="grid5000,local,lumi_g",
        help="comma-separated verification scopes: grid5000, local, lumi_g",
    )
    args = parser.parse_args()
    scopes = {scope.strip() for scope in args.scopes.split(",") if scope.strip()}
    allowed_scopes = {"grid5000", "local", "lumi_g"}
    unknown_scopes = scopes - allowed_scopes
    if unknown_scopes:
        parser.error(f"unknown verification scope(s): {', '.join(sorted(unknown_scopes))}")
    report = {}
    if "grid5000" in scopes:
        report["black_box_equal_call"] = verify_black_box()
        report["knowledge_driven"] = verify_knowledge_driven()
    if "local" in scopes:
        report["robustness"] = verify_robustness()
    if "lumi_g" in scopes:
        report["lumi_g_scaling"] = verify_lumi_scaling()
        report["lumi_g_solution_quality"] = verify_lumi_quality()
        report["lumi_g_raw_evidence"] = verify_lumi_raw_evidence()
    if not args.skip_checksums:
        report["checksummed_files"] = verify_checksums()
    print(json.dumps(report, indent=2, sort_keys=True))
    print("reproducibility verification: OK")


if __name__ == "__main__":
    main()
