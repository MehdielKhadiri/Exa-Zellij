#!/usr/bin/env python3
"""Summarize the compact quantum-aware VQE campaign."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def incumbent_at(trace: list[dict], budget: int, work_key: str, value_key: str):
    eligible = [row for row in trace if int(row[work_key]) <= budget]
    if not eligible:
        return None
    return float(eligible[-1][value_key])


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_trace(path: Path) -> list[dict]:
    if path.suffix == ".jsonl":
        return load_jsonl(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--pfda-root", type=Path)
    parser.add_argument("--checkpoints", nargs="+", type=int, default=[500, 1000, 2000, 5000])
    args = parser.parse_args()
    rows = []
    for path in sorted((args.root / "runs").glob("*/*/summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows.append({
            "molecule": data["molecule"],
            "method": data["method"],
            "n_qubits": data["n_qubits"],
            "n_params": data["n_params"],
            "candidate_domain": data.get("candidate_domain", "full-periodic"),
            "parameter_bounds": json.dumps(data.get("parameter_bounds", [-3.141592653589793, 3.141592653589793])),
            "quantum_work": data["quantum_work"],
            "energy_circuits": data["energy_circuits"],
            "metric_circuits": data["metric_circuits"],
            "best_energy": data["best_energy"],
            "fci_energy": data["fci_energy"],
            "absolute_error_ha": data["absolute_error_ha"],
            "chemical_accuracy": data["chemical_accuracy"],
            "walltime_s": data["walltime_s"],
        })
    if not rows:
        raise SystemExit(f"No summaries found below {args.root}")
    output = args.root / "quantum_aware_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("molecule method             work   best_energy       |E-FCI|       chem wall_s")
    for row in sorted(rows, key=lambda item: (item["molecule"], item["method"])):
        print(
            f"{row['molecule']:<8} {row['method']:<18} {row['quantum_work']:>5} "
            f"{row['best_energy']:>16.10f} {row['absolute_error_ha']:>12.4e} "
            f"{str(row['chemical_accuracy']):>5} {row['walltime_s']:>7.1f}"
        )
    print(f"wrote {output}")

    if args.pfda_root is None:
        return
    curve_rows = []
    fci_by_molecule = {row["molecule"]: float(row["fci_energy"]) for row in rows}
    for molecule, fci_energy in sorted(fci_by_molecule.items()):
        pfda_trace_path = args.pfda_root / "runs" / "pfda-bs" / molecule / "seed_0" / "trace.json"
        if not pfda_trace_path.is_file():
            print(f"warning: missing PFDA-BS trace {pfda_trace_path}")
            continue
        traces = {"pfda-bs": (load_trace(pfda_trace_path), "evaluation", "best_value")}
        for method in ("ps-adam", "qng-diag", "excitation-solve"):
            trace_path = args.root / "runs" / method / molecule / "trace.jsonl"
            if trace_path.is_file():
                traces[method] = (load_trace(trace_path), "quantum_work", "best_energy")
        for budget in args.checkpoints:
            for method, (trace, work_key, value_key) in traces.items():
                energy = incumbent_at(trace, budget, work_key, value_key)
                if energy is None:
                    continue
                curve_rows.append({
                    "molecule": molecule,
                    "method": method,
                    "quantum_work_cap": budget,
                    "best_energy": energy,
                    "fci_energy": fci_energy,
                    "absolute_error_ha": abs(energy - fci_energy),
                })
    curve_output = args.root / "matched_quantum_work_curves.csv"
    if curve_rows:
        with curve_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
            writer.writeheader()
            writer.writerows(curve_rows)
        print(f"wrote {curve_output}")


if __name__ == "__main__":
    main()
