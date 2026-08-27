#!/usr/bin/env python3
"""Same-ansatz VQE benchmark against quantum-aware optimizers.

The experiment keeps the molecule, Hamiltonian, full UCCSD ansatz, initial
point, and requested candidate domain fixed.  Its stopping variable is
*quantum work*:
one full-Hamiltonian energy evaluation or one auxiliary metric-tensor circuit
counts as one normalized unit, with both ledgers also reported separately.
This prevents gradient and natural-gradient iterations from hiding the
additional quantum work required by a QPU implementation.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize_scalar


METHODS = ("ps-adam", "qng-diag", "excitation-solve")
LOWER = -math.pi
UPPER = math.pi
CHEMICAL_ACCURACY_HA = 1.6e-3


def candidate_projector(
    lower: float, upper: float, *, periodic: bool
) -> Callable[[np.ndarray], np.ndarray]:
    """Return the candidate projection used by every compared optimizer."""

    if not lower < upper:
        raise ValueError("candidate bounds must satisfy lower < upper")

    def project(theta: np.ndarray) -> np.ndarray:
        x = np.asarray(theta, dtype=np.float64)
        if periodic:
            width = upper - lower
            return (x - lower) % width + lower
        return np.clip(x, lower, upper)

    return project


class QuantumWorkLimit(RuntimeError):
    pass


class QuantumWorkOracle:
    """Count energy and metric circuits separately under one work cap."""

    def __init__(
        self,
        evaluate_batch: Callable[[np.ndarray], np.ndarray],
        dimension: int,
        work_budget: int,
        project_candidate: Callable[[np.ndarray], np.ndarray],
    ) -> None:
        self._evaluate_batch = evaluate_batch
        self.dimension = int(dimension)
        self.work_budget = int(work_budget)
        self._project_candidate = project_candidate
        self.energy_circuits = 0
        self.metric_circuits = 0
        self.best_energy = math.inf
        self.best_parameters = np.zeros(self.dimension, dtype=np.float64)
        self.trace: list[dict[str, float | int | str]] = []

    @property
    def quantum_work(self) -> int:
        return self.energy_circuits + self.metric_circuits

    @property
    def remaining(self) -> int:
        return self.work_budget - self.quantum_work

    def require(self, units: int) -> None:
        if units < 0 or units > self.remaining:
            raise QuantumWorkLimit(
                f"request for {units} work units exceeds remaining {self.remaining}"
            )

    def energies(
        self,
        points: np.ndarray,
        *,
        role: str,
        candidate: bool = False,
    ) -> np.ndarray:
        x = np.asarray(points, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(f"expected (B,{self.dimension}) points, got {x.shape}")
        self.require(len(x))
        values = np.asarray(self._evaluate_batch(x), dtype=np.float64).reshape(-1)
        if len(values) != len(x) or not np.all(np.isfinite(values)):
            raise RuntimeError("energy evaluator returned invalid values")
        self.energy_circuits += len(x)

        if candidate:
            if len(x) != 1:
                raise ValueError("candidate evaluation must contain exactly one point")
            value = float(values[0])
            point = self._project_candidate(x[0])
            if value < self.best_energy:
                self.best_energy = value
                self.best_parameters = point.copy()
            self.trace.append(
                {
                    "quantum_work": self.quantum_work,
                    "energy_circuits": self.energy_circuits,
                    "metric_circuits": self.metric_circuits,
                    "candidate_energy": value,
                    "best_energy": self.best_energy,
                    "role": role,
                }
            )
        return values

    def consume_metric(self, circuits: int) -> None:
        self.require(circuits)
        self.metric_circuits += int(circuits)


def generalized_excitation_gradient(
    oracle: QuantumWorkOracle, theta: np.ndarray
) -> np.ndarray:
    """Exact four-term shift gradient for frequencies {1/2, 1}.

    PennyLane SingleExcitation and DoubleExcitation gates both expose these
    two frequencies.  The four shifted energy circuits are evaluated in one
    batch for every UCCSD parameter.
    """
    theta = np.asarray(theta, dtype=np.float64)
    d = len(theta)
    points = np.repeat(theta[None, :], 4 * d, axis=0)
    for i in range(d):
        points[4 * i + 0, i] += math.pi / 2.0
        points[4 * i + 1, i] -= math.pi / 2.0
        points[4 * i + 2, i] += math.pi
        points[4 * i + 3, i] -= math.pi
    values = oracle.energies(points, role="parameter-shift", candidate=False)
    values = values.reshape(d, 4)
    d_half_pi = values[:, 0] - values[:, 1]
    d_pi = values[:, 2] - values[:, 3]
    return 0.5 * d_half_pi + 0.25 * (1.0 - math.sqrt(2.0)) * d_pi


def run_ps_adam(
    oracle: QuantumWorkOracle,
    initial: np.ndarray,
    learning_rate: float,
    project_candidate: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, int, dict[str, float]]:
    theta = project_candidate(initial)
    oracle.energies(theta, role="initial", candidate=True)
    first = np.zeros_like(theta)
    second = np.zeros_like(theta)
    beta1, beta2, epsilon = 0.9, 0.999, 1e-8
    step = 0
    per_step = 4 * len(theta) + 1
    while oracle.remaining >= per_step:
        step += 1
        gradient = generalized_excitation_gradient(oracle, theta)
        first = beta1 * first + (1.0 - beta1) * gradient
        second = beta2 * second + (1.0 - beta2) * gradient**2
        first_hat = first / (1.0 - beta1**step)
        second_hat = second / (1.0 - beta2**step)
        theta = project_candidate(
            theta - learning_rate * first_hat / (np.sqrt(second_hat) + epsilon)
        )
        oracle.energies(theta, role="adam-iterate", candidate=True)
    return theta, step, {"learning_rate": float(learning_rate)}


def run_qng_diag(
    oracle: QuantumWorkOracle,
    initial: np.ndarray,
    metric_diag: Callable[[np.ndarray], np.ndarray],
    learning_rate: float,
    regularization: float,
    project_candidate: Callable[[np.ndarray], np.ndarray],
) -> tuple[np.ndarray, int, dict[str, float | str]]:
    theta = project_candidate(initial)
    oracle.energies(theta, role="initial", candidate=True)
    d = len(theta)
    # Four shifted circuits per excitation gradient, one circuit per
    # single-parameter metric block, and one evaluated candidate.
    per_step = 5 * d + 1
    step = 0
    while oracle.remaining >= per_step:
        gradient = generalized_excitation_gradient(oracle, theta)
        oracle.consume_metric(d)
        metric = np.asarray(metric_diag(theta), dtype=np.float64).reshape(-1)
        if metric.shape != (d,) or not np.all(np.isfinite(metric)):
            raise RuntimeError("invalid diagonal Fubini--Study metric")
        direction = gradient / (metric + regularization)
        theta = project_candidate(theta - learning_rate * direction)
        oracle.energies(theta, role="qng-iterate", candidate=True)
        step += 1
    return theta, step, {
        "learning_rate": float(learning_rate),
        "regularization": float(regularization),
        "metric": "diagonal Fubini--Study",
    }


def bounded_excitation_minimum(
    coefficients: np.ndarray, lower: float, upper: float
) -> tuple[float, float]:
    """Minimize a reconstructed excitation profile on the candidate interval."""
    coeffs = np.asarray(coefficients, dtype=np.float64)

    def reconstructed(x):
        return (
            coeffs[0]
            + coeffs[1] * np.cos(x / 2.0)
            + coeffs[2] * np.sin(x / 2.0)
            + coeffs[3] * np.cos(x)
            + coeffs[4] * np.sin(x)
        )

    grid = np.linspace(lower, upper, 4097)
    values = reconstructed(grid)
    index = int(np.argmin(values))
    lo = grid[max(0, index - 1)]
    hi = grid[min(len(grid) - 1, index + 1)]
    if lo == hi:
        return float(grid[index]), float(values[index])
    result = minimize_scalar(
        reconstructed,
        bounds=(float(lo), float(hi)),
        method="bounded",
        options={"xatol": 1e-12},
    )
    boundary_candidates = [(float(result.x), float(result.fun))]
    boundary_candidates.extend((float(x), float(reconstructed(x))) for x in (lower, upper))
    return min(boundary_candidates, key=lambda pair: pair[1])


def run_excitation_solve(
    oracle: QuantumWorkOracle,
    initial: np.ndarray,
    project_candidate: Callable[[np.ndarray], np.ndarray],
    lower: float,
    upper: float,
) -> tuple[np.ndarray, int, dict[str, int | str]]:
    from excitationsolve import excitation_solve_step

    theta = project_candidate(initial)
    oracle.energies(theta, role="initial", candidate=True)
    d = len(theta)
    shifts = (np.arange(5, dtype=np.float64) - 2.0) * (4.0 * math.pi / 5.0)
    sweep_cost = 5 * d + 1
    sweeps = 0
    coordinate_updates = 0
    while oracle.remaining >= sweep_cost:
        for parameter in range(d):
            points = np.repeat(theta[None, :], 5, axis=0)
            variations = theta[parameter] + shifts
            points[:, parameter] = variations
            energies = oracle.energies(
                points, role="excitation-reconstruction", candidate=False
            )
            _, _, coefficients = excitation_solve_step(
                variations, energies, return_coeffs=True
            )
            theta[parameter], _ = bounded_excitation_minimum(
                coefficients, lower, upper
            )
            coordinate_updates += 1
        theta = project_candidate(theta)
        oracle.energies(theta, role="excitation-sweep", candidate=True)
        sweeps += 1
    return theta, sweeps, {
        "coordinate_updates": coordinate_updates,
        "samples_per_coordinate": 5,
        "implementation": "ExcitationSolve 2.2.2 reconstruction with bounded-domain minimization",
    }


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_experiment(args: argparse.Namespace) -> dict:
    # Import lazily so --self-test does not require the Grid'5000 VQE stack.
    import vqe_bridge

    started = time.perf_counter()
    dimension = int(vqe_bridge.init(args.molecule, args.basis))
    meta = vqe_bridge.metadata()
    if meta.get("wrap_mode") != "none":
        raise RuntimeError(
            "quantum-aware shifts require VQE_WRAP_MODE=none; candidate bounds are enforced by the optimizer"
        )
    if args.candidate_domain == "pfda-trust-box":
        radius = min(math.sqrt(3.0 / max(1, dimension)) * 0.30, math.pi / 8.0)
        lower, upper, periodic = -radius, radius, False
    else:
        radius = math.pi
        lower, upper, periodic = LOWER, UPPER, True
    project_candidate = candidate_projector(lower, upper, periodic=periodic)
    oracle = QuantumWorkOracle(
        vqe_bridge.eval_batch,
        dimension,
        args.work_budget,
        project_candidate,
    )
    initial = np.zeros(dimension, dtype=np.float64)

    if args.method == "ps-adam":
        final_parameters, iterations, method_config = run_ps_adam(
            oracle, initial, args.adam_learning_rate, project_candidate
        )
    elif args.method == "qng-diag":
        final_parameters, iterations, method_config = run_qng_diag(
            oracle,
            initial,
            vqe_bridge.state_metric_diag,
            args.qng_learning_rate,
            args.qng_regularization,
            project_candidate,
        )
    elif args.method == "excitation-solve":
        final_parameters, iterations, method_config = run_excitation_solve(
            oracle, initial, project_candidate, lower, upper
        )
    else:
        raise ValueError(args.method)

    final_energy = float(
        oracle.energies(final_parameters, role="final-verification", candidate=True)[0]
    ) if oracle.remaining >= 1 else float(oracle.trace[-1]["candidate_energy"])
    elapsed = time.perf_counter() - started
    fci_energy = float(meta["fci_energy"])
    summary = {
        "protocol": "same-UCCSD-quantum-aware-v2",
        "method": args.method,
        "molecule": args.molecule,
        "basis": args.basis,
        "seed": int(args.seed),
        "n_qubits": int(meta["n_qubits"]),
        "n_params": dimension,
        "candidate_domain": args.candidate_domain,
        "parameter_bounds": [float(lower), float(upper)],
        "trust_box_half_width": float(radius),
        "initialization": "zero UCCSD amplitudes; the circuit then prepares the Hartree--Fock reference determinant",
        "work_budget": int(args.work_budget),
        "quantum_work": oracle.quantum_work,
        "energy_circuits": oracle.energy_circuits,
        "metric_circuits": oracle.metric_circuits,
        "unused_work": oracle.remaining,
        "iterations_or_sweeps": int(iterations),
        "final_energy": final_energy,
        "best_energy": float(oracle.best_energy),
        "fci_energy": fci_energy,
        "absolute_error_ha": abs(float(oracle.best_energy) - fci_energy),
        "chemical_accuracy": abs(float(oracle.best_energy) - fci_energy) <= CHEMICAL_ACCURACY_HA,
        "walltime_s": elapsed,
        "device": meta["device"],
        "dtype": meta["dtype"],
        "method_config": method_config,
        "resource_accounting": {
            "unit": "one full-Hamiltonian energy evaluation or one auxiliary metric-tensor circuit (reported separately as well)",
            "parameter_shift": "four circuits per UCCSD excitation parameter (frequencies 1/2 and 1)",
            "qng_metric": "one circuit per single-parameter diagonal metric block",
            "excitation_solve": "five energy circuits per reconstructed coordinate",
            "classical_reconstruction_and_updates": "not charged as quantum work",
        },
    }

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", summary)
    np.save(output / "best_parameters.npy", oracle.best_parameters)
    with (output / "trace.jsonl").open("w", encoding="utf-8") as handle:
        for record in oracle.trace:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    (output / "RUN_SUCCESS").write_text("OK\n", encoding="utf-8")
    return summary


def self_test() -> None:
    """Validate the generalized shift rule without PennyLane."""
    rng = np.random.default_rng(7)
    dimension = 5
    coefficients = rng.normal(size=(dimension, 4))
    theta = rng.uniform(-2.0, 2.0, size=dimension)

    def energy_batch(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        values = np.zeros(len(points))
        for i in range(dimension):
            values += (
                coefficients[i, 0] * np.cos(points[:, i] / 2.0)
                + coefficients[i, 1] * np.sin(points[:, i] / 2.0)
                + coefficients[i, 2] * np.cos(points[:, i])
                + coefficients[i, 3] * np.sin(points[:, i])
            )
        return values

    expected = np.array([
        -0.5 * coefficients[i, 0] * math.sin(theta[i] / 2.0)
        + 0.5 * coefficients[i, 1] * math.cos(theta[i] / 2.0)
        - coefficients[i, 2] * math.sin(theta[i])
        + coefficients[i, 3] * math.cos(theta[i])
        for i in range(dimension)
    ])
    oracle = QuantumWorkOracle(
        energy_batch,
        dimension,
        4 * dimension,
        candidate_projector(-math.pi, math.pi, periodic=True),
    )
    observed = generalized_excitation_gradient(oracle, theta)
    if not np.allclose(observed, expected, atol=1e-11, rtol=1e-11):
        raise AssertionError(f"shift-rule mismatch\n{observed}\n{expected}")
    if oracle.quantum_work != 4 * dimension:
        raise AssertionError("incorrect quantum-work accounting")
    print("self-test OK: four-term excitation shift rule and accounting", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--molecule")
    parser.add_argument("--basis", default="STO-3G")
    parser.add_argument("--work-budget", type=int, default=5000)
    parser.add_argument("--adam-learning-rate", type=float, default=0.05)
    parser.add_argument("--qng-learning-rate", type=float, default=0.01)
    parser.add_argument("--qng-regularization", type=float, default=1e-3)
    parser.add_argument(
        "--candidate-domain",
        choices=("pfda-trust-box", "full-periodic"),
        default="pfda-trust-box",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        if not args.method or not args.molecule or not args.output_dir:
            parser.error("--method, --molecule, and --output-dir are required")
        if args.work_budget < 2:
            parser.error("--work-budget must be at least 2")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    summary = run_experiment(args)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
