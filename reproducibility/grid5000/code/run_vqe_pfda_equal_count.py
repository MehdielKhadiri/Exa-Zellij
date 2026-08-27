#!/usr/bin/env python3
"""Equal-evaluation VQE optimizer benchmark anchored to native PFDA-BS.

PFDA-BS is configured by its FBD structure, not by a global evaluation
budget.  Its number of energy evaluations is calculated exactly from the
depth, beam width, the three-point Promising Hypersphere Search (PHS)
operator, and the terminal coordinate-search procedure. Comparator methods receive that
calculated number as an external stopping cap.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


METHODS = ("pfda-bs", "cobyla", "l-bfgs-b", "spsa", "pso", "sobol")
SAMPLING_STRATEGIES = ("promising-hypersphere",)
SEARCH_DOMAINS = ("dimension-scaled", "fixed-pi8", "full-periodic")


class EvaluationLimit(RuntimeError):
    pass


class CountedObjective:
    """Count physical objective calls and retain a complete incumbent trace."""

    def __init__(
        self,
        evaluate_batch: Callable[[np.ndarray], np.ndarray],
        dimension: int,
        lower: float,
        upper: float,
        limit: int | None,
        periodic: bool = False,
    ) -> None:
        self._evaluate_batch = evaluate_batch
        self.dimension = int(dimension)
        self.lower = float(lower)
        self.upper = float(upper)
        self.limit = None if limit is None else int(limit)
        self.periodic = bool(periodic)
        self.evaluations = 0
        self.best_value = math.inf
        self.best_x = np.zeros(self.dimension, dtype=np.float64)
        self.trace: list[dict[str, float | int]] = []

    @property
    def remaining(self) -> int | None:
        return None if self.limit is None else self.limit - self.evaluations

    def project(self, points: np.ndarray) -> np.ndarray:
        """Project candidates into the declared bounded or periodic domain."""
        x = np.asarray(points, dtype=np.float64)
        if self.periodic:
            width = self.upper - self.lower
            return (x - self.lower) % width + self.lower
        return np.clip(x, self.lower, self.upper)

    def batch(self, points: np.ndarray) -> np.ndarray:
        x = np.asarray(points, dtype=np.float64)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        if x.ndim != 2 or x.shape[1] != self.dimension:
            raise ValueError(f"expected (B,{self.dimension}) points, got {x.shape}")
        if self.limit is not None and self.evaluations + len(x) > self.limit:
            raise EvaluationLimit(
                f"request for {len(x)} calls exceeds remaining {self.remaining}"
            )
        x = self.project(x)
        values = np.asarray(self._evaluate_batch(x), dtype=np.float64).reshape(-1)
        if len(values) != len(x):
            raise RuntimeError("objective returned the wrong batch length")
        if not np.all(np.isfinite(values)):
            raise RuntimeError("objective returned a non-finite energy")
        for point, value in zip(x, values):
            self.evaluations += 1
            if float(value) < self.best_value:
                self.best_value = float(value)
                self.best_x = point.copy()
            self.trace.append(
                {"evaluation": self.evaluations, "best_value": self.best_value}
            )
        return values

    def __call__(self, point: np.ndarray) -> float:
        return float(self.batch(np.asarray(point).reshape(1, -1))[0])


@dataclass
class Fractal:
    center: np.ndarray
    radius: float
    score: float = math.inf


def pfda_level_sizes(dimension: int, depth: int, beam_width: int) -> list[int]:
    """Exact number of children evaluated at every decomposition depth."""
    branching = 2 * dimension
    levels: list[int] = []
    parents = 1
    for _ in range(depth):
        children = parents * branching
        levels.append(children)
        parents = min(beam_width, children)
    return levels


def pfda_evaluation_count(
    dimension: int,
    depth: int,
    beam_width: int,
    samples_per_child: int,
    ils_iterations: int,
) -> dict[str, int | list[int]]:
    """Exact count for this benchmark's deterministic PFDA-BS operators.

    One root call is followed by ``m`` samples per generated child.  Terminal
    ILS evaluates its leaf center once and then both coordinate directions in
    every iteration, hence N_ILS = 1 + 2*n*maxiter per retained leaf.
    """
    levels = pfda_level_sizes(dimension, depth, beam_width)
    leaves = min(beam_width, levels[-1])
    exploration = samples_per_child * sum(levels)
    ils_per_leaf = 1 + 2 * dimension * ils_iterations
    exploitation = leaves * ils_per_leaf
    return {
        "root": 1,
        "level_children": levels,
        "exploration": exploration,
        "terminal_leaves": leaves,
        "ils_per_leaf": ils_per_leaf,
        "exploitation": exploitation,
        "total": 1 + exploration + exploitation,
    }


def decompose_hyperspheres(parents: Iterable[Fractal], dimension: int) -> list[Fractal]:
    alpha = 1.0 / (1.0 + math.sqrt(2.0))
    children: list[Fractal] = []
    for parent in parents:
        child_radius = alpha * parent.radius
        offset = parent.radius - child_radius
        for axis in range(dimension):
            for sign in (-1.0, 1.0):
                center = parent.center.copy()
                center[axis] += sign * offset
                children.append(Fractal(center=center, radius=child_radius))
    return children


def child_samples(
    child: Fractal,
    samples_per_child: int,
    lower: float,
    upper: float,
    strategy: str,
    phs_inflation: float,
    project: Callable[[np.ndarray], np.ndarray] | None = None,
) -> np.ndarray:
    """Canonical PHS probes: center and two equidistant diagonal points."""
    if strategy != "promising-hypersphere":
        raise ValueError(f"PFDA-BS requires promising-hypersphere, got {strategy}")
    if samples_per_child != 3:
        raise ValueError(
            "Promising Hypersphere Search requires exactly three probes per child"
        )
    if phs_inflation <= 0.0:
        raise ValueError("PHS inflation must be positive")

    dimension = len(child.center)
    diagonal_offset = (
        phs_inflation * child.radius / math.sqrt(max(1, dimension))
    )
    points = np.stack(
        [
            child.center,
            child.center + diagonal_offset,
            child.center - diagonal_offset,
        ]
    )
    if project is not None:
        return project(points)
    return np.clip(points, lower, upper)


def run_pfda_bs(
    objective: CountedObjective,
    dimension: int,
    radius: float,
    depth: int,
    beam_width: int,
    samples_per_child: int,
    ils_iterations: int,
    ils_contraction: float,
    sampling_strategy: str,
    phs_inflation: float,
) -> None:
    """Canonical PHS + hypersphere FBD + beam search + terminal ILS."""
    objective(np.zeros(dimension, dtype=np.float64))
    open_fractals = [Fractal(np.zeros(dimension, dtype=np.float64), radius)]

    for _ in range(depth):
        children = decompose_hyperspheres(open_fractals, dimension)
        points = np.concatenate(
            [
                child_samples(
                    child,
                    samples_per_child,
                    objective.lower,
                    objective.upper,
                    sampling_strategy,
                    phs_inflation,
                    objective.project,
                )
                for child in children
            ],
            axis=0,
        )
        values = objective.batch(points)
        for index, child in enumerate(children):
            lo = index * samples_per_child
            hi = lo + samples_per_child
            child.score = float(np.min(values[lo:hi]))
        children.sort(key=lambda item: item.score)
        open_fractals = children[: min(beam_width, len(children))]

    # Fixed-work ILS makes the native PFDA-BS count known before execution.
    for leaf in open_fractals:
        current = objective.project(leaf.center.copy())
        current_value = objective(current)
        step = max(leaf.radius, np.finfo(float).eps)
        for _ in range(ils_iterations):
            for axis in range(dimension):
                plus = current.copy()
                minus = current.copy()
                plus[axis] += step
                minus[axis] -= step
                plus = objective.project(plus)
                minus = objective.project(minus)
                values = objective.batch(np.stack([plus, minus]))
                best_index = int(np.argmin(values))
                if float(values[best_index]) < current_value:
                    current = plus if best_index == 0 else minus
                    current_value = float(values[best_index])
            step *= ils_contraction


def _fill_remaining_with_sobol(
    objective: CountedObjective, seed: int, skip: int = 0
) -> None:
    from scipy.stats import qmc

    remaining = objective.remaining or 0
    if remaining <= 0:
        return
    sampler = qmc.Sobol(d=objective.dimension, scramble=True, seed=seed)
    if skip:
        sampler.fast_forward(skip)
        unit = sampler.random(remaining)
    else:
        exponent = int(math.ceil(math.log2(max(1, remaining))))
        unit = sampler.random_base2(exponent)[:remaining]
    points = qmc.scale(unit, objective.lower, objective.upper)
    objective.batch(points)


def run_sobol(objective: CountedObjective, seed: int) -> None:
    objective(np.zeros(objective.dimension))
    _fill_remaining_with_sobol(objective, seed)


def run_pso(objective: CountedObjective, seed: int) -> None:
    rng = np.random.default_rng(seed)
    swarm_size = min(32, objective.limit or 32)
    positions = rng.uniform(
        objective.lower, objective.upper, size=(swarm_size, objective.dimension)
    )
    positions[0] = 0.0
    velocities = np.zeros_like(positions)
    pbest = positions.copy()
    pvalues = objective.batch(positions)
    gbest = pbest[int(np.argmin(pvalues))].copy()
    while (objective.remaining or 0) > 0:
        count = min(swarm_size, objective.remaining or 0)
        r1 = rng.random((swarm_size, objective.dimension))
        r2 = rng.random((swarm_size, objective.dimension))
        velocities = (
            0.7298 * velocities
            + 1.49618 * r1 * (pbest - positions)
            + 1.49618 * r2 * (gbest - positions)
        )
        positions = np.clip(positions + velocities, objective.lower, objective.upper)
        values = objective.batch(positions[:count])
        improved = values < pvalues[:count]
        improved_indices = np.flatnonzero(improved)
        pbest[improved_indices] = positions[improved_indices]
        pvalues[improved_indices] = values[improved_indices]
        gbest = pbest[int(np.argmin(pvalues))].copy()


def run_spsa(objective: CountedObjective, seed: int) -> None:
    rng = np.random.default_rng(seed)
    x = np.zeros(objective.dimension, dtype=np.float64)
    objective(x)
    iteration = 0
    a = 0.15 * (objective.upper - objective.lower)
    c = 0.10 * (objective.upper - objective.lower)
    while (objective.remaining or 0) >= 2:
        iteration += 1
        ak = a / ((iteration + 10) ** 0.602)
        ck = c / (iteration**0.101)
        delta = rng.choice((-1.0, 1.0), size=objective.dimension)
        pair = np.stack([x + ck * delta, x - ck * delta])
        values = objective.batch(pair)
        gradient = (values[0] - values[1]) / (2.0 * ck) * delta
        x = np.clip(x - ak * gradient, objective.lower, objective.upper)
    if objective.remaining == 1:
        objective(x)


def run_scipy(objective: CountedObjective, method: str) -> None:
    from scipy.optimize import minimize

    bounds = [(objective.lower, objective.upper)] * objective.dimension
    scipy_method = "COBYLA" if method == "cobyla" else "L-BFGS-B"
    options: dict[str, float | int] = {
        "maxiter": max(1, objective.limit or 1),
    }
    if scipy_method == "COBYLA":
        options.update({"rhobeg": 0.25 * (objective.upper - objective.lower), "tol": 1e-8})
    else:
        options.update({"ftol": 0.0, "gtol": 0.0, "maxfun": objective.limit or 1})
    try:
        kwargs: dict[str, object] = {
            "fun": objective,
            "x0": np.zeros(objective.dimension),
            "method": scipy_method,
            "options": options,
        }
        if scipy_method == "COBYLA":
            kwargs["constraints"] = [
                {"type": "ineq", "fun": lambda x: x - objective.lower},
                {"type": "ineq", "fun": lambda x: objective.upper - x},
            ]
        else:
            kwargs["bounds"] = bounds
        minimize(**kwargs)
    except EvaluationLimit:
        pass


def checkpoint_values(trace: list[dict[str, float | int]], checkpoints: list[int]) -> dict[str, float]:
    result: dict[str, float] = {}
    for checkpoint in checkpoints:
        if checkpoint <= len(trace):
            result[str(checkpoint)] = float(trace[checkpoint - 1]["best_value"])
        elif trace:
            result[str(checkpoint)] = float(trace[-1]["best_value"])
    return result


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--molecule", required=True)
    parser.add_argument("--basis", default="STO-3G")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=100)
    parser.add_argument("--samples-per-child", type=int, choices=(3,), default=3)
    parser.add_argument("--ils-iterations", type=int, default=1)
    parser.add_argument("--ils-contraction", type=float, default=0.8)
    parser.add_argument("--phs-inflation", type=float, default=1.75)
    parser.add_argument(
        "--sampling-strategy",
        choices=SAMPLING_STRATEGIES,
        default="promising-hypersphere",
    )
    parser.add_argument("--budget", type=int, default=None,
                        help="Required for comparators; forbidden for PFDA-BS")
    parser.add_argument("--self-test-dimension", type=int, default=0)
    parser.add_argument(
        "--geometry-profile",
        choices=("linear-h4",),
        default=None,
        help="Optional controlled molecular geometry instead of a dataset entry",
    )
    parser.add_argument(
        "--spacing-angstrom",
        type=float,
        default=None,
        help="Nearest-neighbor spacing for --geometry-profile linear-h4",
    )
    parser.add_argument(
        "--search-domain",
        choices=SEARCH_DOMAINS,
        default="dimension-scaled",
        help=(
            "Parameter domain: the PFDA-BS dimension-scaled trust box, a "
            "dimension-independent [-pi/8,pi/8]^d box, or the full "
            "periodic [-pi,pi]^d UCCSD domain"
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.method == "pfda-bs" and args.budget is not None:
        raise SystemExit("PFDA-BS is structurally configured; do not pass --budget")
    if args.method != "pfda-bs" and (args.budget is None or args.budget <= 0):
        raise SystemExit("comparator methods require the calculated PFDA-BS --budget")

    if args.self_test_dimension > 0:
        dimension = args.self_test_dimension
        n_qubits = 0
        fci_energy = 0.0

        def evaluate_batch(points: np.ndarray) -> np.ndarray:
            return np.sum(points * points, axis=1)

        metadata = {"device": "self-test", "dtype": "float64"}
    elif args.geometry_profile is not None:
        if args.spacing_angstrom is None or args.spacing_angstrom <= 0.0:
            raise SystemExit("--geometry-profile requires a positive --spacing-angstrom")
        import vqe_bridge
        from vqe_geometry_profiles import linear_h4

        profile = linear_h4(args.spacing_angstrom, args.basis)
        original_loader = vqe_bridge.load_molecule
        try:
            vqe_bridge.load_molecule = lambda _name, _basis: (
                profile.molecule,
                profile.fci_energy,
            )
            label = f"H4-linear-R{args.spacing_angstrom:.3f}A"
            dimension = int(vqe_bridge.init(label, args.basis))
        finally:
            vqe_bridge.load_molecule = original_loader
        metadata = vqe_bridge.metadata()
        metadata.update(profile.metadata)
        n_qubits = int(metadata["n_qubits"])
        fci_energy = float(profile.fci_energy)
        evaluate_batch = vqe_bridge.eval_batch
    else:
        import vqe_bridge

        dimension = int(vqe_bridge.init(args.molecule, args.basis))
        metadata = vqe_bridge.metadata()
        n_qubits = int(metadata["n_qubits"])
        fci_energy = float(metadata["fci_energy"])
        evaluate_batch = vqe_bridge.eval_batch

    canonical_radius = min(
        math.sqrt(3.0 / max(1, dimension)) * 0.30,
        math.pi / 8.0,
    )
    if args.search_domain == "dimension-scaled":
        radius = canonical_radius
    elif args.search_domain == "fixed-pi8":
        radius = math.pi / 8.0
    else:
        radius = math.pi

    count = pfda_evaluation_count(
        dimension,
        args.depth,
        args.beam_width,
        args.samples_per_child,
        args.ils_iterations,
    )
    matched_budget = int(count["total"])
    if args.method != "pfda-bs" and args.budget != matched_budget:
        raise SystemExit(
            f"comparator budget {args.budget} != calculated PFDA-BS count {matched_budget}"
        )

    run_dir = args.output_dir
    success = run_dir / "RUN_SUCCESS"
    if success.exists():
        print(f"skip completed: {run_dir}", flush=True)
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)
    objective = CountedObjective(
        evaluate_batch=evaluate_batch,
        dimension=dimension,
        lower=-radius,
        upper=radius,
        limit=None if args.method == "pfda-bs" else matched_budget,
        periodic=args.search_domain == "full-periodic",
    )

    start = time.perf_counter()
    if args.method == "pfda-bs":
        run_pfda_bs(
            objective,
            dimension,
            radius,
            args.depth,
            args.beam_width,
            args.samples_per_child,
            args.ils_iterations,
            args.ils_contraction,
            args.sampling_strategy,
            args.phs_inflation,
        )
    elif args.method == "sobol":
        run_sobol(objective, args.seed)
    elif args.method == "pso":
        run_pso(objective, args.seed)
    elif args.method == "spsa":
        run_spsa(objective, args.seed)
    else:
        run_scipy(objective, args.method)
    elapsed = time.perf_counter() - start

    if args.method == "pfda-bs" and objective.evaluations != matched_budget:
        raise RuntimeError(
            f"{args.method} used {objective.evaluations}, expected {matched_budget} calls"
        )
    if objective.evaluations > matched_budget:
        raise RuntimeError(
            f"{args.method} exceeded the matched cap: "
            f"{objective.evaluations} > {matched_budget}"
        )
    checkpoints = sorted(
        set([matched_budget] + [max(1, int(matched_budget * f)) for f in (0.25, 0.5)])
    )
    hartree_fock_energy = float(objective.trace[0]["best_value"])
    max_abs_parameter = float(np.max(np.abs(objective.best_x))) if dimension else 0.0
    boundary_threshold = 0.99 * radius
    boundary_coordinates = int(
        np.count_nonzero(np.abs(objective.best_x) >= boundary_threshold)
    )
    summary = {
        "protocol": "PFDA-BS-VQE-domain-ablation-v1",
        "method": args.method,
        "molecule": args.molecule,
        "basis": args.basis,
        "geometry_profile": args.geometry_profile,
        "spacing_angstrom": args.spacing_angstrom,
        "seed": args.seed,
        "dimension": dimension,
        "n_qubits": n_qubits,
        "search_domain": args.search_domain,
        "domain_projection": (
            "periodic wrapping" if args.search_domain == "full-periodic"
            else "bounded clipping"
        ),
        "search_half_width": radius,
        "canonical_trust_box_half_width": canonical_radius,
        "trust_box_half_width": radius,
        "pfda_configuration": {
            "depth": args.depth,
            "beam_width": args.beam_width,
            "samples_per_child": args.samples_per_child,
            "ils_iterations": args.ils_iterations,
            "ils_contraction": args.ils_contraction,
            "sampling_strategy": args.sampling_strategy,
            "phs_inflation": args.phs_inflation,
            "exploration": "center and two equidistant diagonal probes",
            "scoring": "minimum objective value over the three PHS probes",
        },
        "pfda_evaluation_count": count,
        "matched_evaluations": matched_budget,
        "evaluations_used": objective.evaluations,
        "checkpoints": checkpoint_values(objective.trace, checkpoints),
        "best_energy": objective.best_value,
        "hartree_fock_energy": hartree_fock_energy,
        "improvement_over_hartree_fock": hartree_fock_energy - objective.best_value,
        "fci_energy": fci_energy,
        "absolute_error": abs(objective.best_value - fci_energy),
        "best_parameter_l2_norm": float(np.linalg.norm(objective.best_x)),
        "max_absolute_parameter": max_abs_parameter,
        "boundary_threshold_fraction": 0.99,
        "boundary_coordinates": boundary_coordinates,
        "boundary_coordinate_fraction": boundary_coordinates / max(1, dimension),
        "maximum_radius_fraction": max_abs_parameter / radius,
        "wall_time_s": elapsed,
        "objective_metadata": metadata,
    }
    write_json(run_dir / "trace.json", objective.trace)
    write_json(run_dir / "summary.json", summary)
    np.save(run_dir / "best_parameters.npy", objective.best_x)
    success.write_text("OK\n", encoding="ascii")
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
