from __future__ import annotations

import argparse
import numpy as np
from scipy.stats import qmc

try:
    from hdbobench.objectives import make_objective
except Exception:
    make_objective = None

OBJECTIVE_BOUNDS = {
    "sphere": (-5.12, 5.12),
    "rastrigin": (-5.12, 5.12),
    "ellipsoid": (-5.12, 5.12),
    "rosenbrock": (-5.12, 5.12),
    "ackley": (-5.12, 5.12),
    "levy": (-5.12, 5.12),
    "multimodal": (-5.12, 5.12),
    "shiftedsphere": (-5.12, 5.12),
    "shifted_sphere": (-5.12, 5.12),
    "shifted-sphere": (-5.12, 5.12),
    "schwefel": (-500.0, 500.0),
    "griewank": (-600.0, 600.0),
    "zakharov": (-10.0, 10.0),
    "styblinski": (-5.0, 5.0),
    "styblinski-tang": (-5.0, 5.0),
    "styblinski_tang": (-5.0, 5.0),
    "dixon-price": (-10.0, 10.0),
    "dixon_price": (-10.0, 10.0),
    "dixonprice": (-10.0, 10.0),
    "bent-cigar": (-100.0, 100.0),
    "bent_cigar": (-100.0, 100.0),
    "bentcigar": (-100.0, 100.0),
    "discus": (-100.0, 100.0),
    "sharp-ridge": (-100.0, 100.0),
    "sharp_ridge": (-100.0, 100.0),
    "sharpridge": (-100.0, 100.0),
    "different-powers": (-100.0, 100.0),
    "different_powers": (-100.0, 100.0),
    "differentpowers": (-100.0, 100.0),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--objective", required=True)
    p.add_argument("--dim", type=int, required=True)
    p.add_argument("--n-init", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--d-eff", type=int, default=None)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--lower-bound", type=float, default=None)
    p.add_argument("--upper-bound", type=float, default=None)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()

    try:
        if make_objective is None:
            raise RuntimeError("hdbobench is unavailable")
        spec = make_objective(
            args.objective,
            dim=args.dim,
            seed=args.seed,
            d_eff=args.d_eff,
            block_size=args.block_size,
        )
        lower = np.asarray(spec.lower, dtype=float)
        upper = np.asarray(spec.upper, dtype=float)
        if lower.ndim == 0:
            lower = np.full(args.dim, float(lower), dtype=float)
        if upper.ndim == 0:
            upper = np.full(args.dim, float(upper), dtype=float)
    except Exception:
        key = args.objective.lower()
        if key not in OBJECTIVE_BOUNDS:
            raise
        lo, hi = OBJECTIVE_BOUNDS[key]
        lower = np.full(args.dim, lo, dtype=float)
        upper = np.full(args.dim, hi, dtype=float)

    if args.lower_bound is not None:
        lower = np.full(args.dim, args.lower_bound, dtype=float)
    if args.upper_bound is not None:
        upper = np.full(args.dim, args.upper_bound, dtype=float)
    if not np.all(lower < upper):
        raise ValueError("lower bounds must be strictly smaller than upper bounds")

    engine = qmc.Sobol(
        d=args.dim,
        scramble=True,
        seed=args.seed,
    )

    # Deliberately draw one point at a time, exactly like SobolSearch.ask().
    X01 = np.vstack([engine.random(1)[0] for _ in range(args.n_init)])

    X = lower + X01 * (upper - lower)

    np.savetxt(args.out, X, delimiter=",", fmt="%.17g")

    print(f"Wrote DOE: {args.out}")
    print(f"shape = {X.shape}")
    print(f"objective = {args.objective}")
    print(f"dim = {args.dim}")
    print(f"n_init = {args.n_init}")
    print(f"seed = {args.seed}")
    print(f"bounds = [{float(lower[0])}, {float(upper[0])}]")


if __name__ == "__main__":
    main()
