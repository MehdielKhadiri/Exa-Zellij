#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./build/fractal_optimizer}"
MPI_RANKS="${MPI_RANKS:-2}"

mpirun -np "$MPI_RANKS" "$BIN" \
  32 100 200 1 sphere \
  --optimizer-family fbo \
  --fbo-doe sobol \
  --fbo-doe-distribution distributed \
  --fbo-decomposition hypersphere \
  --fbo-split-policy cyclic \
  --fbo-surrogate pca-gp \
  --fbo-gp-kernel rbf \
  --fbo-exploitation surrogate-local \
  --fbo-budget astra \
  --fbo-radius-mode adaptive \
  --fbo-exploitation-guidance adaptive \
  --rho 0.8 \
  --seed 0
