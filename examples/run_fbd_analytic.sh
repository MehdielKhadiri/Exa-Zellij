#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./build/fractal_optimizer}"
MPI_RANKS="${MPI_RANKS:-2}"

mpirun -np "$MPI_RANKS" "$BIN" \
  32 100 200 1 sphere \
  --optimizer-family fbd \
  --fbd-initial-design halton \
  --fbd-decomposition hypersphere \
  --fbd-exploration center-axial \
  --fbd-scoring improvement-distance \
  --fbd-tree-search beam \
  --fbd-exploitation best-cell \
  --fbd-beam-width 32 \
  --seed 0
