#!/usr/bin/env bash
set -euo pipefail

BIN="${BIN:-./build/fractal_optimizer}"
MPI_RANKS="${MPI_RANKS:-2}"
export VQE_PY_DIR="${VQE_PY_DIR:-$(pwd)/python}"
export VQE_BATCH="${VQE_BATCH:-64}"

mpirun -np "$MPI_RANKS" "$BIN" \
  32 1 80 1 vqe \
  --vqe-mol "${VQE_MOLECULE:-LiH}" \
  --vqe-basis "${VQE_BASIS:-STO-3G}" \
  --optimizer-family fbd \
  --fbd-initial-design halton \
  --fbd-decomposition hypersphere \
  --fbd-exploration center \
  --fbd-scoring best-value \
  --fbd-tree-search beam \
  --fbd-exploitation random-local \
  --fbd-beam-width 32 \
  --seed 0
