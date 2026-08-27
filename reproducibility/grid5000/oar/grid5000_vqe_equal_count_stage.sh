#!/usr/bin/env bash

set -euo pipefail

: "${PROJECT_ROOT:?PROJECT_ROOT is required}"
: "${IMAGE:?IMAGE is required}"
: "${CONTAINER_RUNTIME:?CONTAINER_RUNTIME is required}"
: "${NODE_SCRATCH:?NODE_SCRATCH is required}"
: "${MOLECULES_CSV:?MOLECULES_CSV is required}"

repro_root="${REPRO_ROOT:-$PROJECT_ROOT/reproducibility/grid5000}"
repro_code="${REPRO_CODE_DIR:-$repro_root/code}"
repro_oar="${REPRO_OAR_DIR:-$repro_root/oar}"
export PYTHONPATH="$PROJECT_ROOT/python:$repro_code:${PYTHONPATH:-}"

mkdir -p "$NODE_SCRATCH/xdg" "$NODE_SCRATCH/jax_stage"
export XDG_CACHE_HOME="$NODE_SCRATCH/xdg"
export VQE_JAX_CACHE_ROOT="$NODE_SCRATCH/jax_stage"
export VQE_OFFLINE=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false

"$repro_oar/grid5000_vqe_python_bootstrap.sh"
vqe_python="$NODE_SCRATCH/venv/bin/python"
IFS=',' read -r -a molecules <<< "$MOLECULES_CSV"

"$CONTAINER_RUNTIME" exec --nv \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
  --bind "$NODE_SCRATCH:$NODE_SCRATCH" \
  --pwd "$PROJECT_ROOT" \
  "$IMAGE" "$vqe_python" -c \
    'import numpy, scipy, jax, pennylane, pyscf, h5py; print("dependencies OK")'

"$CONTAINER_RUNTIME" exec --nv \
  --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
  --bind "$NODE_SCRATCH:$NODE_SCRATCH" \
  --pwd "$PROJECT_ROOT" \
  "$IMAGE" "$vqe_python" "$repro_code/prepare_vqe_equal_count_datasets.py" \
    --molecules "${molecules[@]}" --basis STO-3G
