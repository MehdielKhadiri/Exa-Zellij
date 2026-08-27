#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-$HOME/koko_llm}"
image="${IMAGE:-$HOME/koko-bo-hpc_a100.sif}"
runtime="${CONTAINER_RUNTIME:-$(command -v apptainer || command -v singularity || true)}"
node_scratch="${NODE_SCRATCH:?NODE_SCRATCH must be set}"
venv="$node_scratch/venv"

[[ -x "$runtime" ]] || { echo "ERROR: container runtime unavailable" >&2; exit 127; }
mkdir -p "$node_scratch" "$node_scratch/pip" "$node_scratch/xdg"

"$runtime" exec --nv \
  --bind "$project_root:$project_root" \
  --bind "$node_scratch:$node_scratch" \
  --pwd "$project_root" \
  "$image" bash -s -- \
    "$venv" "$node_scratch" "${VQE_REQUIRE_LIGHTNING_GPU:-0}" <<'CONTAINER'
set -euo pipefail
venv="$1"
node_scratch="$2"
require_lightning_gpu="$3"

if [[ ! -x "$venv/bin/python" ]]; then
  python3 -m venv --system-site-packages "$venv"
fi
export PIP_CACHE_DIR="$node_scratch/pip"

if ! "$venv/bin/python" - <<'PY' >/dev/null 2>&1
import excitationsolve
import fsspec
import h5py
import jax
import openfermion
import openfermionpyscf
import pennylane
import pyscf
import scipy
import aiohttp

assert jax.__version__ == "0.4.28"
assert pennylane.__version__ == "0.41.1"
assert pyscf.__version__ == "2.7.0"
assert openfermion.__version__ == "1.7.1"
PY
then
  "$venv/bin/python" -m pip install --upgrade pip setuptools wheel
  "$venv/bin/python" -m pip install \
    'jax[cuda12]==0.4.28' \
    'jaxlib==0.4.28' \
    'pennylane==0.41.1' \
    'pennylane-lightning==0.41.1' \
    'autoray==0.6.12' \
    'pyscf==2.7.0' \
    'openfermion==1.7.1' \
    'openfermionpyscf==0.5' \
    'excitationsolve==2.2.2' \
    'h5py>=3.10' \
    'scipy>=1.11' \
    'aiohttp>=3.9' \
    'fsspec>=2024.2'
fi

if [[ "$require_lightning_gpu" == "1" ]]; then
  if ! "$venv/bin/python" - <<'PY' >/dev/null 2>&1
import numpy as np
import pennylane as qml

qml.device("lightning.gpu", wires=2, c_dtype=np.complex64)
PY
  then
    "$venv/bin/python" -m pip install 'pennylane-lightning-gpu==0.41.1'
  fi
  "$venv/bin/python" - <<'PY'
import numpy as np
import pennylane as qml

device = qml.device("lightning.gpu", wires=2, c_dtype=np.complex64)
print("Native VQE backend", device, flush=True)
PY
fi

"$venv/bin/python" - <<'PY'
import jax
import pennylane
import pyscf
import openfermion
import openfermionpyscf
import excitationsolve
import scipy
devices = jax.devices()
print('JAX', jax.__version__, devices, flush=True)
print('PennyLane', pennylane.__version__, flush=True)
print('PySCF', pyscf.__version__, flush=True)
print('OpenFermion', openfermion.__version__, flush=True)
print('ExcitationSolve', getattr(excitationsolve, '__version__', '2.2.2'), flush=True)
if not any(device.platform == 'gpu' for device in devices):
    raise SystemExit('JAX did not detect an NVIDIA GPU')
PY
CONTAINER

printf '%s\n' "$venv/bin/python"
