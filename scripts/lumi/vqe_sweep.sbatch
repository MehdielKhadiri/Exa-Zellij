#!/bin/bash -l
#SBATCH --job-name=fractal-vqe
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --gpus-per-node=8
#SBATCH --time=02:00:00
#SBATCH --account=CHANGE_ME
#SBATCH --exclusive
#SBATCH --output=fractal-vqe-%j.out
#SBATCH --error=fractal-vqe-%j.err

set -euo pipefail

module load LUMI/24.03 partition/G
module load PrgEnv-amd rocm craype-accel-amd-gfx90a
module load EasyBuild-user
module unload cray-python 2>/dev/null || true

REPO_ROOT="${REPO_ROOT:-$HOME/fractal-quantum-optimizer}"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-lumi-hip}"
BIN="${BIN:-$BUILD_DIR/fractal_optimizer}"
GPU_WRAPPER="$REPO_ROOT/scripts/lumi/select_gpu.sh"

# Site-provided ROCm/JAX container and a user venv containing PennyLane/PySCF.
SIF="${SIF:-/appl/local/containers/sif-images/lumi-jax-rocm-6.2.4-python-3.12-jax-0.4.35.sif}"
VENV="${VENV:-/project/CHANGE_ME/$USER/venvs/jax312-pl}"
PY_HOME="${PY_HOME:-/opt/miniconda3/envs/jax}"
SINGULARITY_BIND="${SINGULARITY_BIND:-/pfs,/project,/scratch,/appl,/opt/cray,/var/spool/slurmd,/users}"
MPICH_CLANG="${MPICH_CLANG:-/opt/cray/pe/mpich/8.1.29/ofi/crayclang/17.0}"

[[ -x "$BIN" ]] || { echo "Missing executable: $BIN" >&2; exit 1; }
[[ -x "$GPU_WRAPPER" ]] || { echo "Missing GPU wrapper: $GPU_WRAPPER" >&2; exit 1; }
[[ -f "$SIF" ]] || { echo "Missing container: $SIF" >&2; exit 1; }
[[ -d "$VENV" ]] || { echo "Missing Python venv: $VENV" >&2; exit 1; }

export CRAYPE_LINK_TYPE=dynamic
export CRAY_MPICH_DIR="$MPICH_CLANG"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$HOME/pl-data-cache}"
export VQE_PY_DIR="$REPO_ROOT/python"
export VQE_BATCH="${VQE_BATCH:-256}"
export HDF5_USE_FILE_LOCKING=FALSE
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=spread
export OMP_PLACES=threads

MOLS_CSV="${MOLS:-LiH}"
BASES_CSV="${BASES:-STO-3G}"
IFS=',' read -r -a MOLECULES <<< "$MOLS_CSV"
IFS=',' read -r -a BASES <<< "$BASES_CSV"

# Populate the shared qchem cache once before the parallel ranks start. This
# avoids concurrent downloads and permits the timed runs to use offline mode.
singularity exec --rocm \
  -B "$SINGULARITY_BIND" \
  --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
  --env PYTHONHOME="$PY_HOME" \
  --env PYTHONPATH="$VENV/lib/python3.12/site-packages:$PY_HOME/lib/python3.12/site-packages" \
  --env FQO_MOLECULES="$MOLS_CSV" \
  --env FQO_BASES="$BASES_CSV" \
  "$SIF" python -c '
import os
import pennylane as qml
for molecule in os.environ["FQO_MOLECULES"].split(","):
    for basis in os.environ["FQO_BASES"].split(","):
        qml.data.load(
            "qchem",
            molname=molecule,
            basis=basis,
            attributes=["molecule", "fci_energy"],
        )
        print(f"prefetched {molecule}/{basis}", flush=True)
'

BEAM_CAP="${BEAM_CAP:-100}"
BUDGET="${BUDGET:-80}"
RHO="${RHO:-0.8}"
MPI_RANKS="${MPI_RANKS:-8}"

CPU_BIND="mask_cpu:fe000000000000,fe00000000000000"
CPU_BIND="${CPU_BIND},fe0000,fe000000"
CPU_BIND="${CPU_BIND},fe,fe00"
CPU_BIND="${CPU_BIND},fe00000000,fe0000000000"

RESULT_DIR="${RESULT_DIR:-$REPO_ROOT/results/lumi-vqe-${SLURM_JOB_ID}}"
mkdir -p "$RESULT_DIR"
CSV="$RESULT_DIR/results.csv"
echo "molecule,basis,ranks,budget,status,final_objective,total_time_s,log" > "$CSV"

run_pair() {
  local molecule="$1"
  local basis="$2"
  local safe="${molecule//[^A-Za-z0-9]/_}_${basis//[^A-Za-z0-9]/_}"
  local log="$RESULT_DIR/${safe}.log"
  local status="OK"

  if ! srun -N "$SLURM_JOB_NUM_NODES" -n "$MPI_RANKS" \
      --ntasks-per-node=8 \
      --cpu-bind="$CPU_BIND" \
      --distribution=block:block \
      singularity exec --rocm \
        -B "$SINGULARITY_BIND" \
        --env CRAY_MPICH_DIR="$CRAY_MPICH_DIR" \
        --env XDG_CACHE_HOME="$XDG_CACHE_HOME" \
        --env VQE_PY_DIR="$VQE_PY_DIR" \
        --env VQE_BATCH="$VQE_BATCH" \
        --env VQE_OFFLINE=1 \
        --env PYTHONHOME="$PY_HOME" \
        --env PYTHONPATH="$VENV/lib/python3.12/site-packages:$PY_HOME/lib/python3.12/site-packages:$VQE_PY_DIR" \
        "$SIF" \
        "$GPU_WRAPPER" "$BIN" \
          "$BEAM_CAP" 1 "$BUDGET" 1 vqe \
          --vqe-mol "$molecule" \
          --vqe-basis "$basis" \
          --optimizer-family fbo \
          --fbo-doe sobol \
          --fbo-doe-distribution distributed \
          --fbo-decomposition hypersphere \
          --fbo-surrogate pca-gp \
          --fbo-gp-kernel rbf \
          --fbo-exploitation surrogate-local \
          --fbo-budget astra \
          --rho "$RHO" \
          --seed 0 >"$log" 2>&1; then
    status="FAIL"
  fi

  local best
  local seconds
  best="$(awk '/^Minimum objective value:/ {v=$4} END {print v}' "$log")"
  seconds="$(awk '/^Total Execution time:/ {v=$4} END {print v}' "$log")"
  echo "$molecule,$basis,$MPI_RANKS,$BUDGET,$status,${best:-NA},${seconds:-NA},$log" >> "$CSV"
}

for molecule in "${MOLECULES[@]}"; do
  for basis in "${BASES[@]}"; do
    run_pair "$molecule" "$basis"
  done
done

echo "Results: $CSV"
