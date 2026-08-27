#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 WORKER_ID GPU_ID CAMPAIGN_ROOT NODE_SCRATCH" >&2
  exit 2
fi

worker_id="$1"
gpu_id="$2"
campaign_root="$3"
node_scratch="$4"
project_root="${PROJECT_ROOT:-$HOME/koko_llm}"
repro_root="${REPRO_ROOT:-$project_root/reproducibility/grid5000}"
repro_code="${REPRO_CODE_DIR:-$repro_root/code}"
image="${IMAGE:-$HOME/koko-bo-hpc_a100.sif}"
runtime="${CONTAINER_RUNTIME:-$(command -v apptainer || command -v singularity || true)}"
vqe_python="${VQE_PYTHON:-$node_scratch/venv/bin/python}"
candidate_domain="${CANDIDATE_DOMAIN:-pfda-trust-box}"

[[ -x "$runtime" ]] || { echo "ERROR: container runtime unavailable" >&2; exit 127; }
[[ -x "$vqe_python" ]] || { echo "ERROR: VQE Python unavailable: $vqe_python" >&2; exit 2; }

# The three methods are deterministic from the shared zero-amplitude start,
# so each GPU owns one complete method track rather than repeating fake seeds.
case "$worker_id" in
  0) method="ps-adam" ;;
  1) method="excitation-solve" ;;
  2) method="qng-diag" ;;
  *) echo "ERROR: worker id must be 0, 1, or 2" >&2; exit 2 ;;
esac

# Finish the cheapest and most diagnostic molecules first. Long conditions
# remain condition-level resumable through RUN_SUCCESS markers. A launcher may
# provide a disjoint molecule shard when multiple nodes are available.
if [[ -n "${MOLECULES_CSV:-}" ]]; then
  IFS=',' read -r -a molecules <<< "$MOLECULES_CSV"
else
  molecules=("HeH+" "H3+" "H4" "HF" "NeH+" "OH-" "H5" "LiH")
fi

work_cap() {
  case "$1" in
    "HeH+") echo 19989 ;;
    "H3+")  echo 19987 ;;
    "H4")   echo 19919 ;;
    "H5")   echo 19983 ;;
    "HF"|"NeH+"|"OH-") echo 19953 ;;
    "LiH")  echo 19691 ;;
    *) return 2 ;;
  esac
}

dimension_of() {
  case "$1" in
    "HeH+") echo 3 ;;
    "H3+")  echo 8 ;;
    "H4")   echo 26 ;;
    "H5")   echo 54 ;;
    "HF"|"NeH+"|"OH-") echo 35 ;;
    "LiH")  echo 92 ;;
    *) return 2 ;;
  esac
}

export CUDA_VISIBLE_DEVICES="$gpu_id"
export NVIDIA_VISIBLE_DEVICES="$gpu_id"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu_id"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu_id"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$node_scratch/xdg"
export VQE_DTYPE=fp64
export VQE_DEVICE=default.qubit
export VQE_WRAP_MODE=none
export VQE_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$project_root/python:$repro_code:${PYTHONPATH:-}"

failed=0
for molecule in "${molecules[@]}"; do
  if [[ -n "${QUANTUM_WORK_BUDGET_OVERRIDE:-}" ]]; then
    budget="$QUANTUM_WORK_BUDGET_OVERRIDE"
  else
    budget="$(work_cap "$molecule")"
  fi
  dimension="$(dimension_of "$molecule")"
  run_dir="$campaign_root/runs/$method/$molecule"
  log="$campaign_root/logs/${method}_${molecule}.log"
  if [[ -s "$run_dir/RUN_SUCCESS" && -s "$run_dir/summary.json" ]]; then
    echo "[$(date +'%F %T')] skip completed method=$method molecule=$molecule"
    continue
  fi

  if [[ "$method" == "excitation-solve" ]]; then
    vqe_batch=5
  else
    vqe_batch=$((4 * dimension))
  fi

  export VQE_BATCH="$vqe_batch"
  export VQE_JAX_CACHE_ROOT="$node_scratch/jax_worker_${worker_id}_${method}_${molecule}"
  mkdir -p "$run_dir" "$(dirname "$log")" "$VQE_JAX_CACHE_ROOT"
  rm -f "$run_dir/RUN_FAILED"
  echo "[$(date +'%F %T')] START worker=$worker_id gpu=$gpu_id method=$method molecule=$molecule work_cap=$budget domain=$candidate_domain"

  command=(
    "$vqe_python" "$repro_code/run_vqe_quantum_aware.py"
    --method "$method"
    --molecule "$molecule"
    --basis STO-3G
    --work-budget "$budget"
    --candidate-domain "$candidate_domain"
    --seed 0
    --adam-learning-rate "${ADAM_LEARNING_RATE:-0.05}"
    --qng-learning-rate "${QNG_LEARNING_RATE:-0.01}"
    --qng-regularization "${QNG_REGULARIZATION:-0.001}"
    --output-dir "$run_dir"
  )
  if "$runtime" exec --nv \
    --bind "$project_root:$project_root" \
    --bind "$node_scratch:$node_scratch" \
    --pwd "$project_root" \
    "$image" "${command[@]}" >"$log" 2>&1; then
    echo "[$(date +'%F %T')] DONE method=$method molecule=$molecule"
  else
    rc=$?
    printf '%s\n' "$rc" > "$run_dir/RUN_FAILED"
    echo "[$(date +'%F %T')] FAIL method=$method molecule=$molecule rc=$rc log=$log" >&2
    failed=$((failed + 1))
  fi
done

printf '%s\n' \
  "worker=$worker_id" \
  "host=$(hostname)" \
  "gpu=$gpu_id" \
  "method=$method" \
  "failures=$failed" \
  > "$campaign_root/status/worker_${worker_id}${STATUS_SUFFIX:-}.summary"

(( failed == 0 ))
