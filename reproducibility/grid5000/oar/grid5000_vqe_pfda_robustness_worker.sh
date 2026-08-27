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
job_id="${OAR_JOB_ID:-manual}"

[[ -x "$runtime" ]] || { echo "ERROR: container runtime unavailable" >&2; exit 127; }
[[ -x "$vqe_python" ]] || { echo "ERROR: VQE Python unavailable: $vqe_python" >&2; exit 2; }
[[ -s "$campaign_root/jobs.tsv" ]] || { echo "ERROR: missing $campaign_root/jobs.tsv" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="$gpu_id"
export NVIDIA_VISIBLE_DEVICES="$gpu_id"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu_id"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu_id"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$node_scratch/xdg"
export VQE_JAX_CACHE_ROOT="$node_scratch/jax_robustness_worker_${worker_id}"
export VQE_DTYPE="${VQE_DTYPE:-fp64}"
export VQE_DEVICE=default.qubit
export VQE_OFFLINE=1
export VQE_BATCH="${VQE_BATCH:-32}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$project_root/python:$repro_code:${PYTHONPATH:-}"

mkdir -p "$VQE_JAX_CACHE_ROOT" "$campaign_root/logs" "$campaign_root/status" \
  "$campaign_root/claims/$job_id"
cd "$project_root"

failed=0
completed=0
claimed=0
while IFS=$'\t' read -r variant molecule molecule_slug seed evaluations; do
  [[ "$variant" == "variant" ]] && continue
  run_dir="$campaign_root/runs/$variant/$molecule/seed_${seed}"
  log="$campaign_root/logs/${variant}_${molecule_slug}_seed_${seed}.log"
  claim="$campaign_root/claims/$job_id/${variant}_${molecule_slug}_seed_${seed}"

  if [[ -s "$run_dir/RUN_SUCCESS" && -s "$run_dir/summary.json" ]]; then
    continue
  fi
  if ! mkdir "$claim" 2>/dev/null; then
    continue
  fi
  claimed=$((claimed + 1))
  mkdir -p "$run_dir"
  rm -f "$run_dir/RUN_FAILED"

  mapfile -t variant_args < <(
    "$vqe_python" - "$variant" <<'PY'
import sys
from vqe_pfda_robustness_protocol import VARIANT_BY_NAME
for token in VARIANT_BY_NAME[sys.argv[1]].cli():
    print(token)
PY
  )

  echo "[$(date +'%F %T')] START worker=$worker_id variant=$variant molecule=$molecule seed=$seed expected_calls=$evaluations"
  command=(
    "$vqe_python" "$repro_code/run_vqe_pfda_equal_count.py"
    --method pfda-bs
    --molecule "$molecule"
    --basis STO-3G
    --seed "$seed"
    --output-dir "$run_dir"
    "${variant_args[@]}"
  )
  if "$runtime" exec --nv \
    --bind "$project_root:$project_root" \
    --bind "$node_scratch:$node_scratch" \
    --pwd "$project_root" \
    "$image" "${command[@]}" >"$log" 2>&1; then
    completed=$((completed + 1))
    echo "[$(date +'%F %T')] DONE worker=$worker_id variant=$variant molecule=$molecule seed=$seed"
  else
    rc=$?
    failed=$((failed + 1))
    printf '%s\n' "$rc" > "$run_dir/RUN_FAILED"
    echo "[$(date +'%F %T')] FAIL worker=$worker_id variant=$variant molecule=$molecule seed=$seed rc=$rc log=$log" >&2
  fi
done < "$campaign_root/jobs.tsv"

printf '%s\n' \
  "worker=$worker_id" \
  "host=$(hostname)" \
  "gpu=$gpu_id" \
  "claimed=$claimed" \
  "completed=$completed" \
  "failures=$failed" \
  > "$campaign_root/status/worker_${job_id}_${worker_id}.summary"

(( failed == 0 ))
