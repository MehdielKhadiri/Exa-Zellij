#!/usr/bin/env bash
#OAR -q default
#OAR -t exotic
#OAR -p ecotaxe
#OAR -l host=1,walltime=04:00:00
#OAR -n vqe_30q_pre
#OAR -O vqe_30q_pre_%jobid%.out
#OAR -E vqe_30q_pre_%jobid%.err

set -uo pipefail

project_root="${PROJECT_ROOT:-$HOME/koko_llm}"
repro_root="${REPRO_ROOT:-$project_root/reproducibility/grid5000}"
repro_code="$repro_root/code"
repro_oar="$repro_root/oar"
image="${IMAGE:-$HOME/koko-bo-hpc_a100.sif}"
job_id="${OAR_JOB_ID:-manual}"
campaign_root="${RESULT_ROOT:-$project_root/results/vqe_lih_30q_statevector_weekend_v1}"
node_scratch="/tmp/${USER}_vqe_30q_pre_${job_id}"
runner="$repro_code/run_vqe_large_statevector_checkpointed.py"

cd "$project_root" || exit 1
module load apptainer >/dev/null 2>&1 || module load singularity >/dev/null 2>&1 || true
runtime="$(command -v apptainer || command -v singularity || true)"
host="$(sort -u "${OAR_NODEFILE:?missing OAR_NODEFILE}" | head -n 1)"
mkdir -p "$campaign_root/logs" "$campaign_root/status"

cleanup() {
  oarsh "$host" rm -rf "$node_scratch" >/dev/null 2>&1 || true
}
trap cleanup EXIT

stage_log="$campaign_root/logs/preflight_stage_${job_id}.log"
oarsh "$host" /usr/bin/env \
  PROJECT_ROOT="$project_root" IMAGE="$image" CONTAINER_RUNTIME="$runtime" \
  NODE_SCRATCH="$node_scratch" VQE_REQUIRE_LIGHTNING_GPU=1 \
  "$repro_oar/grid5000_vqe_python_bootstrap.sh" \
  >"$stage_log" 2>&1 || exit $?

run_log="$campaign_root/logs/q30_complex64_preflight_${job_id}.log"
oarsh "$host" /usr/bin/env \
  CUDA_VISIBLE_DEVICES=0 NVIDIA_VISIBLE_DEVICES=0 \
  APPTAINERENV_CUDA_VISIBLE_DEVICES=0 SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  XLA_PYTHON_CLIENT_PREALLOCATE=false VQE_DEVICE=lightning.gpu \
  VQE_COMPUTE_EXACT_REFERENCE=1 VQE_VARIATIONAL_TOLERANCE_HA=0.005 \
  XDG_CACHE_HOME="$node_scratch/xdg" \
  timeout --signal=TERM --kill-after=120 210m \
  "$runtime" exec --nv \
    --bind "$project_root:$project_root" --bind "$node_scratch:$node_scratch" \
    --pwd "$project_root" "$image" "$node_scratch/venv/bin/python" "$runner" \
    --output-root "$campaign_root/runs/q30/complex64" \
    --candidate-count 19 --candidate-indices 0 --worker-id 0 --worker-count 1 \
    --active-electrons 2 --active-orbitals 15 --precision complex64 --seed 0 \
    >"$run_log" 2>&1
rc=$?
printf '%s\n' "$rc" > "$campaign_root/status/q30_complex64_preflight_rc_${job_id}.txt"

python3 "$runner" --aggregate-only --campaign-root "$campaign_root" || true
exit "$rc"
