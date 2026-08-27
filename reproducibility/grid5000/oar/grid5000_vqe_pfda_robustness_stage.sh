#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -lt 7 ]]; then
  echo "usage: $0 ROBUSTNESS_ROOT PROJECT_ROOT IMAGE RUNTIME JOB_ID HOST1 HOST2" >&2
  exit 2
fi

robustness_root="$1"
project_root="$2"
image="$3"
runtime="$4"
job_id="$5"
shift 5
hosts=("$@")
if [[ "${#hosts[@]}" -ne 2 ]]; then
  echo "ERROR: robustness stage requires exactly two hosts" >&2
  exit 2
fi

analysis_scratch="/tmp/${USER}_vqe_equal_count_${job_id}"
repro_root="${REPRO_ROOT:-$project_root/reproducibility/grid5000}"
repro_code="$repro_root/code"
repro_oar="$repro_root/oar"
mkdir -p "$robustness_root/logs" "$robustness_root/status"
{
  echo "protocol=PFDA-BS-hyperparameter-robustness-v1"
  echo "job_id=$job_id"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "hosts=${hosts[*]}"
  echo "molecules=HeH+ H3+ H4"
  echo "seeds=0 1 2 3 4"
  echo "design=one-factor-at-a-time around D2-beta100-m1-maxiter1"
  echo "comparisons=native-final and fixed-call incumbent trace"
  echo "result_root=$robustness_root"
} > "$robustness_root/campaign_manifest_${job_id}.txt"

oarsh "${hosts[0]}" /usr/bin/env \
  PROJECT_ROOT="$project_root" \
  IMAGE="$image" \
  CONTAINER_RUNTIME="$runtime" \
  ANALYSIS_SCRATCH="$analysis_scratch" \
  ROBUSTNESS_ROOT="$robustness_root" \
  bash -lc '
    "$CONTAINER_RUNTIME" exec \
      --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
      --bind "$ANALYSIS_SCRATCH:$ANALYSIS_SCRATCH" \
      --pwd "$PROJECT_ROOT" \
      "$IMAGE" "$ANALYSIS_SCRATCH/venv/bin/python" \
        "$PROJECT_ROOT/reproducibility/grid5000/code/prepare_vqe_pfda_robustness_jobs.py" "$ROBUSTNESS_ROOT"
  ' || exit 3

robust_worker="$repro_oar/grid5000_vqe_pfda_robustness_worker.sh"
[[ -x "$robust_worker" ]] || { echo "ERROR: robustness worker is not executable" >&2; exit 3; }

pids=()
for worker_id in 0 1 2 3 4 5; do
  node_index=$((worker_id / 3))
  gpu_id=$((worker_id % 3))
  host="${hosts[$node_index]}"
  node_scratch="/tmp/${USER}_vqe_equal_count_${job_id}"
  worker_log="$robustness_root/logs/worker_${job_id}_${worker_id}_${host}_gpu${gpu_id}.log"
  echo "robustness worker=$worker_id host=$host gpu=$gpu_id"
  oarsh "$host" /usr/bin/env \
    PROJECT_ROOT="$project_root" \
    IMAGE="$image" \
    CONTAINER_RUNTIME="$runtime" \
    VQE_PYTHON="$node_scratch/venv/bin/python" \
    OAR_JOB_ID="$job_id" \
    "$robust_worker" "$worker_id" "$gpu_id" "$robustness_root" "$node_scratch" \
    >"$worker_log" 2>&1 &
  pids+=("$!")
done

failed=0
for worker_id in 0 1 2 3 4 5; do
  if wait "${pids[$worker_id]}"; then
    echo "robustness worker=$worker_id parent_status=OK"
  else
    rc=$?
    echo "robustness worker=$worker_id parent_status=FAIL rc=$rc" >&2
    failed=$((failed + 1))
  fi
done

analysis_rc=0
oarsh "${hosts[0]}" /usr/bin/env \
  PROJECT_ROOT="$project_root" \
  IMAGE="$image" \
  CONTAINER_RUNTIME="$runtime" \
  ANALYSIS_SCRATCH="$analysis_scratch" \
  ROBUSTNESS_ROOT="$robustness_root" \
  bash -lc '
    "$CONTAINER_RUNTIME" exec \
      --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
      --bind "$ANALYSIS_SCRATCH:$ANALYSIS_SCRATCH" \
      --pwd "$PROJECT_ROOT" \
      "$IMAGE" "$ANALYSIS_SCRATCH/venv/bin/python" \
        "$PROJECT_ROOT/reproducibility/grid5000/code/analyze_vqe_pfda_robustness.py" "$ROBUSTNESS_ROOT"
  ' || analysis_rc=$?

if (( failed == 0 && analysis_rc == 0 )); then
  touch "$robustness_root/CAMPAIGN_SUCCESS"
  echo "Robustness campaign complete: $robustness_root"
  exit 0
fi

echo "ERROR: robustness incomplete (worker_failures=$failed analysis_rc=$analysis_rc); completed runs remain resumable" >&2
exit 1
