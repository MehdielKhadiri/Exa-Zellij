#!/usr/bin/env bash
#OAR -q default
#OAR -t exotic
#OAR -p ecotaxe
#OAR -l host=1,walltime=02:00:00
#OAR -n vqe_eq_seedfirst
#OAR -O vqe_eq_seedfirst_%jobid%.out
#OAR -E vqe_eq_seedfirst_%jobid%.err

set -uo pipefail

project_root="${PROJECT_ROOT:-$HOME/koko_llm}"
repro_root="${REPRO_ROOT:-$project_root/reproducibility/grid5000}"
repro_code="$repro_root/code"
repro_oar="$repro_root/oar"
image="${IMAGE:-$HOME/koko-bo-hpc_a100.sif}"
job_id="${OAR_JOB_ID:-manual}"
campaign_root="${RESULT_ROOT:-$project_root/results/vqe_pfda_equal_count_h5_seed0_depth5_beam32_canonical_phs_v1}"
node_scratch="/tmp/${USER}_vqe_equal_count_${job_id}"
gpu_count="${GPUS_PER_NODE:-3}"
execution_label="${EXECUTION_LABEL:-one-Ecotaxe-node-seed-major-daytime-head-start}"
molecules_list="${MOLECULES_LIST:-H5}"
seed_order_text="${SEED_ORDER:-0}"
pfda_depth="${PFDA_DEPTH:-5}"
pfda_beam_width="${PFDA_BEAM_WIDTH:-32}"
pfda_samples_per_child="${PFDA_SAMPLES_PER_CHILD:-3}"
pfda_ils_iterations="${PFDA_ILS_ITERATIONS:-1}"
pfda_ils_contraction="${PFDA_ILS_CONTRACTION:-0.8}"
pfda_sampling_strategy="${PFDA_SAMPLING_STRATEGY:-promising-hypersphere}"
pfda_phs_inflation="${PFDA_PHS_INFLATION:-1.75}"
(( gpu_count >= 1 && gpu_count <= 6 )) || { echo "ERROR: GPUS_PER_NODE must be in 1..6" >&2; exit 2; }

cd "$project_root"
module load apptainer >/dev/null 2>&1 ||
  module load singularity >/dev/null 2>&1 || true
runtime="$(command -v apptainer || command -v singularity || true)"
[[ -x "$runtime" ]] || { echo "ERROR: no Apptainer/Singularity runtime" >&2; exit 127; }
[[ -f "$image" ]] || { echo "ERROR: missing image: $image" >&2; exit 2; }
[[ -s "${OAR_NODEFILE:-}" ]] || { echo "ERROR: missing OAR_NODEFILE" >&2; exit 2; }
host="$(sort -u "$OAR_NODEFILE" | head -n 1)"
[[ -n "$host" ]] || { echo "ERROR: allocated host not found" >&2; exit 2; }

mkdir -p "$campaign_root/logs" "$campaign_root/status"
{
  echo "protocol=PFDA-BS-VQE-canonical-count-v2"
  echo "execution=$execution_label"
  echo "job_id=$job_id"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$host"
  echo "methods=pfda-bs cobyla l-bfgs-b spsa pso sobol"
  echo "molecules=$molecules_list"
  echo "seed_order=$seed_order_text"
  echo "pfda_depth=$pfda_depth"
  echo "pfda_beam_width=$pfda_beam_width"
  echo "pfda_samples_per_child=$pfda_samples_per_child"
  echo "pfda_ils_iterations=$pfda_ils_iterations"
  echo "pfda_ils_contraction=$pfda_ils_contraction"
  echo "pfda_sampling_strategy=$pfda_sampling_strategy"
  echo "pfda_phs_inflation=$pfda_phs_inflation"
  echo "pfda_scoring=minimum objective value over the three PHS probes"
  echo "matched_calls_H5=45285"
  echo "comparison_rule=comparator cap equals native PFDA-BS structural call count"
  echo "result_root=$campaign_root"
} > "$campaign_root/campaign_manifest_${job_id}.txt"

echo "==== One-node seed-major equal-evaluation campaign ===="
echo "job_id=$job_id host=$host root=$campaign_root"
echo "molecules=$molecules_list seeds=$seed_order_text"
echo "Each seed is completed across all six methods and declared molecules before the next seed."

stage_log="$campaign_root/logs/stage_${job_id}_${host}.log"
if ! /usr/bin/env \
  PROJECT_ROOT="$project_root" \
  IMAGE="$image" \
  CONTAINER_RUNTIME="$runtime" \
  NODE_SCRATCH="$node_scratch" \
  MOLECULES_LIST="$molecules_list" \
  bash -lc '
    set -euo pipefail
    mkdir -p "$NODE_SCRATCH/xdg" "$NODE_SCRATCH/jax_stage"
    export XDG_CACHE_HOME="$NODE_SCRATCH/xdg"
    export VQE_JAX_CACHE_ROOT="$NODE_SCRATCH/jax_stage"
    export VQE_OFFLINE=0
    export XLA_PYTHON_CLIENT_PREALLOCATE=false
    "$PROJECT_ROOT/reproducibility/grid5000/oar/grid5000_vqe_python_bootstrap.sh"
    VQE_PYTHON="$NODE_SCRATCH/venv/bin/python"
    read -r -a molecules <<< "$MOLECULES_LIST"
    "$CONTAINER_RUNTIME" exec --nv \
      --bind "$PROJECT_ROOT:$PROJECT_ROOT" \
      --bind "$NODE_SCRATCH:$NODE_SCRATCH" \
      --pwd "$PROJECT_ROOT" \
      "$IMAGE" "$VQE_PYTHON" "$PROJECT_ROOT/reproducibility/grid5000/code/prepare_vqe_equal_count_datasets.py" \
        --molecules "${molecules[@]}" --basis STO-3G
  ' >"$stage_log" 2>&1; then
  echo "ERROR: staging failed; see $stage_log" >&2
  tail -n 80 "$stage_log" >&2 || true
  exit 3
fi

worker_script="$repro_oar/grid5000_vqe_equal_count_worker.sh"
[[ -x "$worker_script" ]] || { echo "ERROR: missing worker: $worker_script" >&2; exit 3; }
vqe_python="$node_scratch/venv/bin/python"

total_failed=0
read -r -a seed_order <<< "$seed_order_text"
for campaign_seed in "${seed_order[@]}"; do
  echo "==== SEED $campaign_seed START ===="
  seed_failed=0

  # Consecutive waves map the six methods onto the available GPUs without
  # oversubscription while ensuring a complete seed is available first.
  for (( wave_start=0; wave_start<6; wave_start+=gpu_count )); do
    pids=()
    worker_ids=()
    wave_size=$((6 - wave_start))
    (( wave_size > gpu_count )) && wave_size=$gpu_count
    for (( gpu_id=0; gpu_id<wave_size; gpu_id++ )); do
      worker_id=$((wave_start + gpu_id))
      worker_log="$campaign_root/logs/headstart_${job_id}_seed${campaign_seed}_worker${worker_id}_gpu${gpu_id}.log"
      echo "seed=$campaign_seed wave=$wave_start worker=$worker_id gpu=$gpu_id"
      /usr/bin/env \
        PROJECT_ROOT="$project_root" \
        IMAGE="$image" \
        CONTAINER_RUNTIME="$runtime" \
        VQE_PYTHON="$vqe_python" \
        OAR_JOB_ID="$job_id" \
        MOLECULES_LIST="$molecules_list" \
        SEEDS_LIST="$campaign_seed" \
        PFDA_DEPTH="$pfda_depth" \
        PFDA_BEAM_WIDTH="$pfda_beam_width" \
        PFDA_SAMPLES_PER_CHILD="$pfda_samples_per_child" \
        PFDA_ILS_ITERATIONS="$pfda_ils_iterations" \
        PFDA_ILS_CONTRACTION="$pfda_ils_contraction" \
        PFDA_SAMPLING_STRATEGY="$pfda_sampling_strategy" \
        PFDA_PHS_INFLATION="$pfda_phs_inflation" \
        "$worker_script" "$worker_id" "$gpu_id" "$campaign_root" "$node_scratch" \
        >"$worker_log" 2>&1 &
      pids+=("$!")
      worker_ids+=("$worker_id")
    done

    for (( index=0; index<wave_size; index++ )); do
      if wait "${pids[$index]}"; then
        echo "seed=$campaign_seed worker=${worker_ids[$index]} status=OK"
      else
        rc=$?
        echo "seed=$campaign_seed worker=${worker_ids[$index]} status=FAIL rc=$rc" >&2
        seed_failed=$((seed_failed + 1))
      fi
    done
  done

  read -r -a molecules <<< "$molecules_list"
  "$runtime" exec \
    --bind "$project_root:$project_root" \
    --bind "$node_scratch:$node_scratch" \
    --pwd "$project_root" \
    "$image" "$vqe_python" "$repro_code/analyze_vqe_pfda_equal_count.py" "$campaign_root" \
      --molecules "${molecules[@]}" --seeds "$campaign_seed" \
    >"$campaign_root/logs/analysis_after_seed_${campaign_seed}.log" 2>&1 || true

  if (( seed_failed != 0 )); then
    echo "ERROR: seed $campaign_seed incomplete; later seeds are deferred for resumable retry" >&2
    total_failed=$((total_failed + seed_failed))
    break
  fi
  touch "$campaign_root/SEED_${campaign_seed}_SUCCESS"
  echo "==== SEED $campaign_seed COMPLETE ===="
done

if (( total_failed != 0 )); then
  exit 1
fi

# This marker is created only if all declared seeds completed within the slot.
all_done=1
for campaign_seed in "${seed_order[@]}"; do
  [[ -f "$campaign_root/SEED_${campaign_seed}_SUCCESS" ]] || all_done=0
done
if (( all_done == 1 )); then
  touch "$campaign_root/CAMPAIGN_SUCCESS"
  echo "Full equal-evaluation campaign complete: $campaign_root"
else
  echo "Walltime ended after a valid seed prefix; later reservations resume the same root."
fi
