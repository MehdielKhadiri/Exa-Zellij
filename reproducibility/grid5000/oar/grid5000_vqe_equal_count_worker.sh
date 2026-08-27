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

methods=(pfda-bs cobyla l-bfgs-b spsa pso sobol)
if (( worker_id < 0 || worker_id >= ${#methods[@]} )); then
  echo "ERROR: worker id must be in 0..5, got $worker_id" >&2
  exit 2
fi
method="${methods[$worker_id]}"
pfda_single_seed="${PFDA_DETERMINISTIC_SINGLE_SEED:-0}"

project_root="${PROJECT_ROOT:-$HOME/koko_llm}"
repro_root="${REPRO_ROOT:-$project_root/reproducibility/grid5000}"
repro_code="${REPRO_CODE_DIR:-$repro_root/code}"
image="${IMAGE:-$HOME/koko-bo-hpc_a100.sif}"
runtime="${CONTAINER_RUNTIME:-}"
if [[ -z "$runtime" ]]; then
  runtime="$(command -v apptainer || command -v singularity || true)"
fi
[[ -x "$runtime" ]] || { echo "ERROR: container runtime unavailable" >&2; exit 127; }
[[ -f "$image" ]] || { echo "ERROR: image unavailable: $image" >&2; exit 2; }
vqe_python="${VQE_PYTHON:-$node_scratch/venv/bin/python}"
[[ -x "$vqe_python" ]] || { echo "ERROR: VQE Python unavailable: $vqe_python" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="$gpu_id"
export NVIDIA_VISIBLE_DEVICES="$gpu_id"
export APPTAINERENV_CUDA_VISIBLE_DEVICES="$gpu_id"
export SINGULARITYENV_CUDA_VISIBLE_DEVICES="$gpu_id"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export XDG_CACHE_HOME="$node_scratch/xdg"
export VQE_JAX_CACHE_ROOT="$node_scratch/jax_worker_${worker_id}"
export VQE_DTYPE="${VQE_DTYPE:-fp64}"
export VQE_DEVICE=default.qubit
export VQE_OFFLINE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONPATH="$project_root/python:$repro_code:${PYTHONPATH:-}"
case "$method" in
  pfda-bs|pso|sobol) export VQE_BATCH=32 ;;
  spsa)              export VQE_BATCH=2 ;;
  *)                 export VQE_BATCH=1 ;;
esac

mkdir -p "$VQE_JAX_CACHE_ROOT" "$campaign_root/logs/$method"
cd "$project_root"

if [[ -n "${MOLECULES_CSV:-}" ]]; then
  IFS=',' read -r -a molecules <<< "$MOLECULES_CSV"
else
  read -r -a molecules <<< "${MOLECULES_LIST:-HeH+ H3+ H4}"
fi
if [[ -n "${SEEDS_CSV:-}" ]]; then
  IFS=',' read -r -a seeds <<< "$SEEDS_CSV"
else
  read -r -a seeds <<< "${SEEDS_LIST:-0 1 2 3 4}"
fi
pfda_depth="${PFDA_DEPTH:-2}"
pfda_beam_width="${PFDA_BEAM_WIDTH:-100}"
pfda_samples_per_child="${PFDA_SAMPLES_PER_CHILD:-3}"
pfda_ils_iterations="${PFDA_ILS_ITERATIONS:-1}"
pfda_global_budget="${PFDA_GLOBAL_BUDGET:-0}"
pfda_ils_contraction="${PFDA_ILS_CONTRACTION:-0.8}"
pfda_sampling_strategy="${PFDA_SAMPLING_STRATEGY:-promising-hypersphere}"
pfda_phs_inflation="${PFDA_PHS_INFLATION:-1.75}"
(( pfda_depth >= 1 )) || { echo "ERROR: PFDA_DEPTH must be positive" >&2; exit 2; }
(( pfda_beam_width >= 1 )) || { echo "ERROR: PFDA_BEAM_WIDTH must be positive" >&2; exit 2; }
(( pfda_samples_per_child == 3 )) || { echo "ERROR: canonical PHS requires exactly three probes" >&2; exit 2; }
[[ "$pfda_sampling_strategy" == "promising-hypersphere" ]] || {
  echo "ERROR: PFDA-BS requires the promising-hypersphere exploration operator" >&2
  exit 2
}
(( pfda_ils_iterations >= 0 )) || { echo "ERROR: PFDA_ILS_ITERATIONS must be nonnegative" >&2; exit 2; }
(( pfda_global_budget >= 0 )) || { echo "ERROR: PFDA_GLOBAL_BUDGET must be nonnegative" >&2; exit 2; }

dimension_for() {
  case "$1" in
    "HeH+") echo 3 ;;
    "H3+")  echo 8 ;;
    "H4")   echo 26 ;;
    "OH-"|"HF"|"NeH+") echo 35 ;;
    "H5")   echo 54 ;;
    "LiH")  echo 92 ;;
    "H6")   echo 117 ;;
    "BH3")  echo 360 ;;
    "CH4")  echo 560 ;;
    *) echo "ERROR: no dimension registered for $1" >&2; return 2 ;;
  esac
}

budget_components() {
  local molecule="$1" ils_iterations="$2"
  local dimension parents children exploration level ils_per_leaf exploitation total
  dimension="$(dimension_for "$molecule")" || return $?
  parents=1
  exploration=0
  for (( level=0; level<pfda_depth; level++ )); do
    children=$((parents * 2 * dimension))
    exploration=$((exploration + pfda_samples_per_child * children))
    if (( children < pfda_beam_width )); then
      parents=$children
    else
      parents=$pfda_beam_width
    fi
  done
  ils_per_leaf=$((1 + 2 * dimension * ils_iterations))
  exploitation=$((parents * ils_per_leaf))
  total=$((1 + exploration + exploitation))
  printf '%s %s %s %s %s\n' "$dimension" "$parents" "$exploration" "$exploitation" "$total"
}

ils_for() {
  local molecule="$1" dimension parents exploration exploitation base increment iterations
  if (( pfda_global_budget == 0 )); then
    echo "$pfda_ils_iterations"
    return 0
  fi
  read -r dimension parents exploration exploitation base < <(budget_components "$molecule" 0)
  if (( base > pfda_global_budget )); then
    echo "ERROR: zero-ILS PFDA structure for $molecule already costs $base > cap $pfda_global_budget" >&2
    return 2
  fi
  increment=$((parents * 2 * dimension))
  iterations=$(((pfda_global_budget - base) / increment))
  echo "$iterations"
}

budget_for() {
  local molecule="$1" ils_iterations="$2"
  local dimension parents exploration exploitation total
  read -r dimension parents exploration exploitation total < <(budget_components "$molecule" "$ils_iterations")
  echo "$total"
}

failed=0
for seed in "${seeds[@]}"; do
  if [[ "$method" == "pfda-bs" && "$pfda_single_seed" == "1" && "$seed" != "0" ]]; then
    echo "[$(date +'%F %T')] skip deterministic PFDA-BS duplicate seed=$seed"
    continue
  fi
  for molecule in "${molecules[@]}"; do
    run_ils_iterations="$(ils_for "$molecule")" || exit $?
    matched_budget="$(budget_for "$molecule" "$run_ils_iterations")" || exit $?
    run_dir="$campaign_root/runs/$method/$molecule/seed_${seed}"
    log="$campaign_root/logs/$method/${molecule}_seed_${seed}.log"
    if [[ -s "$run_dir/RUN_SUCCESS" && -s "$run_dir/summary.json" ]]; then
      echo "[$(date +'%F %T')] skip completed method=$method molecule=$molecule seed=$seed"
      continue
    fi
    mkdir -p "$run_dir"
    rm -f "$run_dir/RUN_FAILED"
    echo "[$(date +'%F %T')] START method=$method molecule=$molecule seed=$seed matched_calls=$matched_budget ils_iterations=$run_ils_iterations"
    command=(
      "$vqe_python" "$repro_code/run_vqe_pfda_equal_count.py"
      --method "$method"
      --molecule "$molecule"
      --basis STO-3G
      --seed "$seed"
      --depth "$pfda_depth"
      --beam-width "$pfda_beam_width"
      --samples-per-child "$pfda_samples_per_child"
      --ils-iterations "$run_ils_iterations"
      --ils-contraction "$pfda_ils_contraction"
      --sampling-strategy "$pfda_sampling_strategy"
      --phs-inflation "$pfda_phs_inflation"
      --output-dir "$run_dir"
    )
    if [[ "$method" != "pfda-bs" ]]; then
      command+=(--budget "$matched_budget")
    fi
    if "$runtime" exec --nv \
      --bind "$project_root:$project_root" \
      --bind "$node_scratch:$node_scratch" \
      --pwd "$project_root" \
      "$image" "${command[@]}" >"$log" 2>&1; then
      echo "[$(date +'%F %T')] DONE method=$method molecule=$molecule seed=$seed"
    else
      rc=$?
      echo "[$(date +'%F %T')] FAIL method=$method molecule=$molecule seed=$seed rc=$rc log=$log" >&2
      printf '%s\n' "$rc" > "$run_dir/RUN_FAILED"
      failed=$((failed + 1))
    fi
  done
done

printf '%s\n' \
  "worker=$worker_id" \
  "method=$method" \
  "host=$(hostname)" \
  "gpu=$gpu_id" \
  "failures=$failed" \
  > "$campaign_root/status/worker_${worker_id}.summary"

(( failed == 0 ))
