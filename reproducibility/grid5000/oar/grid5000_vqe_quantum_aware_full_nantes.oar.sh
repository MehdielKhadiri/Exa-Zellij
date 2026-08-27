#!/usr/bin/env bash
#OAR -q default
#OAR -t exotic
#OAR -p ecotaxe
#OAR -l host=1,walltime=06:00:00
#OAR -n vqe_qaware_full
#OAR -O vqe_qaware_full_%jobid%.out
#OAR -E vqe_qaware_full_%jobid%.err

set -uo pipefail

project_root="${PROJECT_ROOT:-$HOME/koko_llm}"
repro_root="${REPRO_ROOT:-$project_root/reproducibility/grid5000}"
repro_code="$repro_root/code"
repro_oar="$repro_root/oar"
image="${IMAGE:-$HOME/koko-bo-hpc_a100.sif}"
job_id="${OAR_JOB_ID:-manual}"
campaign_root="${RESULT_ROOT:-$project_root/results/vqe_quantum_aware_matched_full_v2}"
candidate_domain="${CANDIDATE_DOMAIN:-pfda-trust-box}"
pfda_root="${PFDA_ROOT:-$project_root/results/vqe_pfda_equal_count_beam2_all_molecules_b20k_v1}"

cd "$project_root"
module load apptainer >/dev/null 2>&1 || module load singularity >/dev/null 2>&1 || true
runtime="$(command -v apptainer || command -v singularity || true)"
[[ -x "$runtime" ]] || { echo "ERROR: no Apptainer/Singularity runtime" >&2; exit 127; }
[[ -f "$image" ]] || { echo "ERROR: missing image: $image" >&2; exit 2; }
[[ -s "${OAR_NODEFILE:-}" ]] || { echo "ERROR: missing OAR_NODEFILE" >&2; exit 2; }
host="$(sort -u "$OAR_NODEFILE" | head -n 1)"
node_scratch="/tmp/${USER}_vqe_quantum_full_${job_id}"

mkdir -p "$campaign_root/logs" "$campaign_root/status"
{
  echo "protocol=same-UCCSD-quantum-aware-v2"
  echo "job_id=$job_id"
  echo "started_at=$(date --iso-8601=seconds)"
  echo "host=$host"
  echo "molecules=HeH+ H3+ H4 HF NeH+ OH- H5 LiH"
  echo "methods=ps-adam qng-diag excitation-solve"
  echo "stochastic_repetitions=none; all methods are deterministic from the shared initialization"
  echo "ansatz=the same complete UCCSD ansatz and fixed molecular Hamiltonians used by PFDA-BS"
  echo "initialization=zero UCCSD amplitudes, which prepare the Hartree--Fock reference determinant"
  echo "candidate_domain=$candidate_domain"
  echo "candidate_bounds=the exact dimension-dependent PFDA-BS trust box when candidate_domain=pfda-trust-box"
  echo "quantum_work_caps=matched molecule by molecule to the completed PFDA-BS campaign"
  echo "work_unit=one full-Hamiltonian energy evaluation or one auxiliary metric-tensor circuit; ledgers reported separately"
  echo "parameter_shift_cost=4 energy circuits per UCCSD parameter"
  echo "qng_metric_cost=1 auxiliary circuit per diagonal single-parameter block"
  echo "excitation_solve_cost=5 energy circuits per reconstructed coordinate"
  echo "pfda_reference_root=$pfda_root"
  echo "result_root=$campaign_root"
} > "$campaign_root/campaign_manifest_${job_id}.txt"

echo "==== MATCHED-DOMAIN QUANTUM-AWARE VQE CAMPAIGN ===="
echo "job_id          : $job_id"
echo "host            : $host"
echo "candidate domain: $candidate_domain"
echo "result root     : $campaign_root"

stage_log="$campaign_root/logs/stage_${host}_${job_id}.log"
oarsh "$host" /usr/bin/env \
  PROJECT_ROOT="$project_root" \
  IMAGE="$image" \
  CONTAINER_RUNTIME="$runtime" \
  NODE_SCRATCH="$node_scratch" \
  MOLECULES_CSV="HeH+,H3+,H4,HF,NeH+,OH-,H5,LiH" \
  "$repro_oar/grid5000_vqe_equal_count_stage.sh" \
  >"$stage_log" 2>&1
stage_rc=$?
if (( stage_rc != 0 )); then
  echo "ERROR: dependency/dataset staging failed rc=$stage_rc" >&2
  tail -n 100 "$stage_log" >&2 || true
  exit "$stage_rc"
fi

worker="$repro_oar/grid5000_vqe_quantum_aware_full_worker.sh"
[[ -x "$worker" ]] || { echo "ERROR: worker is not executable: $worker" >&2; exit 3; }
pids=()
for worker_id in 0 1 2; do
  gpu_id="$worker_id"
  log="$campaign_root/logs/worker_${worker_id}_${host}_gpu${gpu_id}_${job_id}.log"
  oarsh "$host" /usr/bin/env \
    PROJECT_ROOT="$project_root" \
    IMAGE="$image" \
    CONTAINER_RUNTIME="$runtime" \
    VQE_PYTHON="$node_scratch/venv/bin/python" \
    CANDIDATE_DOMAIN="$candidate_domain" \
    ADAM_LEARNING_RATE="${ADAM_LEARNING_RATE:-0.05}" \
    QNG_LEARNING_RATE="${QNG_LEARNING_RATE:-0.01}" \
    QNG_REGULARIZATION="${QNG_REGULARIZATION:-0.001}" \
    "$worker" "$worker_id" "$gpu_id" "$campaign_root" "$node_scratch" \
    >"$log" 2>&1 &
  pids+=("$!")
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "worker=$index status=OK"
  else
    rc=$?
    echo "worker=$index status=FAIL rc=$rc" >&2
    failed=$((failed + 1))
  fi
done

python3 "$repro_code/analyze_vqe_quantum_aware.py" "$campaign_root" \
  --pfda-root "$pfda_root" \
  --checkpoints 500 1000 2000 5000 10000 20000 || true

completed="$(find "$campaign_root/runs" -name RUN_SUCCESS -type f 2>/dev/null | wc -l)"
echo "completed=$completed/24"
if (( failed == 0 )) && [[ "$completed" -eq 24 ]]; then
  touch "$campaign_root/CAMPAIGN_SUCCESS"
  echo "Campaign complete: $campaign_root"
  exit 0
fi
echo "Campaign incomplete or failed; completed conditions remain resumable" >&2
exit 1
