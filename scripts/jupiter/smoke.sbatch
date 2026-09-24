#!/usr/bin/env bash
#SBATCH --job-name=fqo-smoke
#SBATCH --partition=booster
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH --gres=gpu:gh200:4
#SBATCH --time=00:03:00
#SBATCH --output=fqo-smoke-%j.out
set -euo pipefail
# Submit from the repository: sbatch --account=YOUR_ACCOUNT scripts/jupiter/smoke.sbatch
module load Stages/2026 GCC/14.3.0
module load Kokkos/4.7.03-CUDA-13 ParaStationMPI/5.13.0-1-mt
if [[ ${FQO_WITH_PYTHON:-0} == 1 ]]; then
  module load Python/3.13.5 SciPy-bundle/2025.07
fi
repo=${SLURM_SUBMIT_DIR:?Submit this job from the repository directory}
exe=${FQO_EXECUTABLE:-$repo/build-jupiter/fractal_optimizer}
mkdir -p "$repo/results/jupiter-smoke-$SLURM_JOB_ID"
cd "$repo/results/jupiter-smoke-$SLURM_JOB_ID"
export OMP_NUM_THREADS=1
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
for family in fbd fbo; do
  initial=()
  if [[ "$family" == fbo ]]; then initial=(--n-init 20); fi
  # FBD's positional budget is retained for legacy syntax; its work is structural.
  srun --gpus-per-task=gh200:1 "$exe" 4 3 80 1 shiftedsphere \
    --optimizer-family "$family" --max-depth 2 --seed 7 \
    --fbo-doe halton --fbo-doe-distribution distributed \
    --result-json "$family.json" "${initial[@]}"
  if [[ ${FQO_WITH_PYTHON:-0} == 1 ]]; then
    srun --gpus-per-task=gh200:1 "$exe" 4 3 80 1 external \
      --python-module embedded_objective --python-dir "$repo/tests" \
      --optimizer-family "$family" --max-depth 2 --seed 7 \
      --fbo-doe halton --fbo-doe-distribution distributed \
      --result-json "$family-python.json" "${initial[@]}"
  fi
done
python3 - <<'PY'
import json, math, os
for family in ('fbd', 'fbo'):
    variants = [('', 5, 1)]
    if os.environ.get('FQO_WITH_PYTHON') == '1':
        variants.append(('-python', 7, .37))
    for suffix, offset, centre in variants:
        with open(family + suffix + '.json') as stream:
            result = json.load(stream)
        expected = offset + sum((x - centre)**2 for x in result['x'])
        assert math.isclose(result['fun'], expected, rel_tol=1e-6, abs_tol=1e-6), result
        print('PASS', family + suffix, 'point/value consistency:', result['fun'])
PY
