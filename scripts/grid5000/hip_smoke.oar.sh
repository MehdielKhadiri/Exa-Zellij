#!/usr/bin/env bash
#OAR -q default
#OAR -p larochette
#OAR -l nodes=1,walltime=00:20:00
#OAR -n fqo_hip_smoke
#OAR -O fqo_hip_smoke_%jobid%.out
#OAR -E fqo_hip_smoke_%jobid%.err
set -euo pipefail
repo=${FQO_SOURCE_DIR:-${OAR_WORKDIR:?Submit from the repository directory}}
out=$repo/results/grid5000-hip-${OAR_JOB_ID:?This script requires an OAR allocation}
mkdir -p "$out"
# Build products stay on the allocated node; preserve logs and the binary.
work=$(mktemp -d /tmp/fqo-hip-XXXXXX)
build=$work/build
finish() {
  code=$?
  trap - EXIT
  if [[ -f "$build/fractal_optimizer" ]]; then cp "$build/fractal_optimizer" "$out/"; fi
  printf '%s\n' "$code" > "$out/exit-code.txt"
  exit "$code"
}
trap finish EXIT
source /etc/profile
module load rocm/7.1.1-abaca rocsolver/7.1.1 cmake/4.1.3 openmpi/4.1.6 python/3.12.12
{
  hostname
  date --iso-8601=seconds
  module -t list
  hipcc --version
  rocm-smi --showproductname
} > "$out/environment.log" 2>&1
cd "$repo"
python3 configure.py --arch VEGA90A --mpi --build-dir "$build" --jobs 4 --build \
  > "$out/build.log" 2>&1
export OMP_NUM_THREADS=1
python3 tests/test_mpi.py --executable "$build/fractal_optimizer" \
  --mpiexec "$(command -v mpirun)" > "$out/mpi-tests.log" 2>&1
FQO_EXECUTABLE="$build/fractal_optimizer" python3 tests/test_api.py \
  > "$out/api-tests.log" 2>&1
printf 'PASS HIP build, two-rank native tests and Python API tests\n' | tee "$out/PASS.txt"
