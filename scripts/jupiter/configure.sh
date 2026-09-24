#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
module load Stages/2026 GCC/14.3.0
module load Kokkos/4.7.03-CUDA-13 ParaStationMPI/5.13.0-1-mt CMake/3.31.8
python_options=(-DFQO_ENABLE_PYTHON=OFF)
if [[ ${FQO_WITH_PYTHON:-0} == 1 ]]; then
  module load Python/3.13.5 SciPy-bundle/2025.07
  python_options=(-DFQO_ENABLE_PYTHON=ON -DFQO_FETCH_PYBIND11=ON
    "-DPython_EXECUTABLE=$(command -v python3)")
fi
build=${FQO_BUILD_DIR:-$repo/build-jupiter}
cmake -S "$repo" -B "$build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER="$(command -v nvcc_wrapper)" \
  -DFQO_GPU_BACKEND=CUDA -DFQO_ENABLE_MPI=ON \
  "${python_options[@]}" -DFQO_KOKKOSKERNELS_TAG=4.7.01 \
  -DBUILD_TESTING=ON "$@"
cmake --build "$build" --parallel "${FQO_BUILD_JOBS:-4}"
# Only asset tests run on login nodes. GPU execution belongs in an allocation.
ctest --test-dir "$build" -R '^release_assets$' --output-on-failure
