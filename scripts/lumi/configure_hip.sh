#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$REPO_ROOT/build-lumi-hip}"

module load LUMI/24.03 partition/G
module load PrgEnv-amd rocm craype-accel-amd-gfx90a
module load EasyBuild-user
module load Kokkos
module load Kokkos-kernels

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DFQO_GPU_BACKEND=HIP \
  -DFQO_ENABLE_URANIE=OFF \
  "$@"

cmake --build "$BUILD_DIR" -j
ctest --test-dir "$BUILD_DIR" --output-on-failure
