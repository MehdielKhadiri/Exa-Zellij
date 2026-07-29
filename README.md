# Fractal Quantum Optimizer

A configurable MPI/Kokkos implementation of:

- **Fractal-Based Decomposition (FBD)**, where decomposition, exploration,
  scoring, tree search, and exploitation are selected independently; and
- **Fractal Bayesian Optimization (FBO)**, where a surrogate and acquisition
  function guide the fractal search under a fixed objective-evaluation budget.

The repository also contains the hybrid VQE path exercised on LUMI-G:
the compiled optimizer calls a batched PennyLane/JAX objective through an
embedded Python interface, while MPI distributes independent fractal work and
Kokkos targets NVIDIA CUDA or AMD HIP accelerators.

## Main capabilities

1. The optimizer family is selected at runtime with
   `--optimizer-family fbd|fbo`.
2. FBD and FBO components are configurable at runtime through command-line
   options, and the effective configuration is printed at startup.
3. `--objective vqe` activates the compiled C++/Python bridge, obtains the
   UCCSD parameter count from the chosen molecule, and uses batched objective
   calls.
4. The LUMI-G script maps one MPI rank to one MI250X GCD and records the
   configuration, result, and timing of each run.
5. The same optimizer can be smoke-tested without quantum dependencies on the
   included analytic objectives.

The present VQE backend is a PennyLane/JAX simulator path. It is a feasibility
implementation for hybrid HPC–quantum workflows, not a claim that a GPU
simulator is physical quantum hardware. A physical QPU adapter can implement
the same `init`, `eval`, and `eval_batch` Python interface described in
[docs/quantum-interface.md](docs/quantum-interface.md).

## Repository map

```text
.
├── include/                 Public C++ headers and runtime configuration
├── src/                     MPI/Kokkos FBD/FBO implementation and VQE shim
├── python/                  PennyLane/JAX VQE evaluator
├── data/                    Portable LiH/STO-3G Pauli-term reference
├── examples/                Small FBD, FBO, and VQE launch examples
├── scripts/lumi/            LUMI-G build/run material
├── docs/                    Configuration and hybrid-interface documentation
└── tests/                   Dependency-light release integrity checks
```

## Build

Requirements:

- CMake 3.20 or newer and a C++20 compiler;
- MPI;
- a CUDA- or HIP-enabled Kokkos and Kokkos Kernels installation;
- pybind11 2.12 or newer;
- CUDA Toolkit for NVIDIA builds, or rocBLAS/rocSOLVER for AMD builds.

CUDA:

```bash
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFQO_GPU_BACKEND=CUDA \
  -DCMAKE_PREFIX_PATH="/path/to/kokkos;/path/to/kokkos-kernels;/path/to/pybind11"
cmake --build build-cuda -j
ctest --test-dir build-cuda --output-on-failure
```

HIP:

```bash
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DFQO_GPU_BACKEND=HIP \
  -DCMAKE_PREFIX_PATH="/path/to/kokkos;/path/to/kokkos-kernels;/path/to/pybind11"
cmake --build build-hip -j
ctest --test-dir build-hip --output-on-failure
```

The optional HSIC-GP surrogate requires Uranie/ROOT and
`-DFQO_ENABLE_URANIE=ON`. The default PCA-GP and the remaining FBD/FBO
components do not require Uranie.

## First run without quantum dependencies

The executable accepts five positional arguments:

```text
fractal_optimizer <beam_cap> <dimension> <budget> <expense_factor> <objective>
```

For example:

```bash
mpirun -np 2 ./build-cuda/fractal_optimizer \
  32 100 200 1 sphere \
  --optimizer-family fbo \
  --fbo-decomposition hypersphere \
  --fbo-doe sobol \
  --fbo-doe-distribution distributed \
  --fbo-surrogate pca-gp \
  --fbo-gp-kernel rbf \
  --fbo-exploitation surrogate-local \
  --fbo-budget astra \
  --rho 0.8
```

Use `--list-objectives` to list the built-in and VQE objectives.

## VQE run

Create a local CPU/emulator environment:

```bash
conda env create -f environment.yml
conda activate fractal-quantum-optimizer
export VQE_PY_DIR="$PWD/python"
```

Then run a small LiH example:

```bash
export VQE_BATCH=64
mpirun -np 2 ./build-cuda/fractal_optimizer \
  32 1 80 1 vqe \
  --vqe-mol LiH \
  --vqe-basis STO-3G \
  --optimizer-family fbo \
  --fbo-decomposition hypersphere \
  --fbo-doe sobol \
  --fbo-doe-distribution distributed \
  --fbo-surrogate pca-gp \
  --rho 0.8
```

The placeholder dimension `1` is replaced at runtime by the UCCSD parameter
count returned by the VQE evaluator. The first online run may download
PennyLane quantum-chemistry data and compile JAX kernels.

For LUMI-G, use [docs/lumi-g.md](docs/lumi-g.md) and
[`scripts/lumi/vqe_sweep.sbatch`](scripts/lumi/vqe_sweep.sbatch).

## Configuration

The complete runtime component matrix is documented in
[docs/configuration.md](docs/configuration.md). Ready-to-edit examples are in
[`examples/`](examples/).

## Research status

This is a research release assembled from the implementation used for the
configurable FBD/FBO and LUMI-G VQE experiments. It favors traceable runtime
configuration and reproducible launch material over a stable library API.

## License

CeCILL-C Free Software License Agreement, version 1.0. See
[LICENSE](LICENSE).
