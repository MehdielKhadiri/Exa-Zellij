# Fractal-Based Optimizers

A configurable optimization core in C++/Kokkos, with a Python interface, for:

- **Fractal-Based Decomposition (FBD)**, where decomposition, exploration,
  scoring, tree search, and exploitation are selected independently; and
- **Fractal Bayesian Optimization (FBO)**, where a surrogate and acquisition
  function guide the fractal search under a fixed objective-evaluation budget.

Use your own objective function, select a family and its components, and run on
Linux or WSL. MPI, NVIDIA CUDA, AMD HIP, and the VQE quantum-chemistry objective
are optional. **Quantum simulation is an addon, not a requirement of either
optimizer family.**

The project was previously called Fractal Quantum Optimizer. The source package
is now named `fractal-based-optimizers`; existing `from fqo import ...`, the
`fqo` command, `FQO_*` build options, and the `fractal_optimizer` executable
remain compatible. The GitHub URL and historical release environments still
use the old name. See the [installation guide](docs/user-guide.md) for upgrades.

## Start on Linux or WSL

Start with [QUICKSTART.txt](QUICKSTART.txt) for copy-and-paste installation and
first-run instructions, or the [full guide](docs/user-guide.md) for more options.

The supported operating system is Linux, including Ubuntu through WSL on a
Windows PC. Native Windows is outside the supported scope. The developing 0.3
interface runs on CPU without MPI or quantum dependencies. Source installation
needs Python 3.10+, Git and a C++20 compiler such as GCC 11+.

```sh
python -m pip install .
fqo doctor
python examples/python_quickstart.py
```

```python
from fqo import Config, minimize

def loss(x):
    return sum((v - 0.3)**2 for v in x)

result = minimize(loss, bounds=[(-5, 5)] * 3,
                  config=Config(family="fbo", budget=100,
                                components={"surrogate": "rbf"}))
print(result.x, result.fun, result.nfev)
```

For FBD use `Config(family="fbd", max_depth=3, beam=4)`: **FBD has no
evaluation-budget parameter**; its configuration determines its evaluation
count. FBO uses an evaluation budget. User callbacks and executable adapters
require no C++ edits. See the [user guide](docs/user-guide.md) for JSON
configuration, batched objectives and install details.

For accelerator builds, load your site's toolchain and run
`python configure.py --build`. The helper detects an existing Kokkos backend,
or CUDA/HIP compilers, with explicit target overrides for HPC login nodes.
Use [JUPITER](docs/jupiter.md) or [LUMI-G](docs/lumi-g.md) instructions on those
systems. See [validation status](docs/validation.md) before relying on a platform;
source installation and tested/published binary wheels are distinct.

## Optional VQE objective and research bundle

The repository also contains the hybrid VQE path exercised on Grid'5000 and LUMI-G:
the compiled optimizer calls a batched PennyLane/JAX objective through an
embedded Python interface, while MPI distributes independent fractal work and
Kokkos targets NVIDIA CUDA or AMD HIP accelerators.

## Main capabilities

1. The optimizer family is selected at runtime with
   `--optimizer-family fbd|fbo`.
2. FBD and FBO components are configurable at runtime through command-line
   options, and the effective configuration is printed at startup.
3. User objectives can be Python scalar/batch callbacks, executable adapters,
   embedded MPI Python modules, or compiled native functions.
4. The default CPU install has no MPI or quantum dependencies; site-specific
   CUDA/HIP builds support GPU execution and optional MPI distribution.
5. Python and JSON interfaces expose bounds, family configuration, seeds,
   result coordinates, objective values, and evaluation counts.
6. VQE is an optional objective adapter. Its simulator environments and
   Grid'5000/LUMI reproducibility bundle are retained separately from core use.

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
├── python/fqo/              Python API, CLI, and lazy VQE addon
├── python/vqe_*.py          Optional PennyLane/JAX VQE evaluator
├── data/                    Portable LiH/STO-3G Pauli-term reference
├── examples/                Small FBD, FBO, and VQE launch examples
├── scripts/lumi/            LUMI-G build/run material
├── reproducibility/         Grid'5000/LUMI protocols, results, and verifier
├── docs/                    Configuration and hybrid-interface documentation
└── tests/                   Dependency-light release integrity checks
```

## Verify the published result artifacts

No accelerator or quantum dependencies are required to check the retained
measurements and recompute the central paper-facing outcomes:

```bash
python reproducibility/verify_results.py
```

The verifier checks file hashes, reconstructs the equal-call and
knowledge-driven win/tie/loss counts, validates all 195 robustness runs, and
recomputes the LUMI-G scaling and solution-quality quantities.  Full provenance
and rerun instructions are in
[`reproducibility/README.md`](reproducibility/README.md).

## Advanced GPU/HPC build

Requirements:

- CMake 3.20 or newer and a C++20 compiler;
- MPI when distributed execution is enabled;
- a CUDA- or HIP-enabled Kokkos and Kokkos Kernels installation;
- pybind11 only when enabling the embedded Python objective adapter;
- CUDA Toolkit for NVIDIA builds, or rocBLAS/rocSOLVER for AMD builds.

CUDA:

```bash
cmake -S . -B build-cuda -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=nvcc_wrapper \
  -DFQO_GPU_BACKEND=CUDA -DFQO_ENABLE_MPI=ON -DFQO_ENABLE_PYTHON=OFF \
  -DCMAKE_PREFIX_PATH="/path/to/kokkos;/path/to/kokkos-kernels"
cmake --build build-cuda --parallel 4
ctest --test-dir build-cuda --output-on-failure
```

HIP:

```bash
cmake -S . -B build-hip -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=hipcc \
  -DFQO_GPU_BACKEND=HIP -DFQO_ENABLE_MPI=ON -DFQO_ENABLE_PYTHON=OFF \
  -DCMAKE_PREFIX_PATH="/path/to/kokkos;/path/to/kokkos-kernels"
cmake --build build-hip --parallel 4
ctest --test-dir build-hip --output-on-failure
```

For embedded Python objectives or the native VQE example below, rebuild with
`-DFQO_ENABLE_PYTHON=ON` and add your pybind11 installation to
`CMAKE_PREFIX_PATH`. This is separate from the ordinary Python API, which
does not need Python embedding. See the [objective adapter guide](docs/user-guide.md#external-objectives-on-mpi-ranks).

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

For Grid'5000, use
[`reproducibility/grid5000/README.md`](reproducibility/grid5000/README.md).
For LUMI-G, use [docs/lumi-g.md](docs/lumi-g.md) and
[`scripts/lumi/vqe_sweep.sbatch`](scripts/lumi/vqe_sweep.sbatch).

## Configuration

The complete runtime component matrix is documented in
[docs/configuration.md](docs/configuration.md). Ready-to-edit examples are in
[`examples/`](examples/).

## Research status

This is a research release assembled from the implementation used for the
configurable FBD/FBO and Grid'5000/LUMI-G VQE experiments. It favors traceable
runtime configuration and reproducible launch material over a stable library
API.  Result files are labeled by provenance: retained run-level output,
hash-indexed execution evidence, aggregate output, or reported coordinates for
which a detailed source trace is not present in the archived experiment data.

## License

CeCILL-C Free Software License Agreement, version 1.0. See
[LICENSE](LICENSE).
