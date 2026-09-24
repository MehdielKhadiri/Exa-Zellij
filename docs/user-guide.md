# Installation and use

This is the developing 0.3 interface. Platform validation is recorded separately
in `docs/validation.md`; a build recipe is not a promise that every platform has
been tested. The research data and previous launch commands remain available.

## Install on Linux or WSL

Linux is the supported operating system. On a Windows PC, run the commands
inside Ubuntu/WSL; native Windows is outside the supported scope.
Install Python 3.10+, Git and a C++20 compiler such as GCC 11+.
From the repository:

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install .
fqo doctor
python examples/python_quickstart.py
```

The source install builds the native CPU optimizer and fetches pinned missing
C++ dependencies. No GPU, MPI, NumPy, quantum package or Python development
library is required for the default core. This can take several minutes.
Ready-made platform wheels would remove the compiler requirement; they must be
built, tested and published before advertising a compiler-free public install.

## Supply an objective and choose the family

```python
from fqo import Config, minimize

def objective(x):
    return sum((v - 0.3)**2 for v in x)

fbo = minimize(objective, [(-5, 5)] * 3,
               config=Config(family="fbo", budget=100,
                             components={"decomposition": "trisection",
                                         "surrogate": "rbf"}))
fbd = minimize(objective, [(-5, 5)] * 3,
               config=Config(family="fbd", max_depth=3, beam=4,
                             components={"exploration": "center-axial"}))
print(fbo.x, fbo.fun, fbo.nfev, fbo.status)
print(fbd.x, fbd.fun, fbd.nfev, fbd.status)
```

FBO accepts an evaluation budget; the Python interface caps actual objective
calls, including initial/probe evaluations. `evaluation_limit` means it stopped
at this cap and returned the best evaluated point. FBD has no evaluation-budget
parameter: depth, beam, geometry and exploration determine its work. Both
families count actual evaluations, including duplicate points. Early completion
is possible. `fun` is the best value actually observed from your callback.

The API uses the compiled algorithm, with affine normalization of each bound
interval to [0,1]. Callbacks and returned points use the original coordinates.
This normalization affects distance-based search compared with an unnormalized
native run, and is part of the Python API's documented behavior. The default
FBO DOE is Halton so a first run needs no external design file. Other component
defaults remain those of the native implementation. Its initial design uses
one quarter of the requested budget (at least two points for GP surrogates),
leaving evaluations for search. Override it with `Config(n_init=...)`. This
avoids the legacy CLI's fixed 100-point initial design exhausting small budgets.

Callbacks receive a list and return one finite number. Optionally pass
`batch_objective=...` accepting a list of rows and returning one value per row.
Errors and nonfinite results raise exceptions; they are not silently penalized.
`timeout` bounds waits for the native engine. An in-process user callback must
implement its own timeout if it can block indefinitely.

## An objective implemented in another language

Wrap a library with a Python callable (for example ctypes, a Python extension,
or an existing solver API), or use the executable adapter:

```python
from fqo import CommandObjective
objective = CommandObjective(["/path/to/my-simulator", "--evaluate"], timeout=60)
```

The child reads `{"x":[0.1,0.2]}` from stdin and writes one JSON number to stdout.
Diagnostics go to stderr. Arguments are passed without a shell. A fresh process
per evaluation is convenient for expensive simulations; use a persistent or
batched callable for cheap objectives. External software still needs its own
runtime and a wrapper matching this contract.

## Run from an editable JSON configuration

```sh
fqo run examples/quickstart.json --output result.json
```

The example imports `user_objective:loss` relative to the configuration file.
Component names match `docs/configuration.md`, without their `--fbo-`/`--fbd-`
prefix. Unknown names fail explicitly. Keep objective modules/configurations
under version control together with the returned configuration and seed.

## CMake and HPC installations

The automatic configure helper inspects an existing Kokkos installation first,
then available CUDA/HIP compilers, then selects CPU. It selects the compiler
before CMake configures the project:

```sh
python configure.py --build
python configure.py --dry-run
# Explicit compute-node targets when a login node cannot detect its GPUs:
python configure.py --backend cuda --arch HOPPER90 --mpi --build
python configure.py --backend hip --arch VEGA90A --mpi --build
```

Load the HPC site's modules first. If both toolchains are present without an
existing Kokkos configuration, select the intended backend explicitly. Kokkos
detects the device architecture where supported; `--arch` overrides it when
building Kokkos. An already installed Kokkos retains its compiled architectures.
Use `--prefix` for dependencies, `--cxx` for a custom compiler, and a fresh
`--build-dir` when changing toolchains. The default Python wheel intentionally
targets CPU for portability; CUDA/HIP installations use a site/device build.
CPU, NVIDIA CUDA and AMD HIP are the implemented math backends. Other Kokkos
backends need solver adapters and tests before they can be advertised.

```sh
cmake -S . -B build-cpu -DFQO_GPU_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu --parallel 2
ctest --test-dir build-cpu --output-on-failure
cmake --install build-cpu --prefix /your/install/prefix
```

Use `-DFQO_ENABLE_MPI=ON` for a real MPI build, `-DFQO_GPU_BACKEND=CUDA|HIP`
for accelerators, and `-DFQO_ENABLE_PYTHON=ON` for embedded objective modules.
The compiler must match the Kokkos installation. Supply dependency prefixes via
`CMAKE_PREFIX_PATH`; `-DFQO_FETCH_DEPS=OFF` prohibits dependency downloads.
For offline builds, preinstall dependencies or provide CMake FetchContent source
directories. Build in a fresh directory when changing backend/compiler.

Use site-native MPI and GPU libraries on HPC. The PC wheel is single-process;
it deliberately rejects accidental multi-rank launches. Set `FQO_EXECUTABLE` to
a custom build when using the Python driver. Pipe callbacks are single-rank;
distributed objectives use the embedded module adapter below.

## External objectives on MPI ranks

Build with Python embedding enabled and make NumPy available to that interpreter.
Your module defines `evaluate(x)` and optionally `evaluate_batch(rows)` and
`init(dimension)`. Each rank initializes its own module instance. Batch output
must be a one-dimensional array/list with one finite value per input row.

```sh
mpirun -np 2 fractal_optimizer 4 3 80 1 external \
  --python-module my_objective --python-dir /path/to/my/code \
  --optimizer-family fbo --fbo-doe halton --n-init 20 --lower-bound -5 --upper-bound 5
```

Use `srun` on JUPITER. Native objective bounds are uniform across dimensions;
the Python PC API supports heterogeneous bounds through normalization.
Callbacks must be safe to evaluate independently on each MPI rank.

## VQE add-on and new components

VQE remains an optional objective. Its existing module and reproducibility
environments are retained. Install `.[vqe]` for the recorded simulator dependency
versions, also listed in `requirements-vqe.txt`. This chemistry stack requires
a platform supported by PySCF; use Linux/WSL for it on a Windows PC. Open-shell
molecules additionally require the OpenFermion/PySCF adapter used by the
reference objective.
GPU JAX and quantum simulator variants require platform-specific installation;
installing the extra does not configure an HPC GPU runtime.

To use VQE from the installed PC API, import `from fqo import vqe`, call
`dimension = vqe.init("H2", "STO-3G")`, then pass `vqe.eval` and
`batch_objective=vqe.eval_batch` to `minimize`, with the documented VQE bounds.
The add-on is loaded lazily and keeps one molecular session per process.
The embedded `--objective vqe` path retains its dimension-aware trust box.
See `docs/quantum-interface.md` for simulator and dataset setup.

Custom objective adapters require no optimizer rebuild. New device surrogates
currently do: add the kind/parser in `FBOTypes.hpp`/`FBOConfig.cpp`, implement
training/prediction in `FBOSurrogateModels.*` or `GPfunctions.*`, and connect it
in `main.cpp`. Add numerical and optimization tests. This is a documented source
extension point, not yet a stable runtime surrogate plugin API.
