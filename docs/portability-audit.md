# Installation and portability

Exa-Zellij runs on Linux, including Ubuntu through WSL on Windows. The default
installation builds the CPU optimizer. MPI, GPU execution and VQE are optional,
so a first installation needs neither a cluster nor quantum software.

## Build options

| Use | Requirements |
| --- | --- |
| CPU and Python API | Python 3.10+, a C++20 compiler and CMake; missing C++ dependencies can be downloaded during the build |
| Several MPI processes | An MPI build using the site's MPI installation |
| NVIDIA GPUs | CUDA, CUDA-enabled Kokkos and Kokkos Kernels |
| AMD GPUs | HIP, HIP-enabled Kokkos and Kokkos Kernels, rocBLAS and rocSOLVER |
| Embedded Python objectives | Python development libraries, pybind11 and the objective's own dependencies |
| VQE simulation | The optional PennyLane/JAX environment described in the quantum guide |

`configure.py` checks an existing Kokkos installation, then available GPU
compilers, and otherwise selects CPU. On an HPC login node, specify the target
GPU explicitly when it cannot be detected. Use a fresh build directory when
changing compiler or backend. A source installation requires a compiler.

Intel GPU execution through Kokkos/SYCL still needs application-side
linear-algebra support and tests. Native Windows builds are not supported.

## Objectives and components

Users can supply a Python function, a batch function, an external executable
or an embedded Python module. These interfaces keep the objective separate
from the optimizer. New device surrogates require C++ changes and a rebuild.

FBD's configuration determines its number of evaluations. FBO takes an
evaluation budget. Both report the evaluations they performed. Invalid
settings and objective errors are reported explicitly.

The [user guide](user-guide.md) explains installation and objective interfaces.
The [configuration guide](configuration.md) lists the search components.

## Tests

The [validation report](validation.md) lists CPU, MPI, CUDA and HIP results,
along with checkpoint and restart tests. CPU and GPU results are checked
numerically; floating-point differences can lead to different search paths.
The September CPU/API tests also cover external FBD scoring, the coordinates
of the best solution exchanged through MPI, and errors in Python objectives.
