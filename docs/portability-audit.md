# Portability and public API audit

Audit baseline: commit `94f6d76`, 2026-09-20. The latest local tag is v0.1.0;
the changelog already describes 0.2.0, but CMake still reports 0.1.0.

The configurable FBD/FBO implementation was present at that baseline. It was
a research executable with the following gaps in end-user portability:

| Requirement | Baseline finding | Acceptance check |
| --- | --- | --- |
| Ordinary PC installation | Requires GPU Kokkos, MPI, pybind11 and GPU solvers | Build and run both families with none of these runtimes |
| Optional VQE | Quantum Python packages are separate, but Python embedding is mandatory | Analytic core builds without Python development libraries |
| External objectives | Callbacks exist internally, but only VQE is exposed | Run a user callable and a user executable without editing C++ |
| Friendly configuration | Many CLI options; no Python package or config file | Install package, choose family/components, obtain a structured result |
| Failures | Unknown objective silently selects Sphere; batched VQE errors become penalties | Invalid configuration and callback errors fail explicitly |
| FBD external scoring | Non-additive probe scoring calls the built-in dispatcher, whose fallback is Sphere | Probe evaluations must invoke the registered external objective |
| MPI FBD incumbent | Cell-ID exchange reconstructs a centre although the value can belong to a probe | Returned coordinates must reproduce the returned value across ranks |
| Extensibility | Surrogate enums and implementations are coupled to device kernels | Document the existing extension points and their rebuild requirement |
| Validation | Asset checks only; no executable CI matrix | Numerical and end-to-end tests on CPU, MPI and available GPUs |
| JUPITER | No platform recipe; ARM build not validated | Site-native build and bounded single-node FBD/FBO tests |

Implementation order: portable compiled core, installable Python/CLI interface,
objective adapters, executable regression tests, then JUPITER deployment.
Preserve the existing algorithm choices and archived result files. GPU and CPU
floating-point trajectories are not promised to be bitwise identical.

FBD intentionally has no user-specified evaluation budget: its geometry,
exploration, depth and beam determine its work. Preserve this contract. FBO
accepts an evaluation budget. Both interfaces should report actual evaluations.

Current scope: Linux, including WSL, with CPU/CUDA/HIP builds. Native Windows
was removed from scope at the user's request. Backend and ARM support must be
distinguished from actual validation. CI definitions alone are not evidence of
passing tests. A compiler-free `pip install` experience requires built and
published platform wheels; source installation still needs a C++ compiler.
