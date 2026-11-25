# Repository map

| Path | Purpose |
|---|---|
| `QUICKSTART.txt` | Linux/WSL installation and first objective |
| `python/fqo/` | Public Python API, JSON CLI, executable-objective adapter and optional VQE add-on |
| `configure.py` | CPU/CUDA/HIP build selection and explicit HPC target overrides |
| `src/main.cpp` | Optimizer driver, distributed initial sampling, fractal search, exploitation and checkpointing |
| `include/FBOTypes.hpp` | Definitions of decomposition, initial-design, surrogate, kernel, exploitation and budget choices |
| `src/FBOConfig.cpp` | Runtime parsing and validation of FBO components |
| `src/FBDConfig.cpp` | Runtime parsing and validation of FBD components |
| `src/FBOSurrogateModels.cpp` | Non-GP surrogate alternatives |
| `src/GPfunctions.cpp` | PCA/HSIC GP training and GPU linear algebra |
| `src/HPCfunctions.cpp` | Fractal construction, objectives, parallel helpers, and CLI parsing |
| `src/py_vqe_shim.cpp` | C++/Python bridge registered as an external objective |
| `python/vqe_bridge.py` | Batched PennyLane/JAX VQE session |
| `python/vqe_cost_reference.py` | Molecular Hamiltonian and UCCSD construction |
| `scripts/lumi/` | LUMI-G build and Slurm launch material |
| `examples/` | Minimal component selections |
| `tests/test_release_assets.py` | Syntax, data, and accidental-secret checks |
| `tests/test_api.py` | User callbacks, components, bounds and error handling |
| `tests/test_mpi.py` | Distributed native/embedded-objective checks |
| `docs/validation.md` | Tested platforms, objective interfaces and restart results |
