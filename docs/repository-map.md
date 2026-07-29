# Repository map

| Path | Purpose |
|---|---|
| `src/main.cpp` | Optimizer driver, distributed DOE, fractal tree, acquisition selection, exploitation, and checkpointing |
| `include/FBOTypes.hpp` | Shared decomposition, DOE, surrogate, kernel, exploitation, and budget enums |
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
