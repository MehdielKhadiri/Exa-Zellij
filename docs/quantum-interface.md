# Hybrid quantum objective interface

The optimizer and quantum evaluator are deliberately separated.

```text
MPI/Kokkos FBD or FBO
        |
        | C ABI: init + scalar/batched evaluation
        v
embedded Python (pybind11)
        |
        | init(molecule, basis)
        | eval(theta)
        | eval_batch(theta_batch)
        v
PennyLane/JAX backend
```

## Existing implementation

[`src/py_vqe_shim.cpp`](../src/py_vqe_shim.cpp) embeds one Python interpreter
per MPI process. It imports [`python/vqe_bridge.py`](../python/vqe_bridge.py),
obtains the ansatz dimension, and registers scalar and batched callbacks with
the optimizer.

The Python backend:

1. obtains a molecular geometry and reference energy from PennyLane qchem;
2. constructs the Jordan–Wigner qubit Hamiltonian;
3. builds a UCCSD ansatz;
4. selects `lightning.kokkos`, `lightning.gpu`, `lightning.qubit`, or
   `default.qubit`, in that order of availability;
5. JIT-compiles scalar and batched energy evaluations with JAX; and
6. pads calls to the fixed `VQE_BATCH` size to avoid recompilation for every
   batch shape.

For VQE, the executable replaces the placeholder command-line dimension by the
actual number of UCCSD parameters and uses the dimension-aware trust box from
the LUMI-G implementation.

## Replacing the emulator with a physical QPU

A hardware adapter does not need to modify the FBD/FBO tree. It only needs a
Python module with the same three functions:

```python
def init(molname: str, basis: str) -> int:
    """Create the circuit/Hamiltonian session and return parameter count."""

def eval(theta) -> float:
    """Submit one parameter vector and return its measured energy."""

def eval_batch(theta_2d):
    """Submit a batch and return one energy per row."""
```

The adapter is responsible for transpilation, shot allocation, job polling,
error mitigation, and provider authentication. Credentials must remain in the
provider environment or secret store; they must never be committed.

The batch boundary is intentional. A QPU implementation can group independent
representatives into provider jobs, while MPI continues to distribute fractal
regions across classical workers.

## Portable Pauli reference

[`data/lih_sto3g_pauli.json`](../data/lih_sto3g_pauli.json) is an exported
LiH/STO-3G Pauli-term representation. It is included as a portable inspection
artifact and as a starting point for a provider-specific Hamiltonian adapter;
the current PennyLane path reconstructs the Hamiltonian directly.
