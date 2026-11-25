# Hybrid quantum objective interface

The optimizer proposes parameter values. The quantum evaluator runs the
corresponding circuit and returns its energy. This interface lets you change
the evaluator while keeping the optimization algorithm.

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

## VQE simulator

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
actual number of UCCSD parameters and sets the search bounds according to
that dimension, as in the LUMI-G implementation.

## Connecting a physical QPU

To connect a QPU, implement a Python module with the same three functions:

```python
def init(molname: str, basis: str) -> int:
    """Create the circuit/Hamiltonian session and return parameter count."""

def eval(theta) -> float:
    """Submit one parameter vector and return its measured energy."""

def eval_batch(theta_2d):
    """Submit a batch and return one energy per row."""
```

The adapter translates the circuit into the device's supported gates, chooses
the number of repeated measurements, submits the job and retrieves the result.
It also handles authentication and any error mitigation. Credentials must remain in the
provider environment or secret store; they must never be committed.

The batch function can group several parameter vectors into one provider job.
MPI distributes the fractal searches across classical workers. A provider
adapter is required: this repository supplies the simulator interface.

## Portable Pauli reference

[`data/lih_sto3g_pauli.json`](../data/lih_sto3g_pauli.json) is an exported
LiH/STO-3G Pauli-term representation. You can inspect it or use it to prepare a Hamiltonian for a QPU provider;
the current PennyLane path reconstructs the Hamiltonian directly.
