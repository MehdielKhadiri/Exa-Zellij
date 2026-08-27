# Grid'5000 reproduction

The launch material targets an NVIDIA A100 node on the Nantes Ecotaxe cluster.
Each worker owns one GPU and one sequential optimizer run; independent methods
or conditions execute concurrently without sharing result files.

## Prerequisites

- an OAR allocation on a Grid'5000 GPU cluster;
- Apptainer or Singularity;
- a CUDA-capable image containing Python, NumPy, SciPy, JAX, PennyLane,
  PennyLane-Lightning, PySCF, and h5py;
- this repository visible from the allocated node.

Set the portable paths before submission:

```bash
export PROJECT_ROOT="$HOME/fractal-quantum-optimizer"
export IMAGE="$HOME/path/to/vqe-a100.sif"
```

The published sequential experiments used `fp64`.  The worker scripts set
`VQE_DTYPE=fp64`, restrict CPU numerical libraries to one thread, assign one
full GPU to each worker, and disable JAX preallocation.

## Black-box equal-call experiment

Edit the OAR resource header and desired `MOLECULES_LIST`/`SEED_ORDER`, then:

```bash
oarsub -S reproducibility/grid5000/oar/grid5000_vqe_equal_count_one_node_nantes.oar.sh
```

The six methods are mapped onto successive GPU waves.  Completed conditions
are detected through `RUN_SUCCESS`; a later reservation can reuse the same
`RESULT_ROOT` and resume rather than repeat them.

## Knowledge-driven experiment

After the PFDA-BS equal-call result root exists:

```bash
export PFDA_ROOT="$PROJECT_ROOT/results/vqe_pfda_equal_count_beam2_all_molecules_b20k_v1"
oarsub -S reproducibility/grid5000/oar/grid5000_vqe_quantum_aware_full_nantes.oar.sh
```

The three GPUs independently execute parameter-shift Adam, ExcitationSolve,
and diagonal QNG.  Quantum work is recorded as full-Hamiltonian energy circuits
plus auxiliary metric circuits.

## Robustness experiment

The stage/worker pair preserves the 13-configuration, three-molecule,
five-seed protocol.  It is normally called from a two-node OAR wrapper after
both nodes have been staged.  The complete protocol and analyzer are in
`code/`; the retained measurements are in `../local/results/robustness/`.

## 30-qubit evaluator preflight

`grid5000_vqe_30q_preflight_nantes.oar.sh` is a checkpointed evaluator-only
diagnostic for a 30-qubit LiH/cc-pVDZ active space.  It is not a complete
PFDA-BS optimization and is not used by `verify_results.py` as a paper result.
It is included to make the large-statevector feasibility path inspectable and
rerunnable.

## Portable overrides

The launchers accept these environment variables:

| Variable | Purpose |
|---|---|
| `PROJECT_ROOT` | Repository checkout visible on compute nodes |
| `IMAGE` | CUDA VQE Apptainer/Singularity image |
| `RESULT_ROOT` | Persistent, resumable output directory |
| `REPRO_ROOT` | Reproducibility directory if moved externally |
| `VQE_DTYPE` | Evaluator precision; reported sequential runs use `fp64` |

Cluster names and walltimes in `#OAR` lines are examples of the executed
Nantes protocol.  Adapt them to current Grid'5000 availability and policy.
