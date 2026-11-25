# Tested configurations

Exa-Zellij has been tested on CPU, with MPI, and on NVIDIA and AMD GPUs.
The table below gives the configurations and what ran on each device.
The CPU, interface and one-node JUPITER tests were run in September 2026;
the JUPITER checkpoint and restart tests were run on 24 October.

| Configuration | What was tested | Result and record |
| --- | --- | --- |
| Linux x86-64 / WSL, GCC 11.4, Kokkos 4.5.01 and Eigen | CPU build, both optimizer families and Python objective interfaces | CPU solver and API tests passed; details below |
| Linux x86-64, Python 3.12 wheel | Installation, Python API, JSON CLI and first-run example | Nine API tests, the FBD/FBO example and `doctor` passed |
| Open MPI 5.0.6, Python 3.12 and pybind11 3.0.1 | Two-process FBD/FBO runs with analytic and embedded Python objectives | Five CTest groups passed; [MPI tests](../tests/test_mpi.py) |
| JUPITER, four GH200 GPUs on one Booster node | Four MPI processes; optimizer and native analytic objective on GPUs; separate NumPy objective on CPU | Both families passed all four returned point/value checks; [job script](../scripts/jupiter/smoke.sbatch) |
| JUPITER, 1, 2, 10, 25 and 50 Booster nodes | MPI/CUDA checkpoint and recovery for both families, up to 200 processes | Passed; [restart results](validation/jupiter-restart.md) |
| JUPITER, 25 and 50 Booster nodes | Restart from 100 to 200 MPI processes and from 200 to 100 | Both families passed with three seeds; [restart results](validation/jupiter-restart.md) |
| LUMI-G, AMD MI250X, eight MPI processes per node | Earlier VQE runs up to 1,000 nodes and 8,000 processes; optimizer through Kokkos/HIP and objective through PennyLane/JAX on GPUs | [Configuration](lumi-g.md), [results and available logs](../reproducibility/lumi_g/README.md) |

## CPU and objective interfaces

The tests cover FBD and FBO callbacks, all six FBD geometries, and the PCA-GP,
IDW, RBF and no-surrogate FBO options. They compare scalar and batch
evaluations, run external executable objectives, and check invalid settings,
nonfinite values, callback exceptions and unknown objective names.

The native and MPI tests evaluate the objective again at the returned
solution to check that its coordinates and value agree. They include FBD
exploration points away from fractal centres and Python batches with
strided float32 output. FBO tests include search after the initial design
and check the evaluation limit in the Python interface.

See the [API tests](../tests/test_api.py) and [MPI tests](../tests/test_mpi.py).
Windows users run this installation through WSL.

## JUPITER GPU and restart tests

The one-node tests used CUDA 13, Kokkos 4.7.03, Kokkos Kernels 4.7.01 and
ParaStationMPI 5.13.0-1-mt. There was one MPI process per GPU. The embedded
Python test used Python 3.13.5, NumPy 2.3.2 and pybind11 3.0.1. This NumPy
objective ran on CPU; native analytic objectives ran on GPU.

The October tests used the source at commit `a975949`. They cover process
loss, damaged files, backup recovery and restart with a different number
of MPI processes. A separate node-aggregation version also passed these
checks. The [restart report](validation/jupiter-restart.md) identifies the
two versions and gives their results.

## AMD GPUs

The LUMI-G records document HIP execution of the earlier VQE implementation.
They include both optimizer kernels and GPU simulation of the objective.
The [LUMI-G results](../reproducibility/lumi_g/README.md) identify which
measurements have detailed logs and which have only scheduler records.

## Configurations still needing tests

The September Python interface has not been rerun with the PennyLane/JAX
VQE environment. The embedded NumPy tests cover the Python connection,
not the VQE simulator or a physical QPU. Additional Linux wheels also need
testing before distribution. Native Windows builds are not supported.
Intel GPU execution through Kokkos/SYCL needs application-side
linear-algebra support and runtime tests.
