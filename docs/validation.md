# Tested configurations

Tests were run on 20 September 2026, with job records checked on 24 September.
They cover the developing 0.3 interface. The earlier LUMI-G research runs are
listed separately below.

| Configuration | Result |
| --- | --- |
| Linux x86-64 / WSL, GCC 11.4, Kokkos 4.5.01 and Eigen, without MPI or embedded Python | Build, CPU solver tests and Python API tests passed |
| CPU Python wheel, Linux x86-64 / Python 3.12 | Built and installed; nine API tests, the FBD/FBO example, JSON CLI and `doctor` passed |
| CPU with Open MPI 5.0.6, Python 3.12 and pybind11 3.0.1 | Five CTest groups passed, including two-process runs of both families with analytic and Python objectives |
| JUPITER ARM CPUs and NVIDIA GH200 GPUs, CUDA 13, Kokkos 4.7.03 / Kernels 4.7.01 and ParaStationMPI 5.13.0-1-mt | Four-process runs of both families passed with analytic and embedded Python objectives; all four returned point/value checks passed |
| AMD HIP / LUMI-G | Earlier research runs are available in the [LUMI-G results](../reproducibility/lumi_g/README.md); the developing 0.3 interface has not been rerun on AMD |
| AMD HIP / Grid'5000 MI210 | Job 334968 stopped during setup on 21 September 2026, before building or running the tests |
| Restart with a different number of MPI processes | Implemented through `--ft-elastic-resume`; a runtime test report for this version has not been added |

Windows users run the software through WSL. Native Windows builds are not supported.

## CPU and objective interfaces

The tests cover FBD and FBO callbacks, all six FBD geometries, and the PCA-GP,
IDW, RBF and no-surrogate FBO options. They compare scalar and batched
evaluations, run external executable objectives, and check invalid settings,
nonfinite values, callback exceptions and unknown objective names.

The native and MPI tests recompute the objective at the returned solution.
They include FBD's off-centre probes and Python batches with strided float32
output. The FBO tests run the search after its initial design of experiments.
FBD's settings determine its evaluation count. The user-specified evaluation
cap is enforced by the FBO Python interface.

## JUPITER

The two one-node jobs took 81 and 116 seconds, including startup and cleanup.
Optimizer times are stored separately in their JSON results. Each run used
four MPI processes, one per GPU. Native analytic objectives and optimizer
kernels ran on the GPUs. The embedded Python test used NumPy on the CPU.

The Python bridge used Python 3.13.5, NumPy 2.3.2 and pybind11 3.0.1.
Automatic configuration detected the site's CUDA Kokkos installation and compiler.
The [MPI test](../tests/test_mpi.py) and
[JUPITER job script](../scripts/jupiter/smoke.sbatch) contain the test settings.

## Grid'5000 AMD attempt

The Larochette job ran from 00:35:47 to 00:35:49 CEST and returned exit code 1.
Its error output reports a missing user `.bashrc`. It stopped before producing
build or test logs, so the HIP build and execution still need to be tested.
The launch instructions are in [grid5000-hip.md](grid5000-hip.md).

## Earlier results and remaining tests

The saved research results pass the result verifier. These checks recompute
values from the existing files; they do not rerun the optimizations. The
developing interface changes external FBD scoring, MPI best-solution
coordinates and VQE error handling, which can affect comparisons with older
runs. CPU and GPU results are compared numerically; their search paths need
not be identical bit for bit.

The PennyLane/JAX simulator has not been rerun for this interface, and the
NumPy bridge test does not exercise it or a physical QPU. AMD execution and
distributable Linux wheels remain to be validated before release. CI targets
Linux and does not require a native Windows build.
