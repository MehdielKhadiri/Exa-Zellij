# JUPITER

Store source files, builds and results in project storage. JUPITER Booster
uses ARM CPUs and NVIDIA GH200 GPUs, so build the executable on JUPITER.

```sh
bash scripts/jupiter/configure.sh
sbatch --account=YOUR_ACCOUNT scripts/jupiter/smoke.sbatch
```

Submit from the repository directory. The smoke job requests one node for at
most three minutes and runs both families with four MPI ranks, one per GPU.
FBD work is determined by its configuration. FBO receives a budget of 80.
The native command also requires the budget argument for FBD, but FBD
determines its evaluation count from its settings.

The recipe uses Stages/2026, GCC/14.3.0, Kokkos/4.7.03-CUDA-13,
ParaStationMPI/5.13.0-1-mt and CMake/3.31.8, plus fetched Kokkos Kernels 4.7.01.
It uses the site's `nvcc_wrapper` and MPI libraries. Recheck module availability
when the site changes stages. MPI modules are loaded inside the batch script.
The typed GH200 resource request avoids mixing typed and untyped GPU options.

The default build supports native analytic functions and external executable
objectives. To add embedded Python objectives, use the site's Python 3.13.5
and SciPy-bundle/2025.07 modules:

```sh
FQO_WITH_PYTHON=1 bash scripts/jupiter/configure.sh
sbatch --account=YOUR_ACCOUNT --export=ALL,FQO_WITH_PYTHON=1 scripts/jupiter/smoke.sbatch
```

This runs a NumPy objective for each family. The optimizer kernels run on
the GPU, while this Python objective runs on the CPU. Native analytic
objectives run on the GPU as well. VQE requires its own PennyLane/JAX
environment and a separate test; see the [test report](validation.md).

To run the configuration helper directly after loading the same modules:

```sh
python3 configure.py --mpi --kernels-tag 4.7.01 --build
```

An existing Kokkos installation provides its compiled target architecture.
When building Kokkos from source for GH200 on a login node without a GPU, use
`--backend cuda --arch HOPPER90`. Run GPU executables only inside allocations.

Site references: [JUPITER documentation](https://apps.fz-juelich.de/jsc/hps/jupiter/)
and [Kokkos build configuration](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html).
