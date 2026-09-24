# JUPITER

Use project storage, not a quota-limited home directory, for source, builds and
results. JUPITER Booster uses ARM CPUs and NVIDIA GH200 GPUs. Build natively;
do not copy x86 binaries from another machine.

```sh
bash scripts/jupiter/configure.sh
sbatch --account=YOUR_ACCOUNT scripts/jupiter/smoke.sbatch
```

Submit from the repository directory. The smoke job requests one node for at
most three minutes and runs both families with four MPI ranks, one per GPU.
FBD work is determined by its configuration. FBO receives a budget of 80.
The legacy native CLI still requires a positional budget token for FBD, which
does not impose an evaluation cap.

The recipe uses Stages/2026, GCC/14.3.0, Kokkos/4.7.03-CUDA-13,
ParaStationMPI/5.13.0-1-mt and CMake/3.31.8, plus fetched Kokkos Kernels 4.7.01.
It uses the site's `nvcc_wrapper` and MPI libraries. Recheck module availability
when the site changes stages. MPI modules are loaded inside the batch script.
The typed GH200 resource request avoids mixing typed and untyped GPU options.

The default site recipe builds the analytic/custom-pipe core without Python
embedding. Build and test the optional adapter using the site's Python 3.13.5
and SciPy-bundle/2025.07 modules:

```sh
FQO_WITH_PYTHON=1 bash scripts/jupiter/configure.sh
sbatch --account=YOUR_ACCOUNT --export=ALL,FQO_WITH_PYTHON=1 scripts/jupiter/smoke.sbatch
```

This also tests a custom Python objective for each family. Generic MPI objective
modules need NumPy in that interpreter. VQE additionally
needs its quantum/JAX environment; a successful core build does not validate
the simulator stack.

For manual automatic configuration after loading the same modules:

```sh
python3 configure.py --mpi --kernels-tag 4.7.01 --build
```

An existing Kokkos installation provides its compiled target architecture.
When building Kokkos from source for GH200 on a login node without a GPU, use
`--backend cuda --arch HOPPER90`. Run GPU executables only inside allocations.

Site references: [JUPITER documentation](https://apps.fz-juelich.de/jsc/hps/jupiter/)
and [Kokkos build configuration](https://kokkos.org/kokkos-core-wiki/get-started/configuration-guide.html).
