# LUMI-G workflow

The LUMI-G experiments use one MPI process per MI250X GPU die. Kokkos/HIP
runs the optimizer kernels, and PennyLane/JAX evaluates the VQE objective
on the GPUs inside the site ROCm container.

## 1. Clone and build

For a core-only HIP build, after loading a compatible site ROCm/CMake/MPI
environment, the new helper selects the backend/compiler before CMake starts:

```bash
python3 configure.py --backend hip --arch VEGA90A --mpi --build
```

Omit `--backend` when only the intended toolchain is loaded. Use `--prefix` for
a site Kokkos installation; add `--python` for the embedded objective bridge.
An installed Kokkos retains its compiled target architecture. The helper uses
`hipcc`; if the site's MPI setup requires its Cray wrapper, pass the configured
wrapper with `--cxx`. See the site's
[compiler instructions](https://docs.lumi-supercomputer.eu/development/compiling/prgenv/).
The developing 0.3 interface has not yet been tested on AMD; see
[the test report](validation.md).

To use the build script from the earlier VQE experiments:

```bash
git clone <repository-url> fractal-quantum-optimizer
cd fractal-quantum-optimizer
chmod +x scripts/lumi/*.sh
BUILD_DIR="$PWD/build-lumi-hip" scripts/lumi/configure_hip.sh
```

If CMake cannot discover pybind11, add its prefix:

```bash
BUILD_DIR="$PWD/build-lumi-hip" scripts/lumi/configure_hip.sh \
  -DCMAKE_PREFIX_PATH="/path/to/pybind11"
```

## 2. Prepare the Python layer

Use the LUMI ROCm/JAX container rather than installing CPU `jaxlib`. Create a
Python 3.12 venv on the project filesystem and install PennyLane, PySCF, h5py,
and the other packages listed in `requirements-vqe.txt`, omitting `jax` and
`jaxlib` when the container already supplies compatible ROCm builds.

The first online run downloads PennyLane qchem data into `XDG_CACHE_HOME`.
For offline compute nodes, prefetch the exact molecule/basis once and then set:

```bash
export VQE_OFFLINE=1
export XDG_CACHE_HOME="$HOME/pl-data-cache"
```

## 3. Edit only site/user placeholders

In `scripts/lumi/vqe_sweep.sbatch`:

- replace `#SBATCH --account=CHANGE_ME`;
- set the default `VENV` project path;
- verify the site container path in `SIF`.

## 4. Submit

```bash
sbatch --export=ALL,MOLS="LiH",BASES="STO-3G",BUDGET=80,MPI_RANKS=8 \
  scripts/lumi/vqe_sweep.sbatch
```

Multiple independent molecule/basis pairs may be comma-separated:

```bash
sbatch --export=ALL,MOLS="H2,LiH",BASES="STO-3G",BUDGET=160,MPI_RANKS=8 \
  scripts/lumi/vqe_sweep.sbatch
```

Each run writes a full log and one row in
`results/lumi-vqe-<job-id>/results.csv`.

## Binding and parallel work

`scripts/lumi/select_gpu.sh` maps `SLURM_LOCALID` to
`ROCR_VISIBLE_DEVICES`. The Slurm script uses eight ranks per LUMI-G node and
the documented CPU masks. FBD/FBO distribute fractal work across MPI ranks;
each rank evaluates its local batch through the selected PennyLane/JAX device.

## Recording a run

- Record the repository commit, container path/digest, loaded modules, venv
  package list, Slurm job ID, and `results.csv`.
- Keep the global evaluation budget fixed when comparing rank counts.
- JAX compilation and dataset staging should be separated from timed optimizer
  sections when reporting strong scaling.
- QPU credentials or provider tokens belong in the execution environment, not
  in scripts or logs committed to this repository.
