# Exa-Zellij: Ultra-Scale Fractal-Based Optimization framework

This repository provides scalable implementations of **fractal-based decomposition optimization algorithms** designed for ultra-scale systems. It supports execution from **single-GPU intranode setups** to **multi-GPU/multi-node internode environments** using modern C++ parallel programming models.

The approach integrates recursive **fractal space decomposition** with **Bayesian optimization techniques**, enabling efficient global search in high-dimensional domains with limited evaluation budgets.

---

## 🔧 Key Technologies

- **Kokkos** for node-level performance portability and GPU/CPU parallelism  
- **MPI** for internode communication and distributed workloads  
- **CUDA + SERIAL** back-ends for hybrid host/device computation  
- **Eigen** for linear algebra and Gaussian Process modeling  
- **C++20** and separable CUDA compilation support  

---

## 🧠 Optimization Strategy

- The search space is recursively partitioned into **hyperspherical regions** using a fractal decomposition strategy.  
- A **Gaussian Process (GP)** is fitted to known evaluations in each region.  
- The **Bayesian score** (posterior mean/uncertainty) ranks hyperspheres to guide exploration and exploitation.  
- This results in a **multi-scale**, **distributed**, and **Bayes-guided** search mechanism suited for large-scale and high-dimensional optimization problems.

> 📖 Related concepts: surrogate modeling, trust-region BO, space decomposition, ultra-scale search.

---

## 📁 Project Structure

```
.
├── mainfractalehighd.cpp    # Main program orchestrating the optimization
├── GPfunctions.cpp          # Gaussian Process regression and pre-scoring logic
├── HPCFunctions.cpp         # Parallelization, MPI logic, and decomposition tools
├── CMakeLists.txt           # Build configuration
└── README.md                # This file
```

---

## 🚀 Building the Project

### 🔧 Requirements

- **CMake ≥ 3.16**  
- **C++20** compatible compiler (e.g., GCC ≥ 10)  
- **CUDA Toolkit**  
- **MPI implementation** (e.g., OpenMPI with ULFM support)  
- **Kokkos** (installed with appropriate back-ends)  
- **Eigen3** (header-only library)  

### 🛠 CMake Setup for CUDA GPU

```cmake
cmake_minimum_required(VERSION 3.16)
project(KokkosCudaExample LANGUAGES CUDA CXX)

# Set MPI compiler explicitly
set(CMAKE_CXX_COMPILER your path to mpicxx)

# Kokkos installation path
list(APPEND CMAKE_PREFIX_PATH "your path to kokkos-install")

# Find dependencies
find_package(Kokkos REQUIRED)
find_package(MPI REQUIRED)
find_package(Eigen3 REQUIRED)

# Executable target
add_executable(kokkos_cuda_example
    mainfractalehighd.cpp
    GPfunctions.cpp
    HPCFunctions.cpp
)

# Target settings
set_target_properties(kokkos_cuda_example PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION ON
)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR} ${MPI_INCLUDE_PATH})

# Link libraries
target_link_libraries(kokkos_cuda_example
    PUBLIC
        Kokkos::kokkos
        MPI::MPI_CXX
        Eigen3::Eigen
)
```

### 🧪 Build & Run

To build the project:

```bash
mkdir build && cd build
cmake ..
make -j
```

To run the executable with MPI across multiple processes:

```bash
mpirun -np 8 ./kokkos_cuda_example <beta> <dimension> <budget> <expense_factor> <objective> --kokkos-map-device-id-by=mpi_rank
```

#### 🧾 Example (Grid5000 or similar multi-node setup with OAR):

```bash
mpirun --mca orte_base_help_aggregate 0 \
       -np 8 -npernode 2 -machinefile $OAR_NODEFILE \
       ./kokkos_cuda_example 50 10000 150 10 rastrigin \
       --kokkos-map-device-id-by=mpi_rank
```

- `--mca orte_base_help_aggregate 0`: Prevents OpenMPI from suppressing repeated warnings (optional but helpful for debugging).
- `-np 8`: Run with 8 MPI processes.
- `-npernode 2`: Launch 2 processes per node to match the GPU availability.
- `-machinefile $OAR_NODEFILE`: Specifies the list of nodes allocated by the OAR job scheduler.
- `./kokkos_cuda_example 50 10000 150 10 rastrigin`: Run the program with:
  - `50` as the beta parameter (controls decomposition granularity),
  - `10000` as the problem dimensionality,
  - `150` as the evaluation budget,
  - `10` as the model expense factor,
  - `rastrigin` as the selected test function.
- `--kokkos-map-device-id-by=mpi_rank`: Maps each MPI process to a unique GPU, assuming one GPU per rank.

> 📌 Make sure your `Kokkos` build supports device ID mapping by MPI rank, and that your nodes have enough GPUs for the number of ranks per node.



## 📚 References

- T. Firmin and E.-G. Talbi. *A fractal-based decomposition framework for continuous optimization*. Preprint, July 2022.

- T. Firmin and E.-G. Talbi. *Massively parallel asynchronous fractal optimization*. In *IEEE International Parallel and Distributed Processing Symposium Workshops (IPDPSW)*, pages 930–938, 2023.

- A. Nakib, L. Souquet, and E.-G. Talbi. *Parallel fractal decomposition based algorithm for big continuous optimization problems*. *Journal of Parallel and Distributed Computing*, 133:297–306, 2019.

- [Kokkos GitHub Repository](https://github.com/kokkos/kokkos)


## 📝 License

This project is released under the [CeCILL-C FREE SOFTWARE LICENSE AGREEMENT](./LICENSE).

---

## 🙋 Contributing

Issues and pull requests are welcome! Please open a GitHub issue if you'd like to:
- Propose improvements or features  
- Report a bug in decomposition or scoring  
- Share benchmark results or use cases
