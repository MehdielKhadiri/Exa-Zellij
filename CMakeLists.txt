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