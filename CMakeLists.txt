cmake_minimum_required(VERSION 3.20)

project(fractal_quantum_optimizer VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FQO_GPU_BACKEND "AUTO" CACHE STRING "GPU backend: AUTO, CUDA, or HIP")
set_property(CACHE FQO_GPU_BACKEND PROPERTY STRINGS AUTO CUDA HIP)
option(FQO_ENABLE_URANIE "Enable the optional Uranie/ROOT HSIC surrogate" OFF)
option(FQO_FETCH_PYBIND11 "Fetch pybind11 when no local package is available" OFF)

find_package(Kokkos REQUIRED)
find_package(KokkosKernels REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  if(NOT FQO_FETCH_PYBIND11)
    message(FATAL_ERROR
      "pybind11 was not found. Install pybind11 or configure with "
      "-DFQO_FETCH_PYBIND11=ON.")
  endif()
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.12.0
  )
  FetchContent_MakeAvailable(pybind11)
endif()

if(FQO_GPU_BACKEND STREQUAL "AUTO")
  if(Kokkos_ENABLE_HIP)
    set(FQO_GPU_BACKEND "HIP")
  elseif(Kokkos_ENABLE_CUDA)
    set(FQO_GPU_BACKEND "CUDA")
  else()
    message(FATAL_ERROR
      "This release requires a GPU-enabled Kokkos build. Set "
      "-DFQO_GPU_BACKEND=CUDA or HIP if automatic detection fails.")
  endif()
endif()

add_executable(fractal_optimizer
  src/main.cpp
  src/FBOConfig.cpp
  src/FBDConfig.cpp
  src/FBOSurrogateModels.cpp
  src/GPfunctions.cpp
  src/HPCfunctions.cpp
  src/py_vqe_shim.cpp
)

target_include_directories(fractal_optimizer PRIVATE include)
target_compile_definitions(fractal_optimizer PRIVATE
  VQE_PY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/python"
)
target_link_libraries(fractal_optimizer PRIVATE
  Kokkos::kokkos
  KokkosKernels::kokkoskernels
  MPI::MPI_CXX
  pybind11::embed
)

if(FQO_GPU_BACKEND STREQUAL "CUDA")
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(fractal_optimizer PRIVATE CUDA::cusolver CUDA::cublas)
elseif(FQO_GPU_BACKEND STREQUAL "HIP")
  find_library(FQO_ROCBLAS_LIBRARY NAMES rocblas REQUIRED)
  find_library(FQO_ROCSOLVER_LIBRARY NAMES rocsolver REQUIRED)
  target_link_libraries(fractal_optimizer PRIVATE
    "${FQO_ROCBLAS_LIBRARY}"
    "${FQO_ROCSOLVER_LIBRARY}"
  )
else()
  message(FATAL_ERROR "Unsupported FQO_GPU_BACKEND=${FQO_GPU_BACKEND}")
endif()

if(FQO_ENABLE_URANIE)
  if("$ENV{URANIESYS}" STREQUAL "")
    message(FATAL_ERROR "FQO_ENABLE_URANIE=ON requires URANIESYS")
  endif()
  target_compile_definitions(fractal_optimizer PRIVATE HAVE_URANIE=1)
  separate_arguments(FQO_URANIE_CPP_FLAGS NATIVE_COMMAND "$ENV{URANIECPPFLAG}")
  separate_arguments(FQO_URANIE_LD_FLAGS NATIVE_COMMAND "$ENV{URANIELDFLAG}")
  list(FILTER FQO_URANIE_CPP_FLAGS EXCLUDE REGEX "^-std=c\\+\\+")
  target_compile_options(fractal_optimizer PRIVATE ${FQO_URANIE_CPP_FLAGS})
  target_link_options(fractal_optimizer PRIVATE ${FQO_URANIE_LD_FLAGS})
endif()

set_target_properties(fractal_optimizer PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(
    NAME release_assets
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_release_assets.py"
  )
endif()
