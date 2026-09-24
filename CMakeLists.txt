cmake_minimum_required(VERSION 3.20)

project(fractal_quantum_optimizer VERSION 0.3.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(FQO_GPU_BACKEND "CPU" CACHE STRING "Execution backend: CPU, AUTO, CUDA, or HIP")
set_property(CACHE FQO_GPU_BACKEND PROPERTY STRINGS CPU AUTO CUDA HIP)
set(_fqo_hpc_default ON)
if(FQO_GPU_BACKEND STREQUAL "CPU")
  set(_fqo_hpc_default OFF)
endif()
option(FQO_ENABLE_MPI "Use system MPI for distributed execution" ${_fqo_hpc_default})
option(FQO_ENABLE_PYTHON "Build the optional embedded Python/VQE adapter" ${_fqo_hpc_default})
option(FQO_FETCH_DEPS "Download pinned missing build dependencies" ON)
option(FQO_USE_SYSTEM_KOKKOS "Prefer installed Kokkos/Kernels over fetched sources" ON)
option(FQO_ENABLE_URANIE "Enable the optional Uranie/ROOT HSIC surrogate" OFF)
option(FQO_FETCH_PYBIND11 "Fetch pybind11 when no local package is available" OFF)

include(cmake/Dependencies.cmake)
if(FQO_ENABLE_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
endif()

if(FQO_ENABLE_PYTHON)
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
    GIT_TAG v3.0.1
    GIT_SHALLOW TRUE
  )
  FetchContent_MakeAvailable(pybind11)
endif()
endif()

if(FQO_GPU_BACKEND STREQUAL "AUTO")
  if(Kokkos_ENABLE_HIP)
    set(FQO_GPU_BACKEND "HIP")
  elseif(Kokkos_ENABLE_CUDA)
    set(FQO_GPU_BACKEND "CUDA")
  else()
    set(FQO_GPU_BACKEND "CPU")
  endif()
endif()

add_executable(fractal_optimizer
  src/main.cpp
  src/FBOConfig.cpp
  src/FBDConfig.cpp
  src/FBOSurrogateModels.cpp
  src/GPfunctions.cpp
  src/HPCfunctions.cpp
)
if(FQO_ENABLE_PYTHON)
  target_sources(fractal_optimizer PRIVATE src/py_vqe_shim.cpp)
  target_link_libraries(fractal_optimizer PRIVATE pybind11::embed)
else()
  target_sources(fractal_optimizer PRIVATE src/python_disabled.cpp)
endif()

target_include_directories(fractal_optimizer PRIVATE include)
target_compile_definitions(fractal_optimizer PRIVATE
  VQE_PY_DIR="${CMAKE_CURRENT_SOURCE_DIR}/python"
)
target_link_libraries(fractal_optimizer PRIVATE
  Kokkos::kokkos
  KokkosKernels::kokkoskernels
)
if(FQO_ENABLE_MPI)
  target_compile_definitions(fractal_optimizer PRIVATE FQO_ENABLE_MPI=1)
  target_link_libraries(fractal_optimizer PRIVATE MPI::MPI_CXX)
endif()
target_compile_definitions(fractal_optimizer PRIVATE FQO_VERSION="${PROJECT_VERSION}")
if(MSVC)
  target_compile_definitions(fractal_optimizer PRIVATE _USE_MATH_DEFINES NOMINMAX)
  target_compile_options(fractal_optimizer PRIVATE /bigobj /utf-8)
endif()

if(FQO_GPU_BACKEND STREQUAL "CUDA")
  if(NOT Kokkos_ENABLE_CUDA)
    message(FATAL_ERROR "CUDA selected but the installed Kokkos lacks CUDA. Use a matching Kokkos prefix.")
  endif()
  find_package(CUDAToolkit REQUIRED)
  target_link_libraries(fractal_optimizer PRIVATE CUDA::cusolver CUDA::cublas)
elseif(FQO_GPU_BACKEND STREQUAL "HIP")
  if(NOT Kokkos_ENABLE_HIP)
    message(FATAL_ERROR "HIP selected but the installed Kokkos lacks HIP. Use a matching Kokkos prefix.")
  endif()
  find_library(FQO_ROCBLAS_LIBRARY NAMES rocblas REQUIRED)
  find_library(FQO_ROCSOLVER_LIBRARY NAMES rocsolver REQUIRED)
  target_link_libraries(fractal_optimizer PRIVATE
    "${FQO_ROCBLAS_LIBRARY}"
    "${FQO_ROCSOLVER_LIBRARY}"
  )
elseif(FQO_GPU_BACKEND STREQUAL "CPU")
  if(Kokkos_ENABLE_CUDA OR Kokkos_ENABLE_HIP OR Kokkos_ENABLE_SYCL)
    message(FATAL_ERROR "CPU mode requires a CPU Kokkos installation; use a fresh build directory.")
  endif()
  target_link_libraries(fractal_optimizer PRIVATE Eigen3::Eigen)
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

if(UNIX)
set_target_properties(fractal_optimizer PROPERTIES
  BUILD_RPATH "$ORIGIN"
  INSTALL_RPATH "$ORIGIN"
)
endif()

if(SKBUILD)
  install(TARGETS fractal_optimizer RUNTIME DESTINATION fqo/bin COMPONENT fqo)
  install(FILES python/vqe_bridge.py python/vqe_cost_reference.py
    DESTINATION fqo/_vqe COMPONENT fqo)
else()
  include(GNUInstallDirs)
  install(TARGETS fractal_optimizer RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} COMPONENT fqo)
endif()
install(DIRECTORY python/ DESTINATION share/fqo/python COMPONENT fqo
  FILES_MATCHING PATTERN "*.py" PATTERN "fqo" EXCLUDE)
install(DIRECTORY licenses/ DESTINATION share/fqo/licenses COMPONENT fqo)

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/reproducibility/SHA256SUMS")
    add_test(NAME release_assets
      COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_release_assets.py")
  endif()
  add_test(NAME backend_detection COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_configure.py")
  if(FQO_GPU_BACKEND STREQUAL "CPU")
    add_executable(test_cpu_solver tests/test_cpu_solver.cpp)
    target_include_directories(test_cpu_solver PRIVATE include)
    target_link_libraries(test_cpu_solver PRIVATE Eigen3::Eigen)
    add_test(NAME cpu_solver COMMAND test_cpu_solver)
    add_test(NAME python_api COMMAND "${CMAKE_COMMAND}" -E env
      "FQO_EXECUTABLE=$<TARGET_FILE:fractal_optimizer>"
      "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_api.py")
      set_tests_properties(python_api PROPERTIES TIMEOUT 180)
      add_test(NAME native_objectives COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_native_objectives.py"
        --executable "$<TARGET_FILE:fractal_optimizer>")
    if(FQO_ENABLE_MPI)
      set(_fqo_mpi_test_python)
      if(FQO_ENABLE_PYTHON)
        set(_fqo_mpi_test_python --python)
      endif()
      add_test(NAME mpi_objectives COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mpi.py"
        --executable "$<TARGET_FILE:fractal_optimizer>"
        --mpiexec "${MPIEXEC_EXECUTABLE}" "--numproc-flag=${MPIEXEC_NUMPROC_FLAG}"
        ${_fqo_mpi_test_python})
      set_tests_properties(mpi_objectives PROPERTIES TIMEOUT 240)
      add_test(NAME mpi_budget COMMAND "${Python3_EXECUTABLE}"
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_mpi_budget.py"
        --executable "$<TARGET_FILE:fractal_optimizer>"
        --mpiexec "${MPIEXEC_EXECUTABLE}" "--numproc-flag=${MPIEXEC_NUMPROC_FLAG}")
      set_tests_properties(mpi_budget PROPERTIES TIMEOUT 240)
    endif()
  endif()
endif()
