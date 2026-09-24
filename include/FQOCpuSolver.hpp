#pragma once
// CPU equivalents of the existing symmetric eigensolver and Cholesky calls.
// Matrices use column-major storage, matching the GPU solver interfaces.
#include <Eigen/Eigenvalues>
#include <Eigen/Cholesky>
#include <cstdlib>
#include <cstring>
#include <new>
using GpuSolverHandle = void*;
inline void GpuSolverCreate(GpuSolverHandle* h) { *h=nullptr; }
inline void GpuSolverDestroy(GpuSolverHandle) {}
inline void gpu_syevd_buffersize(GpuSolverHandle, int, double*, int, double*, int* n) { *n=0; }
inline void gpu_potrf_buffersize(GpuSolverHandle, int, double*, int, int* n) { *n=0; }
using FQOMatrixMap = Eigen::Map<Eigen::Matrix<double,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>,0,Eigen::OuterStride<>>;
inline void gpu_syevd(GpuSolverHandle, int n, double* a, int lda, double* w, double*, int, int* info) {
    FQOMatrixMap matrix(a,n,n,Eigen::OuterStride<>(lda));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(matrix.selfadjointView<Eigen::Upper>());
    *info=solver.info()==Eigen::Success ? 0 : 1;
    if (*info) return;
    Eigen::Map<Eigen::VectorXd>(w,n)=solver.eigenvalues();
    matrix=solver.eigenvectors();
}
inline void gpu_potrf(GpuSolverHandle, int n, double* a, int lda, double*, int, int* info) {
    FQOMatrixMap matrix(a,n,n,Eigen::OuterStride<>(lda));
    Eigen::LLT<Eigen::MatrixXd,Eigen::Lower> solver(matrix);
    *info=solver.info()==Eigen::Success ? 0 : 1;
    if (!*info) matrix.triangularView<Eigen::Lower>()=solver.matrixL();
}
inline void fqo_cpu_malloc(void** p, size_t n) {
    *p=std::malloc(n ? n : 1);
    if (!*p) throw std::bad_alloc();
}
#define GPU_MALLOC(ptr,bytes) fqo_cpu_malloc((ptr),(bytes))
#define GPU_FREE(ptr) std::free(ptr)
#define GPU_MEMCPY(dst,src,n,k) std::memcpy((dst),(src),(n))
#define GPU_DEV_SYNC() ((void)0)
#define GPU_MEMCPY_D2H 0
#define GPU_MEMCPY_H2D 0
