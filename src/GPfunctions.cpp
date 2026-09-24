#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <chrono>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <KokkosBlas.hpp>
#include <KokkosBlas1_scal.hpp>
#include <KokkosBlas3_trsm.hpp>




// ---------------------------
// Uranie / ROOT for HSIC
// ---------------------------
// We use the developer/manual-supported path:
//  - build a TMatrixD with (N × (D+1)) data
//  - create a URANIE::DataServer::TDSNtupleD from that matrix
//  - attach it to URANIE::DataServer::TDataServer::setTuple()
//  - run URANIE::Sensitivity::THSIC
#ifdef HAVE_URANIE
#include <TMatrixD.h>
#include <TString.h>
#include <TROOT.h>
#include "TDataServer.h"
#include "TDSNtupleD.h"
#include "THSIC.h"
#endif



#if defined(KOKKOS_ENABLE_HIP)
// AMD ROCm path
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <rocsolver/rocsolver.h>

using GpuSolverHandle = rocblas_handle;
inline void GpuSolverCreate (GpuSolverHandle* h){ rocblas_create_handle(h); }
inline void GpuSolverDestroy(GpuSolverHandle  h){ rocblas_destroy_handle(h); }

/* workspace size is managed internally for these routines */
inline void gpu_syevd_buffersize(GpuSolverHandle, int, double*, int, double*, int* lwork){
  *lwork = 0;
}

inline void gpu_syevd(GpuSolverHandle h, int n, double* A, int lda,
                      double* W, double* /*work*/, int /*lwork*/, int* info)
{
  // rocsolver_dsyevd requires an extra device array E of length n-1
  double* E = nullptr;
  if (n > 1) hipMalloc((void**)&E, sizeof(double)*(n-1));
  rocsolver_dsyevd(h,
                   rocblas_evect_original,   // compute eigenvectors
                   rocblas_fill_upper,       // your code uses upper
                   n, A, lda, W, E, info);
  if (E) hipFree(E);
}

inline void gpu_potrf_buffersize(GpuSolverHandle, int, double*, int, int* lwork){
  *lwork = 0;
}

inline void gpu_potrf(GpuSolverHandle h, int n, double* A, int lda,
                      double* /*work*/, int /*lwork*/, int* info)
{
  rocsolver_dpotrf(h, rocblas_fill_lower, n, A, lda, info);
}

// (Optional) silence ‘nodiscard’ warnings from HIP runtime calls:
#define GPU_MALLOC(ptr,bytes)    (void)hipMalloc((ptr),(bytes))
#define GPU_FREE(ptr)            (void)hipFree((ptr))
#define GPU_MEMCPY(dst,src,n,k)  (void)hipMemcpy((dst),(src),(n),(k))
#define GPU_DEV_SYNC()           (void)hipDeviceSynchronize()
#define GPU_MEMCPY_D2H           hipMemcpyDeviceToHost
#define GPU_MEMCPY_H2D           hipMemcpyHostToDevice
#elif defined(KOKKOS_ENABLE_CUDA)
  // --- NVIDIA CUDA path ---
  #include <cuda_runtime.h>
  #include <cusolverDn.h>
  using GpuSolverHandle = cusolverDnHandle_t;
  inline void GpuSolverCreate (GpuSolverHandle* h){ cusolverDnCreate(h); }
  inline void GpuSolverDestroy(GpuSolverHandle  h){ cusolverDnDestroy(h); }
  inline void gpu_syevd_buffersize(GpuSolverHandle h, int n, double* A, int lda, double* W, int* lwork){
    cusolverDnDsyevd_bufferSize(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, A, lda, W, lwork);
  }
  inline void gpu_syevd(GpuSolverHandle h, int n, double* A, int lda, double* W, double* work, int lwork, int* info){
    cusolverDnDsyevd(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, A, lda, W, work, lwork, info);
  }
  inline void gpu_potrf_buffersize(GpuSolverHandle h, int n, double* A, int lda, int* lwork){
    cusolverDnDpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, n, A, lda, lwork);
  }
  inline void gpu_potrf(GpuSolverHandle h, int n, double* A, int lda, double* work, int lwork, int* info){
    cusolverDnDpotrf(h, CUBLAS_FILL_MODE_LOWER, n, A, lda, work, lwork, info);
  }
  #define GPU_MALLOC(ptr,bytes)    cudaMalloc((ptr),(bytes))
  #define GPU_FREE(ptr)            cudaFree((ptr))
  #define GPU_MEMCPY(dst,src,n,k)  cudaMemcpy((dst),(src),(n),(k))
  #define GPU_DEV_SYNC()           cudaDeviceSynchronize()
  #define GPU_MEMCPY_D2H           cudaMemcpyDeviceToHost
  #define GPU_MEMCPY_H2D           cudaMemcpyHostToDevice
#else
  #include "FQOCpuSolver.hpp"
#endif







#include <unordered_map>
#include <iomanip>
#include "GPfunctions.hpp"
#include "HPCfunctions.hpp"
#include <csignal>
#include <stdexcept>
#include <sstream>

#ifndef BO_DEBUG_TRAP_ON_SOLVER_ERROR
#define BO_DEBUG_TRAP_ON_SOLVER_ERROR 1
#endif

namespace {

[[noreturn]] void failFast(const std::string& msg)
{
    std::cerr << msg << std::endl;
#if BO_DEBUG_TRAP_ON_SOLVER_ERROR && defined(SIGTRAP)
    std::raise(SIGTRAP);      // keeps GDB stack alive
#endif
    throw std::runtime_error(msg);
}

inline void checkSolverInfo(int info, const char* what, const char* file, int line)
{
    if (info != 0) {
        std::ostringstream os;
        os << "[Fatal] " << what << " failed with info=" << info
           << " at " << file << ":" << line;
        failFast(os.str());
    }
}

#define CHECK_SOLVER_INFO(info, what) checkSolverInfo((info), (what), __FILE__, __LINE__)

inline void checkTrainingShape(const std::vector<double>& X, int N, int D, const char* where)
{
    if (N <= 0 || D <= 0) {
        std::ostringstream os;
        os << where << ": invalid shape N=" << N << " D=" << D;
        throw std::runtime_error(os.str());
    }
    if (X.size() != size_t(N) * size_t(D)) {
        std::ostringstream os;
        os << where << ": X.size()=" << X.size()
           << " but expected " << size_t(N) * size_t(D);
        throw std::runtime_error(os.str());
    }
}

inline void checkYShape(const std::vector<double>& y, int N, const char* where)
{
    if (y.size() != size_t(N)) {
        std::ostringstream os;
        os << where << ": y.size()=" << y.size() << " but expected " << N;
        throw std::runtime_error(os.str());
    }
}

} // anonymous namespace


namespace {

void evaluateObjectiveBatchAny(const std::vector<double>& X_rowmajor,
                               int N,
                               int D,
                               int expenseFactor,
                               ObjectiveType objType,
                               std::vector<double>& y_out)
{
    y_out.resize(N);
    if (N <= 0) return;

    if (usesExternalObjective(objType)) {
        external_eval_batch_host(X_rowmajor.data(), D, N, y_out.data());
        return;
    }

    GP::evalObjectiveBatchGPU(X_rowmajor, N, D, expenseFactor, objType, y_out);
}

} // namespace

namespace GP {


    // alias the spaces we used in the header
    using exec_space = Kokkos::DefaultExecutionSpace;
    using mem_space  = exec_space::memory_space;
    using layout_t   = Kokkos::LayoutLeft;


KOKKOS_INLINE_FUNCTION
double gpKernelFromDistance2(const double d2,
                             const double ell,
                             const double sigma_f,
                             const FBOGPKernel kernelKind,
                             const double rqAlpha)
{
    const double sf2 = sigma_f * sigma_f;
    const double safeD2 = d2 > 0.0 ? d2 : 0.0;
    const double invEll = 1.0 / ell;
    const double invEll2 = invEll * invEll;

    switch (kernelKind) {
        case FBOGPKernel::Exponential: {
            const double r = sqrt(safeD2) * invEll;
            return sf2 * exp(-r);
        }
        case FBOGPKernel::Matern32: {
            const double z = 1.7320508075688772 * sqrt(safeD2) * invEll;
            return sf2 * (1.0 + z) * exp(-z);
        }
        case FBOGPKernel::Matern52: {
            const double r = sqrt(safeD2) * invEll;
            const double z = 2.23606797749979 * r;
            return sf2 * (1.0 + z + (5.0 / 3.0) * r * r) * exp(-z);
        }
        case FBOGPKernel::RationalQuadratic: {
            const double alpha = rqAlpha > 0.0 ? rqAlpha : 1.0;
            const double base = 1.0 + safeD2 * invEll2 / (2.0 * alpha);
            return sf2 * pow(base, -alpha);
        }
        case FBOGPKernel::SquaredExponential:
        default:
            return sf2 * exp(-0.5 * invEll2 * safeD2);
    }
}

// Build K = k(A,B) with a stationary GP kernel on the device.
// A: (M×r), B: (N×r), K_out: (M×N)
void gpKernelMatrixDevice(
    Kokkos::View<const double**, layout_t, mem_space> A,
    Kokkos::View<const double**, layout_t, mem_space> B,
    double ell,
    double sigma_f,
    FBOGPKernel kernelKind,
    double rqAlpha,
    Kokkos::View<double**, layout_t, mem_space> K_out)
{
    using exec = exec_space;

    const int M = int(A.extent(0));
    const int r = int(A.extent(1));
    const int N = int(B.extent(0));

    if (ell <= 0.0 || !std::isfinite(ell)) {
        throw std::runtime_error("gpKernelMatrixDevice: length scale ell must be positive and finite");
    }
    if (sigma_f <= 0.0 || !std::isfinite(sigma_f)) {
        throw std::runtime_error("gpKernelMatrixDevice: sigma_f must be positive and finite");
    }
    if (int(B.extent(1)) != r) {
        throw std::runtime_error("gpKernelMatrixDevice: latent dimension mismatch");
    }
    if (int(K_out.extent(0)) != M || int(K_out.extent(1)) != N) {
        throw std::runtime_error("gpKernelMatrixDevice: output shape mismatch");
    }

    Kokkos::View<double*, mem_space> nA("nA", M), nB("nB", N);

    Kokkos::parallel_for("norms_A", Kokkos::RangePolicy<exec>(0, M),
        KOKKOS_LAMBDA(int i) {
            double s = 0.0;
            for (int j = 0; j < r; ++j) {
                const double v = A(i, j);
                s += v * v;
            }
            nA(i) = s;
        });

    Kokkos::parallel_for("norms_B", Kokkos::RangePolicy<exec>(0, N),
        KOKKOS_LAMBDA(int i) {
            double s = 0.0;
            for (int j = 0; j < r; ++j) {
                const double v = B(i, j);
                s += v * v;
            }
            nB(i) = s;
        });

    KokkosBlas::gemm("N", "T", 1.0, A, B, 0.0, K_out);

    Kokkos::parallel_for("gp_kernel_from_dots",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {M, N}),
        KOKKOS_LAMBDA(int i, int j) {
            double d2 = nA(i) + nB(j) - 2.0 * K_out(i, j);
            if (d2 < 0.0 && d2 > -1e-10) d2 = 0.0;
            K_out(i, j) =
                gpKernelFromDistance2(d2,
                                      ell,
                                      sigma_f,
                                      kernelKind,
                                      rqAlpha);
        });
}

void rbfKernelMatrixDevice(
    Kokkos::View<const double**, layout_t, mem_space> A,
    Kokkos::View<const double**, layout_t, mem_space> B,
    double ell,
    double sigma_f,
    Kokkos::View<double**, layout_t, mem_space> K_out)
{
    gpKernelMatrixDevice(A,
                         B,
                         ell,
                         sigma_f,
                         FBOGPKernel::SquaredExponential,
                         1.0,
                         K_out);
}

void gpKernelSymmDevice(
    Kokkos::View<const double**, layout_t, mem_space> X,
    double ell,
    double sigma_f,
    double sigma_n,
    FBOGPKernel kernelKind,
    double rqAlpha,
    Kokkos::View<double**, layout_t, mem_space> K_out)
{
    if (sigma_n < 0.0 || !std::isfinite(sigma_n)) {
        throw std::runtime_error("gpKernelSymmDevice: sigma_n must be nonnegative and finite");
    }

    gpKernelMatrixDevice(X, X, ell, sigma_f, kernelKind, rqAlpha, K_out);

    const double noise = sigma_n * sigma_n + 1e-12;
    Kokkos::parallel_for("add_noise_diag",
        Kokkos::RangePolicy<exec_space>(0, int(X.extent(0))),
        KOKKOS_LAMBDA(int i) {
            K_out(i, i) += noise;
        });
}

void rbfKernelSymmDevice(
    Kokkos::View<const double**, layout_t, mem_space> X,
    double ell,
    double sigma_f,
    double sigma_n,
    Kokkos::View<double**, layout_t, mem_space> K_out)
{
    if (sigma_n < 0.0 || !std::isfinite(sigma_n)) {
        throw std::runtime_error("rbfKernelSymmDevice: sigma_n must be nonnegative and finite");
    }

    gpKernelSymmDevice(X,
                       ell,
                       sigma_f,
                       sigma_n,
                       FBOGPKernel::SquaredExponential,
                       1.0,
                       K_out);
}









double mean(const std::vector<double>& v)
{
    if (v.empty()) return 0.0;
    return std::accumulate(v.begin(), v.end(), 0.0) / double(v.size());
}

double stdev(const std::vector<double>& v, double mu)
{
    if (v.empty()) return 1.0;
    double acc = 0.0;
    for (double x : v) {
        const double d = x - mu;
        acc += d * d;
    }
    return std::sqrt(acc / double(v.size()));
}

void standardise(std::vector<double>& v, double& mu, double& sd)
{
    mu = mean(v);
    sd = stdev(v, mu);
    if (!std::isfinite(sd) || sd < 1e-12) sd = 1.0;
    for (double& x : v) x = (x - mu) / sd;
}

double evalObjective(const std::vector<double>& x, ObjectiveType obj, int expenseFactor)
{
    if (usesExternalObjective(obj)) {
        auto& ext = globalExternalObjective();
        return ext.fn ? ext.fn(x.data(), int(x.size())) : 1e300;
    }

    const int D = int(x.size());
    Kokkos::View<const double*,
                 Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>
        xh(x.data(), D);
    return computeBuiltInObjective(xh, expenseFactor, obj);
}

// ---------------------- Stationary GP kernels -----------------------------
double kernel(const std::vector<double>& x,
              const std::vector<double>& xp,
              double l,
              double sigma_f,
              FBOGPKernel kernelKind,
              double rqAlpha) {
    double sqdist = 0.0;
    int d = x.size();
    for (int i = 0; i < d; i++) {
        double diff = x[i] - xp[i];
        sqdist += diff*diff;
    }

    l = l > 0.0 ? l : 1.0;
    const double sf2 = sigma_f * sigma_f;
    const double r = std::sqrt(std::max(0.0, sqdist)) / l;

    switch (kernelKind) {
        case FBOGPKernel::Exponential:
            return sf2 * std::exp(-r);
        case FBOGPKernel::Matern32: {
            const double z = std::sqrt(3.0) * r;
            return sf2 * (1.0 + z) * std::exp(-z);
        }
        case FBOGPKernel::Matern52: {
            const double z = std::sqrt(5.0) * r;
            return sf2 * (1.0 + z + (5.0 / 3.0) * r * r) *
                   std::exp(-z);
        }
        case FBOGPKernel::RationalQuadratic: {
            const double alpha = rqAlpha > 0.0 ? rqAlpha : 1.0;
            return sf2 *
                   std::pow(1.0 + sqdist / (2.0 * alpha * l * l), -alpha);
        }
        case FBOGPKernel::SquaredExponential:
        default:
            return sf2 * std::exp(-0.5 * sqdist / (l*l));
    }
}

double kernel(const std::vector<double>& x,
              const std::vector<double>& xp,
              double l,
              double sigma_f) {
    return kernel(x,
                  xp,
                  l,
                  sigma_f,
                  FBOGPKernel::SquaredExponential,
                  1.0);
}

#ifdef HAVE_URANIE
using namespace URANIE::DataServer;
using namespace URANIE::Sensitivity;
#endif






// --------------------------------------------------------------------------
// HSIC (Uranie/THSIC) helpers (ADDITIVE)
// --------------------------------------------------------------------------
std::vector<double> hsicScoresUranie(
    const std::vector<double>& X_std_rowmajor,
    const std::vector<double>& y_phys,
    int N, int D,
    const std::string& score_type,
    const std::string& kernel_var_method)
{
    PROFILE_SCOPE("HSIC:compute_scores_uranie");

    if ((int)X_std_rowmajor.size() != N * D) {
        throw std::runtime_error("hsicScoresUranie: X_std_rowmajor.size() != N*D");
    }
    if ((int)y_phys.size() != N) {
        throw std::runtime_error("hsicScoresUranie: y_phys.size() != N");
    }
    if (N < 3) {
        // THSIC expects some sample; return zeros
        return std::vector<double>(D, 0.0);
    }

#ifndef HAVE_URANIE
    (void)score_type;
    (void)kernel_var_method;

    // Portable non-Uranie build path used by the benchmark container.
    // The real Uranie THSIC implementation remains active in local builds
    // with ENABLE_URANIE=ON. Here we use squared Pearson correlation as a
    // deterministic dependence score so HSIC-based embedding still compiles
    // and has a sensible ranking fallback.
    double mu_y = 0.0;
    for (double v : y_phys) mu_y += v;
    mu_y /= static_cast<double>(N);

    double var_y = 0.0;
    for (double v : y_phys) {
        const double dy = v - mu_y;
        var_y += dy * dy;
    }

    std::vector<double> scores(D, 0.0);
    if (!(var_y > 0.0) || !std::isfinite(var_y)) return scores;

    for (int j = 0; j < D; ++j) {
        double mu_x = 0.0;
        for (int i = 0; i < N; ++i) mu_x += X_std_rowmajor[i * D + j];
        mu_x /= static_cast<double>(N);

        double var_x = 0.0;
        double cov_xy = 0.0;
        for (int i = 0; i < N; ++i) {
            const double dx = X_std_rowmajor[i * D + j] - mu_x;
            const double dy = y_phys[i] - mu_y;
            var_x += dx * dx;
            cov_xy += dx * dy;
        }

        const double denom = std::sqrt(var_x * var_y);
        if (denom > 0.0 && std::isfinite(denom)) {
            const double rho = cov_xy / denom;
            scores[j] = std::isfinite(rho) ? rho * rho : 0.0;
        }
    }

    return scores;
#else
    // 1) Build varlist: "x0:x1:...:x{D-1}:Y"
    std::string varlist;
    varlist.reserve((size_t)D * 4u + 8u);
    for (int j = 0; j < D; ++j) {
        varlist += "x" + std::to_string(j);
        varlist += (j + 1 < D) ? ":" : ":Y";
    }

    // 2) Fill data matrix (N × (D+1)), columns match varlist
    TMatrixD mat(N, D + 1);
    for (int i = 0; i < N; ++i) {
        // inputs
        const int row0 = i * D;
        for (int j = 0; j < D; ++j) mat(i, j) = X_std_rowmajor[row0 + j];
        // output
        mat(i, D) = y_phys[i];
    }

    // 3) Create TDSNtupleD from matrix
    auto* ntd = new TDSNtupleD(
        "hsic_tuple", "hsic_tuple", varlist.c_str(), mat
    );

    // 4) Create a TDataServer and DECLARE variables (this is what your demo does)
    auto* tds = new URANIE::DataServer::TDataServer("hsic_tds", "hsic_tds");

    // declare inputs x0..x{D-1}
    for (int j = 0; j < D; ++j) {
      tds->addAttribute(Form("x%d", j));   // name-only is enough
    }
    // declare output Y (also required)
    tds->addAttribute("Y");

    // 5) Attach the tuple (now the names match what THSIC will look for)
    tds->setTuple(ntd, /*newismain=*/true, /*att=*/"*");

    // 6) Run THSIC
    std::string inp;
    inp.reserve((size_t)D * 4u);
    for (int j = 0; j < D; ++j) {
      inp += "x" + std::to_string(j);
      if (j + 1 < D) inp += ":";
    }

    URANIE::Sensitivity::THSIC thsic(tds, inp.c_str(), "Y");
    thsic.computeIndexes("quiet");

    // 7) Extract per-dim scores using THSIC::getValue
    // Valid order strings (per Uranie doc): "R2HSic", "HSic", "pValues"
    std::vector<double> scores(D, 0.0);
    for (int j = 0; j < D; ++j) {
        const std::string name = "x" + std::to_string(j);
        scores[j] = thsic.getValue(score_type.c_str(), name.c_str());
        if (!std::isfinite(scores[j])) scores[j] = 0.0;
        if (scores[j] < 0.0) scores[j] = 0.0;
    }

    // 8) Cleanup
    // TDataServer will own the tuple pointer in practice, but to avoid leaks
    // we delete tds; that will delete the underlying tuple depending on Uranie internals.
    // If you see double-free, switch to not deleting ntd explicitly.
     // typically cleans tuple
    // delete ntd; // do NOT delete unless you confirm ownership semantics

    return scores;
#endif
}

std::vector<int> hsicSelectDimsFromScores(
    const std::vector<double>& scores,
    int r_max,
    double hsic_mass_thresh)
{
    PROFILE_SCOPE("HSIC:select_dims");
    const int D = (int)scores.size();
    if (D == 0) return {};
    r_max = std::max(1, std::min(r_max, D));

    // sort dims by descending score
    std::vector<int> order(D);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b){
        return scores[a] > scores[b];
    });

    // if no threshold, take exactly r_max
    if (!(hsic_mass_thresh > 0.0 && hsic_mass_thresh <= 1.0)) {
        order.resize(r_max);
        return order;
    }

    const double total = std::accumulate(scores.begin(), scores.end(), 0.0);
    if (!(total > 0.0)) {
        // degenerate: pick first r_max dims
        order.resize(r_max);
        return order;
    }

    double cum = 0.0;
    int r = 0;
    for (int k = 0; k < D && r < r_max; ++k) {
        cum += scores[order[k]];
        ++r;
        if (cum / total >= hsic_mass_thresh) break;
    }
    order.resize(std::max(1, r));
    return order;
}

std::vector<double> hsicEmbedUranie95(
    const std::vector<double>& X_std_rowmajor,
    const std::vector<double>& y_phys,
    int N, int D,
    int r_max,
    double hsic_mass_thresh,
    std::vector<int>* out_dims,
    std::vector<double>* out_scores,
    const std::string& score_type,
    const std::string& kernel_var_method)
{
    PROFILE_SCOPE("HSIC:embed95");
    auto scores = hsicScoresUranie(X_std_rowmajor, y_phys, N, D, score_type, kernel_var_method);
    auto dims   = hsicSelectDimsFromScores(scores, r_max, hsic_mass_thresh);
    const int r = (int)dims.size();

    std::vector<double> Z((size_t)N * (size_t)r);
    for (int i = 0; i < N; ++i) {
        const int row0 = i * D;
        const int out0 = i * r;
        for (int k = 0; k < r; ++k) {
            Z[out0 + k] = X_std_rowmajor[row0 + dims[k]];
        }
    }

    if (out_dims)   *out_dims = dims;
    if (out_scores) *out_scores = std::move(scores);
    return Z;
}




// ---------------------- PCA Embedder (GPU/cuSolver) -----------------
std::vector<double> pcaEmbed(const std::vector<double>& X_flat, int N, int D, int R, std::vector<double>* out_basis)
{
    PROFILE_SCOPE("trainGP:PCA");
    using Exec   = Kokkos::DefaultExecutionSpace;
    using View2D = Kokkos::View<double**, Kokkos::LayoutLeft, Exec>;
    using View1D = Kokkos::View<double*,  Kokkos::LayoutLeft, Exec>;

    // 1. host → device
    View2D X_d("X", N, D);
    {
        PROFILE_SCOPE("PCA:H2D_copy");
        Kokkos::deep_copy(
            X_d,
            Kokkos::View<const double**, Kokkos::LayoutLeft,
                        Kokkos::HostSpace,
                        Kokkos::MemoryTraits<Kokkos::Unmanaged>>(X_flat.data(), N, D));
    }

    // 2. mean + centring
    {
        PROFILE_SCOPE("PCA:mean+centre");
        View1D mu_d("mu", D);
        Kokkos::parallel_for("mean+centre",
            Kokkos::TeamPolicy<Exec>(D, Kokkos::AUTO),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team)
        {
            int d = team.league_rank();
            double sum = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, N),
                [&](int i, double& loc){ loc += X_d(i,d); }, sum);
            sum /= double(N);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, N),
                [&](int i){ X_d(i,d) -= sum; });
            if (team.team_rank() == 0) mu_d(d) = sum;
        });
    }

    // 3. covariance C = Xᵀ X / (N-1) (GEMM, device)
    View2D C_d("C", D, D);
    {
        PROFILE_SCOPE("PCA:covariance");
        const double alpha = 1.0 / double(N - 1), beta = 0.0;
        KokkosBlas::gemm("T", "N", alpha, X_d, X_d, beta, C_d);
    }

    // 4. eigendecomp on device (cuSolver)
    Kokkos::fence(); // ensure C_d is ready on device

    std::vector<double> eigvals(D);
    {
        PROFILE_SCOPE("PCA:eigen_decomp");

        GpuSolverHandle solver = nullptr;
        GpuSolverCreate(&solver);

        double* d_C = C_d.data();
        int lda = D;

        double* d_W = nullptr;
        GPU_MALLOC((void**)&d_W, D * sizeof(double));

        int lwork = 0;
        gpu_syevd_buffersize(
            solver, D, d_C, lda, d_W, &lwork);

        double* d_work = nullptr;
        GPU_MALLOC((void**)&d_work, lwork * sizeof(double));
        int* devInfo = nullptr;
        GPU_MALLOC((void**)&devInfo, sizeof(int));

        gpu_syevd(
            solver, D, d_C, lda, d_W, d_work, lwork, devInfo);

        GPU_DEV_SYNC();

        int h_devInfo = 0;
        GPU_MEMCPY(&h_devInfo, devInfo, sizeof(int), GPU_MEMCPY_D2H);
        if (h_devInfo != 0) {
            std::cerr << "cuSolver eigendecomp failed: " << h_devInfo << std::endl;
            std::exit(1);
        }



        GPU_MEMCPY(eigvals.data(), d_W, D * sizeof(double), GPU_MEMCPY_D2H);

        GPU_FREE(d_W);
        GPU_FREE(d_work);
        GPU_FREE(devInfo);
        GpuSolverDestroy(solver);
    }

    // 5. Extract last R columns of eigenvectors (principal components)
    View2D B_d("B", D, R);
    {
        PROFILE_SCOPE("PCA:B_H2D_copy");
        auto C_sub = Kokkos::subview(C_d, Kokkos::ALL(), std::make_pair(D - R, D));
        Kokkos::deep_copy(B_d, C_sub);
        Kokkos::fence();
    }

    // 5.5. Copy basis back to host
    if (out_basis) {
        std::vector<double> basis(D * R);
        Kokkos::deep_copy(
            Kokkos::View<double**, Kokkos::LayoutLeft,
                         Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>(basis.data(), D, R),
            B_d
        );
        *out_basis = std::move(basis); // col-major!
    }

    // 6. projection Z = X B (GEMM, device)
    View2D Z_d("Z", N, R);
    { PROFILE_SCOPE("PCA:projection");
      KokkosBlas::gemm("N", "N", 1.0, X_d, B_d, 0.0, Z_d);
    }

    // 7. flatten and return
    std::vector<double> out(N * R);
    {
        PROFILE_SCOPE("PCA:copy_out_to_host");
        Kokkos::deep_copy(
            Kokkos::View<double**, Kokkos::LayoutLeft,
                         Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>(out.data(), N, R),
            Z_d);
    }

    return out;
}




// ---------------------- PCA Embedder (GPU/cuSolver) with 95% rule ---------------
std::vector<double> pcaEmbed95(const std::vector<double>& X_flat, int N, int D,
                               int R_max,               // hard cap for latent dim
                               double var_thresh,       // e.g., 0.95
                               int* out_R,              // returns chosen r
                               std::vector<double>* out_basis)
{
    PROFILE_SCOPE("trainGP:PCA");
    using Exec   = Kokkos::DefaultExecutionSpace;
    using View2D = Kokkos::View<double**, Kokkos::LayoutLeft, Exec>;
    using View1D = Kokkos::View<double*,  Kokkos::LayoutLeft, Exec>;

    // 1) host → device
    View2D X_d("X", N, D);
    {
        PROFILE_SCOPE("PCA:H2D_copy");
        Kokkos::deep_copy(
            X_d,
            Kokkos::View<const double**, Kokkos::LayoutLeft,
                         Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>(X_flat.data(), N, D));
    }

    // 2) mean + centring
    {
        PROFILE_SCOPE("PCA:mean+centre");
        View1D mu_d("mu", D);
        Kokkos::parallel_for("mean+centre",
            Kokkos::TeamPolicy<Exec>(D, Kokkos::AUTO),
            KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team)
        {
            int d = team.league_rank();
            double sum = 0.0;
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, N),
                [&](int i, double& loc){ loc += X_d(i,d); }, sum);
            sum /= double(N);
            Kokkos::parallel_for(
                Kokkos::TeamThreadRange(team, N),
                [&](int i){ X_d(i,d) -= sum; });
            if (team.team_rank() == 0) mu_d(d) = sum;
        });
    }



  // 3–5) PCA path selection:
  //  - If D >> N: snapshot PCA on G = X Xᵀ/(N-1) (N×N), then V = Xᵀ U diag(1/sqrt((N-1)λ))
  //  - Else: original D×D covariance path
  const bool use_snapshots = (N <= D);
  int r = 0;
  std::vector<double> eigvals(use_snapshots ? N : D);
  View2D B_d; // final (D×r) basis on device

  if (use_snapshots) {
    // 3a) Gram matrix G = X Xᵀ/(N-1)  (N×N)
    View2D G_d("G", N, N);
    {
      PROFILE_SCOPE("PCA:gram");
      const double alpha = 1.0 / double(N - 1), beta = 0.0;
      KokkosBlas::gemm("N","T", alpha, X_d, X_d, beta, G_d);
    }
    // 4a) eigendecomp of G
    Kokkos::fence();
    {
      PROFILE_SCOPE("PCA:eigen_decomp(NxN)");


      GpuSolverHandle H=nullptr; GpuSolverCreate(&H);
      double* d_G = G_d.data(); const int lda = N;
      double* d_W=nullptr; GPU_MALLOC((void**)&d_W, sizeof(double)*N);
      int lwork=0; gpu_syevd_buffersize(H, N, d_G, lda, d_W, &lwork);
      double* d_work=nullptr; GPU_MALLOC((void**)&d_work, sizeof(double)*lwork);
      int* devInfo=nullptr;  GPU_MALLOC((void**)&devInfo, sizeof(int));
      gpu_syevd(H, N, d_G, lda, d_W, d_work, lwork, devInfo);
      GPU_DEV_SYNC();
      int hInfo=0; GPU_MEMCPY(&hInfo, devInfo, sizeof(int), GPU_MEMCPY_D2H);



      if (hInfo != 0) { std::cerr<<"cuSolver eigendecomp(G) failed: "<<hInfo<<std::endl; std::exit(1); }
      GPU_MEMCPY(eigvals.data(), d_W, sizeof(double)*N, GPU_MEMCPY_D2H);
      GPU_FREE(d_W); GPU_FREE(d_work); GPU_FREE(devInfo); GpuSolverDestroy(H);

    }
    // 4.5a) choose r to hit var_thresh
    double total = std::accumulate(eigvals.begin(), eigvals.end(), 0.0);
    r = 0; if (total > 0.0) { double cum=0.0; for (int i=N-1; i>=0; --i) { cum += eigvals[i]; ++r; if (cum/total >= var_thresh) break; } }
    r = std::max(1, std::min(r, std::min(R_max, N)));
    if (out_R) *out_R = r;
    // 5a) U_r = last r cols of G_d (N×r); V = Xᵀ U_r diag(1/sqrt((N-1)λ))
    auto U_r = Kokkos::subview(G_d, Kokkos::ALL(), std::make_pair(N - r, N));
    View2D Temp("Temp", D, r); // Temp = Xᵀ * U_r
    {
      PROFILE_SCOPE("PCA:XT*Ur");
      KokkosBlas::gemm("T","N", 1.0, X_d, U_r, 0.0, Temp);
    }
    // scale columns
    {
      PROFILE_SCOPE("PCA:scale_basis");
      for (int j=0; j<r; ++j) {
        const double lam = std::max(eigvals[N - r + j], 1e-16);
        const double s = 1.0 / std::sqrt((N - 1) * lam);
        auto col = Kokkos::subview(Temp, Kokkos::ALL(), j);
        KokkosBlas::scal(col, s, col);
      }
    }
    B_d = Temp; // (D×r)
  } else {
    // Original D×D covariance path (kept as fallback)
    View2D C_d("C", D, D);
    {
      PROFILE_SCOPE("PCA:covariance");
      const double alpha = 1.0 / double(N - 1), beta = 0.0;
      KokkosBlas::gemm("T","N", alpha, X_d, X_d, beta, C_d);
    }
    Kokkos::fence();
    {
      PROFILE_SCOPE("PCA:eigen_decomp(DxD)");
      GpuSolverHandle H=nullptr; GpuSolverCreate(&H);
      double* d_C = C_d.data(); const int lda = D;
      double* d_W=nullptr; GPU_MALLOC((void**)&d_W, sizeof(double)*D);
      int lwork=0; gpu_syevd_buffersize(H, D, d_C, lda, d_W, &lwork);
      double* d_work=nullptr; GPU_MALLOC((void**)&d_work, sizeof(double)*lwork);
      int* devInfo=nullptr;  GPU_MALLOC((void**)&devInfo, sizeof(int));
      gpu_syevd(H, D, d_C, lda, d_W, d_work, lwork, devInfo);
      GPU_DEV_SYNC();
      int hInfo=0; GPU_MEMCPY(&hInfo, devInfo, sizeof(int), GPU_MEMCPY_D2H);

      if (hInfo != 0) { std::cerr<<"cuSolver eigendecomp(C) failed: "<<hInfo<<std::endl; std::exit(1); }
      GPU_MEMCPY(eigvals.data(), d_W, sizeof(double)*D, GPU_MEMCPY_D2H);
      GPU_FREE(d_W); GPU_FREE(d_work); GPU_FREE(devInfo); GpuSolverDestroy(H);
    }
    double total = std::accumulate(eigvals.begin(), eigvals.end(), 0.0);
    r = 0; if (total>0.0){ double cum=0.0; for (int i=D-1; i>=0; --i){ cum+=eigvals[i]; ++r; if (cum/total >= var_thresh) break; } }
    r = std::max(1, std::min(r, std::min(R_max, D)));
    if (out_R) *out_R = r;
    View2D Btmp("B", D, r);
    {
      PROFILE_SCOPE("PCA:B_H2D_copy");
      auto C_sub = Kokkos::subview(C_d, Kokkos::ALL(), std::make_pair(D - r, D));
      Kokkos::deep_copy(Btmp, C_sub);
      Kokkos::fence();
    }
    B_d = Btmp;
  }

    if (out_basis) {
        std::vector<double> basis(D * r);
        Kokkos::deep_copy(
            Kokkos::View<double**, Kokkos::LayoutLeft,
                         Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>(basis.data(), D, r),
            B_d
        );
        *out_basis = std::move(basis);   // col-major
    }

    // 6) Z = X B
    View2D Z_d("Z", N, r);
    {
        PROFILE_SCOPE("PCA:projection");
        KokkosBlas::gemm("N", "N", 1.0, X_d, B_d, 0.0, Z_d);
    }

    // 7) flatten and return
    std::vector<double> out(N * r);
    {
        PROFILE_SCOPE("PCA:copy_out_to_host");
        Kokkos::deep_copy(
            Kokkos::View<double**, Kokkos::LayoutLeft,
                         Kokkos::HostSpace,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>(out.data(), N, r),
            Z_d);
    }

    // Optional: a little log is handy
  if (var_thresh > 0.0) {
    const int M = use_snapshots ? N : D;
    double total = std::accumulate(eigvals.begin(), eigvals.end(), 0.0);
    double keep  = 0.0; for (int i=0;i<r;++i) keep += eigvals[M - 1 - i];
    double ev = (total > 0.0 ? keep / total : 0.0);
        std::cout << "[PCA] chosen r = " << r
                  << " (target " << int(var_thresh * 100) << "%), explained ≈ "
                  << std::fixed << std::setprecision(2) << ev * 100 << "%\n";
    }

    return out;
}


  // --------------------------------------------------------------------------
  // Latin Hypercube Sampling Implementation (host code).
  // Returns a flattened array X_host of length N*d
  // --------------------------------------------------------------------------
  // GPU-parallel across dimensions; per-dim Fisher–Yates happens inside one team.
std::vector<double> latinHypercubeSample(
  const int N,
  const int d,
  const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& lowerBounds,
  const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& upperBounds,
  const unsigned seed)
{
  using exec_space = Kokkos::DefaultExecutionSpace;
  using mem_space  = exec_space::memory_space;

  // Flat device array in *row-major* order to match X_host[i*d + dim].
  Kokkos::View<double*, mem_space> d_X("X_flat", size_t(N) * size_t(d));

  // A single big perm buffer: [d × N], indexed as perm(dim*N + i).
  Kokkos::View<int*, mem_space> d_perm("perm", size_t(d) * size_t(N));

  // Random pool for reproducibility; mix in seed so each dim has distinct stream.
  Kokkos::Random_XorShift64_Pool<mem_space> pool(seed ? seed : 12345u);

  using team_policy  = Kokkos::TeamPolicy<exec_space>;
  using member_type  = team_policy::member_type;

  Kokkos::parallel_for("LHS_per_dimension",
    team_policy(d, Kokkos::AUTO),
    KOKKOS_LAMBDA(const member_type& team)
  {
    const int dim = team.league_rank();
    const size_t row0 = size_t(dim) * size_t(N);

    // Local RNG state for this team/dimension
    auto rng = pool.get_state();

    // 1) init perm to 0..N-1 (parallel fill ok)
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N),
      [&](int i){ d_perm(row0 + i) = i; });
    team.team_barrier();

    // 2) Fisher–Yates shuffle (sequential within team)
    if (team.team_rank() == 0) {
      for (int i = N - 1; i > 0; --i) {
        const int j = int(rng.urand() % (unsigned)(i + 1));
        const int a = d_perm(row0 + i);
        const int b = d_perm(row0 + j);
        d_perm(row0 + i) = b;
        d_perm(row0 + j) = a;
      }
    }
    team.team_barrier();

    // 3) Write coordinates for this dimension
    const double LB = lowerBounds(dim);
    const double UB = upperBounds(dim);
    const double span = (UB - LB);

    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](int i){
      // u in (0,1)
      const double u = Kokkos::rand<decltype(rng), double>::draw(rng);
      const int    b = d_perm(row0 + i);
      const double coord = LB + ((double(b) + u) / double(N)) * span;

      // row-major: [i * d + dim]
      d_X(size_t(i) * size_t(d) + size_t(dim)) = coord;
    });

    // Return RNG state
    pool.free_state(rng);
  });

  // D2H: flatten into std::vector in the same row-major order
  std::vector<double> X_host((size_t)N * (size_t)d);
  Kokkos::deep_copy(
    Kokkos::View<double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(X_host.data(), X_host.size()),
    d_X);

  return X_host;
}


Model trainGP(const std::vector<double>& X_orig_host,  // Now: always original data!
              const int N,
              const int d,
              int r,
              const double sigma_f,
              const double l,
              const double sigma_n,
              int expenseFactor,
              ObjectiveType objType,
              FBOGPKernel kernelKind,
              double rqAlpha)
{
    PROFILE_SCOPE("trainGP_total");

    // (1) Standardize X (make a copy!)
    PROFILE_SCOPE("trainGP:standardise_X");
    std::vector<double> mu_X(d), sd_X(d);
    std::vector<double> X_phys = X_orig_host; // Copy to manipulate

    for (int dim = 0; dim < d; ++dim) {
        std::vector<double> col(N);
        for (int i = 0; i < N; ++i) col[i] = X_phys[i * d + dim];
        standardise(col, mu_X[dim], sd_X[dim]);
        for (int i = 0; i < N; ++i) X_phys[i * d + dim] = col[i];
    }

    // (2) PCA — pick r adaptively with 95% EV (capped by input r)
    std::vector<double> X_norm_flat, pca_basis;
    int r_chosen = r; // cap
    X_norm_flat = pcaEmbed95(X_phys, N, d,
                             /*R_max=*/r,
                             /*var_thresh=*/0.95,
                             /*out_R=*/&r_chosen,
                             &pca_basis);

    // (3) Copy to model
    Model model;
    model.N = N;
    model.d = r_chosen;  // latent dim used by the GP
    model.D_orig = d;

    model.X_host = X_norm_flat;        // size N * r_chosen
    model.X_orig_host = X_orig_host;   // size N * d (original)
    model.X.assign(N, std::vector<double>(r_chosen));
    for (int i = 0; i < N; ++i)
        for (int dim = 0; dim < r_chosen; ++dim)
            model.X[i][dim] = X_norm_flat[i * r_chosen + dim];
    model.pca_basis = std::move(pca_basis);
    model.uses_hsic = false;

    // (4) Objective evaluation (on original, unnormalized points)
    PROFILE_SCOPE("trainGP:eval_objective_gpu");
    evalObjectiveBatchGPU(X_orig_host,
                          N,
                          d,
                          expenseFactor,
                          objType,
                          model.y_phys);
    // (5) Standardize y
    std::vector<double> y_std = model.y_phys;
    double mu_y, sd_y;
    standardise(y_std, mu_y, sd_y);

    model.y   = y_std;
    model.mu_X = mu_X;
    model.sd_X = sd_X;
    model.mu_y = mu_y;
    model.sd_y = sd_y;

    // ========== (6–8) GPU build K (RBF), Cholesky with cuSolver, and solve for alpha ==========
    {
        PROFILE_SCOPE("trainGP:gpu_buildK_chol_solve");

        using Exec     = Kokkos::DefaultExecutionSpace;
        using MemSpace = Exec::memory_space;
        using View2D   = Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace>;

        const int R = model.d;

        // X_d : (N × R) latent coords on device (from model.X_host flattened)
        View2D X_d("X_d", N, R);
        {
            auto Xh = Kokkos::View<const double**, Kokkos::LayoutLeft,
                                   Kokkos::HostSpace,
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
                          model.X_host.data(), N, R);
            Kokkos::deep_copy(X_d, Xh);
        }

        // Build symmetric RBF kernel directly on device (adds σ_n^2 I)
        View2D K_d("K_d", N, N);
        gpKernelSymmDevice(X_d,
                           l,
                           sigma_f,
                           sigma_n,
                           kernelKind,
                           rqAlpha,
                           K_d);
        Kokkos::fence();

        // --- Cholesky on device (cuSolver) ---
        GpuSolverHandle H = nullptr;
        GpuSolverCreate(&H);

        int lwork = 0;
        int* dInfo = nullptr;
        GPU_MALLOC((void**)&dInfo, sizeof(int));

        // buffer size & factorization on K_d
        gpu_potrf_buffersize(H, N, K_d.data(), N, &lwork);
        double* dWork = nullptr;
        GPU_MALLOC((void**)&dWork, sizeof(double) * lwork);

        gpu_potrf(H, N, K_d.data(), N, dWork, lwork, dInfo);
        GPU_DEV_SYNC();

        // check success
        int hInfo = 0;
        GPU_MEMCPY(&hInfo, dInfo, sizeof(int), GPU_MEMCPY_D2H);

        if (hInfo != 0) {
            std::cerr << "cuSolver potrf failed, info=" << hInfo << std::endl;
            std::exit(EXIT_FAILURE);
        }

        // K_d now contains L in its lower triangle

        // --- Solve for alpha: L L^T alpha = y_std ---
        // put y_std to device as (N × 1) so we can use TRSM twice
        View2D B("B", N, 1);
        {
            auto Bh = Kokkos::create_mirror_view(B);
            for (int i = 0; i < N; ++i) Bh(i,0) = model.y[i];   // y is standardized already
            Kokkos::deep_copy(B, Bh);
        }

        // forward solve: L * Z = B  → overwrite B with Z
        KokkosBlas::trsm("L","L","N","N", 1.0, K_d, B);
        // backward solve: L^T * alpha = Z  → overwrite B with alpha
        KokkosBlas::trsm("L","L","T","N", 1.0, K_d, B);

        // copy alpha back to host
        model.alpha.assign(N, 0.0);
        {
            auto Bh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);
            for (int i = 0; i < N; ++i) model.alpha[i] = Bh(i,0);
        }

        // also copy L back to host so your CPU predict path still works
        model.L.assign(N, std::vector<double>(N, 0.0));
        {
            auto Kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), K_d);
            for (int i = 0; i < N; ++i)
                for (int j = 0; j <= i; ++j)
                    model.L[i][j] = Kh(i,j);   // lower-triangular
        }

        // cleanup
        GPU_FREE(dWork);
        GPU_FREE(dInfo);
        GpuSolverDestroy(H);
    }
    // ========== end GPU training block ==========

    model.logMarginal = 0.0; // Optional: compute if needed
    model.sigma_f = sigma_f;
    model.l = l;
    model.sigma_n = sigma_n;
    model.kernel = kernelKind;
    model.rq_alpha = rqAlpha;

    return model;
}



Model updateGP(const Model& oldModel,
               const std::vector<double>& newPoints,
               const int new_N,
               const int r,
               const double sigma_f,
               const double l,
               const double sigma_n,
               int expenseFactor,
               ObjectiveType objType,
               FBOGPKernel kernelKind,
               double rqAlpha)
{
    int oldN = oldModel.N;

    // derive original D from stored original points
    int D_orig = (int)(oldModel.X_orig_host.size() / oldModel.N);

    int totalN = oldN + new_N;

    std::vector<double> combined;
    combined.reserve(totalN * D_orig);
    combined.insert(combined.end(), oldModel.X_orig_host.begin(), oldModel.X_orig_host.end());
    combined.insert(combined.end(), newPoints.begin(), newPoints.end());

    // IMPORTANT: pass D_orig, not oldModel.d
    return trainGP(combined,
                   totalN,
                   D_orig,
                   r,
                   sigma_f,
                   l,
                   sigma_n,
                   expenseFactor,
                   objType,
                   kernelKind,
                   rqAlpha);
}



  // --------------------------------------------------------------------------
  // CPU-based predictGP
  // --------------------------------------------------------------------------
  void predictGP(const Model& model,
                 const std::vector<double>& x_star,
                 double sigma_f,
                 double l,
                 double sigma_n,
                 double& mu,
                 double& sigma) {
    int N = model.N;
    int d = model.d;

    // 1) build k_star
    std::vector<double> k_star(N, 0.0);
    for(int i=0; i<N; i++){
      k_star[i] = kernel(x_star,
                         model.X[i],
                         l,
                         sigma_f,
                         model.kernel,
                         model.rq_alpha);
    }

    // 2) mu = k_star^T alpha
    mu = 0.0;
    for(int i=0; i<N; i++){
      mu += k_star[i]*model.alpha[i];
    }

    // 3) solve L v = k_star
    std::vector<double> v(N,0.0);
    for(int i=0; i<N; i++){
      double sum = k_star[i];
      for(int j=0; j<i; j++){
        sum -= model.L[i][j]*v[j];
      }
      v[i] = sum/model.L[i][i];
    }

    // 4) k_xx = sigma_f^2
    double k_xx = sigma_f*sigma_f + sigma_n*sigma_n;

    // 5) sigma^2 = k_xx - v^T v
    double var = 0.0;
    for(int i=0; i<N; i++){
      var += v[i]*v[i];
    }
    var = std::max(k_xx - var, 1e-10);
    sigma = std::sqrt(var);
  }


  double predictGP(const Model& model,
                 const std::vector<double>& x_star,
                 const double /*kappa*/,
                 double& sigma_out)
{
    double mu;
    predictGP(model,
              x_star,
              model.sigma_f,
              model.l,
              model.sigma_n,
              mu,
              sigma_out);
    return mu;
}





// ----------------------------------------------------------------------------
// 1) Copy the trained CPU model to device memory once
// ----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// makeDeviceCopy
// -----------------------------------------------------------------------------
DeviceGPModel makeDeviceCopy(const Model& h,
                             double sigma_f,
                             double l,
                             double sigma_n,
                             FBOGPKernel kernelKind,
                             double rqAlpha)
{
  DeviceGPModel d;
  d.sigma_f = sigma_f;
  d.l       = l;
  d.sigma_n = sigma_n;
  d.kernel = kernelKind;
  d.rq_alpha = rqAlpha;
  d.uses_hsic = h.uses_hsic;
  d.Ntrain   = h.N;
d.d_latent = h.d;
d.D_orig   = h.D_orig;

  // ---------------- alpha (rank-1) ----------------
  d.alpha = Kokkos::View<double*, layout_t, mem_space>("alpha", h.N);
  {
    auto h_alpha = Kokkos::create_mirror_view(d.alpha);
    for (int i = 0; i < h.N; ++i) h_alpha(i) = h.alpha[i];
    Kokkos::deep_copy(d.alpha, h_alpha);
  }

  // ---------------- X (rank-2) --------------------
  d.X = Kokkos::View<double**, layout_t, mem_space>("X", h.N, h.d);
  {
    auto h_X = Kokkos::create_mirror_view(d.X);
    for (int i = 0; i < h.N; ++i)
      for (int j = 0; j < h.d; ++j)
        h_X(i, j) = h.X[i][j];
    Kokkos::deep_copy(d.X, h_X);
  }

  // -----------------------------------------------------------------
  // lower-triangular Cholesky factor  (host → device)                NEW
  // -----------------------------------------------------------------
  d.L = Kokkos::View<double**, layout_t, mem_space>("L", h.N, h.N);
  {
      auto hL = Kokkos::create_mirror_view(d.L);
      for (int i = 0; i < h.N; ++i)
          for (int j = 0; j <= i; ++j)          // only lower part used
              hL(i,j) = h.L[i][j];
      Kokkos::deep_copy(d.L, hL);
  }



  // mu_X / sd_X
d.mu_X = Kokkos::View<double*, layout_t, mem_space>("mu_X", h.D_orig);
d.sd_X = Kokkos::View<double*, layout_t, mem_space>("sd_X", h.D_orig);
{
  auto hm = Kokkos::create_mirror_view(d.mu_X);
  auto hs = Kokkos::create_mirror_view(d.sd_X);
  for (int i=0;i<h.D_orig;++i){ hm(i)=h.mu_X[i]; hs(i)=h.sd_X[i]; }
  Kokkos::deep_copy(d.mu_X, hm);
  Kokkos::deep_copy(d.sd_X, hs);
}

// PCA basis B (D_orig × d_latent), column-major in pca_basis -> fill by [row + D_orig*col]
// ===== Embedding matrix on device =====
// For PCA: use h.pca_basis (D_orig × d) col-major
// For HSIC: build a sparse "selector" matrix B where B(dim_k, k)=1, else 0.
// This keeps projectOnDevice working if you later use it, and avoids segfault.

d.B = Kokkos::View<double**, layout_t, mem_space>("B", h.D_orig, h.d);
{
  auto hB = Kokkos::create_mirror_view(d.B);

  // zero-init
  for (int r = 0; r < h.D_orig; ++r)
    for (int c = 0; c < h.d; ++c)
      hB(r,c) = 0.0;

  const bool have_pca_basis = ((int)h.pca_basis.size() == h.D_orig * h.d);

  if (have_pca_basis) {
    // PCA: copy col-major basis
    for (int c = 0; c < h.d; ++c)
      for (int r = 0; r < h.D_orig; ++r)
        hB(r,c) = h.pca_basis[r + h.D_orig * c];
  } else if ((int)h.hsic_dims.size() == h.d) {
    // HSIC: selector matrix
    for (int c = 0; c < h.d; ++c) {
      const int dim = h.hsic_dims[c];
      if (0 <= dim && dim < h.D_orig) hB(dim, c) = 1.0;
    }
  } else {
    // last-resort fallback: first d dims
    for (int c = 0; c < h.d; ++c) {
      const int dim = c;
      if (dim < h.D_orig) hB(dim, c) = 1.0;
    }
  }

  Kokkos::deep_copy(d.B, hB);
}


  // HSIC selected dims (optional; may be empty if you haven't used HSIC yet)
  // We still allocate it with size d_latent if provided; otherwise fill identity-ish.
  d.hsic_dims = Kokkos::View<int*, layout_t, mem_space>("hsic_dims", h.d);
  {
    auto hd = Kokkos::create_mirror_view(d.hsic_dims);
    if ((int)h.hsic_dims.size() == h.d) {
      for (int k = 0; k < h.d; ++k) hd(k) = h.hsic_dims[k];
    } else {
      // default: 0..d-1 (safe fallback)
      for (int k = 0; k < h.d; ++k) hd(k) = k;
    }
    Kokkos::deep_copy(d.hsic_dims, hd);
  }

  return d;
}
// ----------------------------------------------------------------------------
// 2) Predict mean/variance for a batch of M points that already live on GPU
//    (very plain, O(M·N·d); good enough to get you running)
// ----------------------------------------------------------------------------
void batchPredictGP(const DeviceGPModel& d_gp,
                    Kokkos::View<const double**, Kokkos::LayoutLeft, mem_space> d_Xstar, // (M × r)
                    Kokkos::View<double*,        mem_space> d_mu,    // (M)
                    Kokkos::View<double*,        mem_space> d_var)   // (M)
{
  using exec = exec_space;
  const int M = d_Xstar.extent(0);
  const int N = d_gp.Ntrain;

  const double sf2   = d_gp.sigma_f*d_gp.sigma_f;
  const double k_xx  = sf2 + d_gp.sigma_n*d_gp.sigma_n;

  Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> Kstar("Kstar", M, N);
  gpKernelMatrixDevice(d_Xstar,
                       d_gp.X,
                       d_gp.l,
                       d_gp.sigma_f,
                       d_gp.kernel,
                       d_gp.rq_alpha,
                       Kstar);

  // mu = K_star * alpha
  KokkosBlas::gemv("N", 1.0, Kstar, d_gp.alpha, 0.0, d_mu);

  // Row-wise solve on the RIGHT:  X = Kstar * L^{-T}
  // (Side='R', Uplo='L', TransA='T', Diag='N')
  KokkosBlas::trsm("R","L","T","N", 1.0, d_gp.L, Kstar);

  // var_j = k_xx - ||row_j(X)||^2
  Kokkos::parallel_for("var_rows", Kokkos::RangePolicy<exec>(0, M),
    KOKKOS_LAMBDA(int j){
      double ss = 0.0;
      for (int i = 0; i < N; ++i) { double v = Kstar(j,i); ss += v*v; }
      double vv = k_xx - ss;
      d_var(j) = vv > 1e-10 ? vv : 1e-10;
    });}





double negLogMarginalLikelihood(const GP::Model& m,
                                double sigma_f,
                                double l,
                                double sigma_n,
                                FBOGPKernel kernelKind,
                                double rqAlpha)
{
  const int N = m.N;
  // (1) Build K
  std::vector<std::vector<double>> K(N, std::vector<double>(N, 0.0));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j) {
      double kij = GP::kernel(m.X[i],
                              m.X[j],
                              l,
                              sigma_f,
                              kernelKind,
                              rqAlpha);
      if (i == j) kij += sigma_n * sigma_n;
      K[i][j] = K[j][i] = kij;
    }
  }
  // (2) Cholesky (lower)
  std::vector<std::vector<double>> L(N, std::vector<double>(N, 0.0));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j <= i; ++j) {
      double s = 0.0;
      for (int k = 0; k < j; ++k) s += L[i][k] * L[j][k];
      if (i == j) L[i][j] = std::sqrt(std::max(1e-15, K[i][i] - s));
      else        L[i][j] = (K[i][j] - s) / L[j][j];
    }
  }
  // (3) Solve K α = y  (standardized y)
  const std::vector<double>& y = m.y;
  std::vector<double> z(N), alpha(N);
  for (int i = 0; i < N; ++i) {
    double s = y[i];
    for (int k = 0; k < i; ++k) s -= L[i][k] * z[k];
    z[i] = s / L[i][i];
  }
  for (int i = N - 1; i >= 0; --i) {
    double s = z[i];
    for (int k = i + 1; k < N; ++k) s -= L[k][i] * alpha[k];
    alpha[i] = s / L[i][i];
  }
  // (4) NLL
  double quad = 0.0, logdet = 0.0;
  for (int i = 0; i < N; ++i) { quad += y[i] * alpha[i]; logdet += std::log(L[i][i]); }
  const double TWO_PI = 6.28318530717958647692;
  return 0.5 * quad + logdet + 0.5 * N * std::log(TWO_PI);
}




void projectOnDevice(const DeviceGPModel& d_gp,
                     Kokkos::View<const double**, Kokkos::LayoutLeft, mem_space> centers_chunk, // (M×D_orig)
                     Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> d_Xstar)       // (M×d_latent)
{
  const int M = (int)centers_chunk.extent(0);
  const int D = d_gp.D_orig;
  const int r = d_gp.d_latent;

  // HSIC mode if we have dims mapping (we always allocate it)
  const bool hsic_mode = d_gp.uses_hsic;

  if (hsic_mode) {
    // O(M·r) direct gather + standardize
    Kokkos::parallel_for("project_hsic_direct",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{M,r}),
      KOKKOS_LAMBDA(int i, int k){
        const int dimk = d_gp.hsic_dims(k);
        // safety: if something goes wrong, map to 0
        const int dj = (0 <= dimk && dimk < D) ? dimk : 0;
        d_Xstar(i,k) = (centers_chunk(i,dj) - d_gp.mu_X(dj)) / d_gp.sd_X(dj);
      });
    return;
  }

  // PCA fallback (dense)
  Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> d_Xstd("Xstd", M, D);
  Kokkos::parallel_for("standardizeX",
    Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{M,D}),
    KOKKOS_LAMBDA(int i,int j){
      d_Xstd(i,j) = (centers_chunk(i,j) - d_gp.mu_X(j)) / d_gp.sd_X(j);
    });

  KokkosBlas::gemm("N","N", 1.0, d_Xstd, d_gp.B, 0.0, d_Xstar);
}


void evalObjectiveBatchGPU(const std::vector<double>& X_host_tail,
                           int new_N, int D,
                           int expenseFactor, ObjectiveType objType,
                           std::vector<double>& y_tail_out)
{
  using Exec   = Kokkos::DefaultExecutionSpace;
  using View2D = Kokkos::View<double**, Kokkos::LayoutLeft, Exec>;
  using View1D = Kokkos::View<double*,  Kokkos::LayoutLeft, Exec>;

  if (new_N <= 0) { y_tail_out.clear(); return; }

  if (usesExternalObjective(objType)) {
    y_tail_out.resize(new_N);
    external_eval_batch_host(X_host_tail.data(), D, new_N, y_tail_out.data());
    return;
  }

  View2D X_d("X_tail", new_N, D);
  Kokkos::deep_copy(
    X_d,
    Kokkos::View<const double**, Kokkos::LayoutLeft, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(X_host_tail.data(), new_N, D));

  View1D y_d("y_tail", new_N);

  if (usesAdditiveDeviceObjective(objType)) {
    Kokkos::parallel_for("eval_objective_gpu_tail_additive",
      Kokkos::TeamPolicy<Exec>(new_N, Kokkos::AUTO),
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const int i = team.league_rank();
        double termSum = 0.0;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, D),
          [&](int dim, double& s){
            const double xi = X_d(i, dim);
            s += objectiveTerm(objType, xi, dim, D, expenseFactor);
          }, termSum);
        if (team.team_rank() == 0) {
          y_d(i) = objectiveConstant(objType) + termSum;
        }
      });
  } else {
    Kokkos::parallel_for("eval_objective_gpu_tail_full",
      Kokkos::RangePolicy<Exec>(0, new_N),
      KOKKOS_LAMBDA(const int i) {
        y_d(i) = computeObjectiveRow(X_d, i, D, expenseFactor, objType);
      });
  }

  y_tail_out.resize(new_N);
  Kokkos::deep_copy(
    Kokkos::View<double*, Kokkos::HostSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>(y_tail_out.data(), new_N),
    y_d);
}

Model trainGP_from_precomputedY(const std::vector<double>& X_orig_host,
                                const std::vector<double>& y_phys,
                                int N, int d, int r,
                                double sigma_f, double l, double sigma_n,
                                int expenseFactor,
                                ObjectiveType objType,
                                FBOGPKernel kernelKind,
                                double rqAlpha)
{
  // This is a clone of trainGP() but *skips* the expensive "eval_objective" loop.
  PROFILE_SCOPE("trainGP_from_precomputedY");

  // (1) Standardize X
  std::vector<double> mu_X(d), sd_X(d);
  std::vector<double> X_phys = X_orig_host;
  for (int dim = 0; dim < d; ++dim) {
    std::vector<double> col(N);
    for (int i = 0; i < N; ++i) col[i] = X_phys[i * d + dim];
    standardise(col, mu_X[dim], sd_X[dim]);
    for (int i = 0; i < N; ++i) X_phys[i * d + dim] = col[i];
  }

  // (2) PCA with 95% EV cap r
  std::vector<double> X_norm_flat, pca_basis;
  int r_chosen = r;
  X_norm_flat = pcaEmbed95(X_phys, N, d, r, 0.95, &r_chosen, &pca_basis);

  // (3) Fill model basics
  Model model;
  model.N = N;
  model.d = r_chosen;
  model.D_orig = d;
  model.X_host = X_norm_flat;
  model.X_orig_host = X_orig_host;
  model.X.assign(N, std::vector<double>(r_chosen));
  for (int i = 0; i < N; ++i)
    for (int dim = 0; dim < r_chosen; ++dim)
      model.X[i][dim] = X_norm_flat[i * r_chosen + dim];
  model.pca_basis = std::move(pca_basis);
  model.uses_hsic = false;

  // (4) Use precomputed y_phys
  model.y_phys = y_phys;
  std::vector<double> y_std = y_phys;
  double mu_y, sd_y;
  standardise(y_std, mu_y, sd_y);
  model.y   = y_std;
  model.mu_X = mu_X;
  model.sd_X = sd_X;
  model.mu_y = mu_y;
  model.sd_y = sd_y;

  // (5) GPU: build K, chol, solve alpha (same as trainGP)
  {
    PROFILE_SCOPE("trainGP:gpu_buildK_chol_solve");
    using Exec     = Kokkos::DefaultExecutionSpace;
    using MemSpace = Exec::memory_space;
    using View2D   = Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace>;
    const int R = model.d;
    View2D X_d("X_d", N, R);
    {
      auto Xh = Kokkos::View<const double**, Kokkos::LayoutLeft,
                             Kokkos::HostSpace,
                             Kokkos::MemoryTraits<Kokkos::Unmanaged>>(
        model.X_host.data(), N, R);
      Kokkos::deep_copy(X_d, Xh);
    }
    View2D K_d("K_d", N, N);
    gpKernelSymmDevice(X_d,
                       l,
                       sigma_f,
                       sigma_n,
                       kernelKind,
                       rqAlpha,
                       K_d);
    Kokkos::fence();
    GpuSolverHandle H = nullptr;
    GpuSolverCreate(&H);
    int lwork = 0; int* dInfo = nullptr;
    GPU_MALLOC((void**)&dInfo, sizeof(int));
    gpu_potrf_buffersize(H, N, K_d.data(), N, &lwork);
    double* dWork = nullptr; GPU_MALLOC((void**)&dWork, sizeof(double)*lwork);
    gpu_potrf(H, N, K_d.data(), N, dWork, lwork, dInfo);
    GPU_DEV_SYNC();
    int hInfo = 0; GPU_MEMCPY(&hInfo, dInfo, sizeof(int), GPU_MEMCPY_D2H);

    if (hInfo != 0) { std::cerr << "cuSolver potrf failed, info=" << hInfo << std::endl; std::exit(EXIT_FAILURE); }
    View2D B("B", N, 1);
    {
      auto Bh = Kokkos::create_mirror_view(B);
      for (int i = 0; i < N; ++i) Bh(i,0) = model.y[i];
      Kokkos::deep_copy(B, Bh);
    }
    KokkosBlas::trsm("L","L","N","N", 1.0, K_d, B);
    KokkosBlas::trsm("L","L","T","N", 1.0, K_d, B);
    model.alpha.assign(N, 0.0);
    {
      auto Bh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);
      for (int i = 0; i < N; ++i) model.alpha[i] = Bh(i,0);
    }
    model.L.assign(N, std::vector<double>(N, 0.0));
    {
      auto Kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), K_d);
      for (int i = 0; i < N; ++i)
        for (int j = 0; j <= i; ++j) model.L[i][j] = Kh(i,j);
    }
    GPU_FREE(dWork); GPU_FREE(dInfo); GpuSolverDestroy(H);
  }
  model.logMarginal = 0.0;
  model.sigma_f = sigma_f;
  model.l = l;
  model.sigma_n = sigma_n;
  model.kernel = kernelKind;
  model.rq_alpha = rqAlpha;
  return model;
}
















Model trainGP_HSIC(const std::vector<double>& X_orig_host,
                   int N, int D,
                   int r_max,
                   double hsic_mass_thresh,
                   double sigma_f, double l, double sigma_n,
                   int expenseFactor,
                   ObjectiveType objType,
                   const std::string& score_type,
                   const std::string& kernel_var_method,
                   FBOGPKernel kernelKind,
                   double rqAlpha)
{

  using Exec     = Kokkos::DefaultExecutionSpace;
  using MemSpace = Exec::memory_space;
  using View2D_d = Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace>;
    PROFILE_SCOPE("trainGP_HSIC_total");

    if ((int)X_orig_host.size() != N * D) {
        throw std::runtime_error("trainGP_HSIC: X_orig_host.size() != N*D");
    }
    if (N <= 0 || D <= 0) {
        throw std::runtime_error("trainGP_HSIC: invalid N or D");
    }
    r_max = std::max(1, std::min(r_max, D));
    if (!(hsic_mass_thresh > 0.0 && hsic_mass_thresh <= 1.0)) {
        // allow "no threshold": behave like "take r_max"
        hsic_mass_thresh = 0.0;
    }

    // ----------------------------
    // (1) Standardize X on host
    // ----------------------------
    PROFILE_SCOPE("trainGP_HSIC:standardise_X");
    std::vector<double> mu_X(D), sd_X(D);
    std::vector<double> X_std = X_orig_host; // row-major, overwritten to standardized

    for (int dim = 0; dim < D; ++dim) {
        std::vector<double> col(N);
        for (int i = 0; i < N; ++i) col[i] = X_std[i * D + dim];
        standardise(col, mu_X[dim], sd_X[dim]);
        for (int i = 0; i < N; ++i) X_std[i * D + dim] = col[i];
    }

    // ----------------------------
    // (2) Evaluate y_phys on GPU (same kernel as trainGP)
    // ----------------------------
    PROFILE_SCOPE("trainGP_HSIC:eval_objective");
    std::vector<double> y_phys;
    evaluateObjectiveBatchAny(X_orig_host, N, D, expenseFactor, objType, y_phys);



    // ----------------------------
    // (3) HSIC embedding (host) -> Z (N×r), plus selected dims + scores
    // ----------------------------
    PROFILE_SCOPE("trainGP_HSIC:hsic_embed");
    std::vector<int> hsic_dims;
    std::vector<double> hsic_scores;
    std::vector<double> Z_flat = hsicEmbedUranie95(
        X_std, y_phys, N, D,
        r_max, hsic_mass_thresh,
        &hsic_dims,
        &hsic_scores,
        score_type,
        kernel_var_method
    );

    const int r = (int)hsic_dims.size();
    if (r <= 0) {
        // ultra-degenerate fallback: take first dim
        hsic_dims = {0};
        Z_flat.resize((size_t)N);
        for (int i = 0; i < N; ++i) Z_flat[i] = X_std[i * D + 0];
    }

    // ----------------------------
    // (4) Standardize y (host)
    // ----------------------------
    PROFILE_SCOPE("trainGP_HSIC:standardise_y");
    std::vector<double> y_std = y_phys;
    double mu_y = 0.0, sd_y = 1.0;
    standardise(y_std, mu_y, sd_y);

    // ----------------------------
    // (5) Fill model (host)
    // ----------------------------
    Model model;
    model.hsic_cfg.r_max = r_max;
    model.hsic_cfg.mass_thresh = hsic_mass_thresh;
    model.hsic_cfg.score_type = score_type;
    model.hsic_cfg.kernel_var_method = kernel_var_method;
    model.N      = N;
    model.d      = (int)hsic_dims.size(); // latent dim used by GP
    model.D_orig = D;

    model.X_orig_host = X_orig_host; // original
    model.X_host      = Z_flat;      // embedded coords (row-major N×r)

    model.X.assign(N, std::vector<double>(model.d));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < model.d; ++j)
            model.X[i][j] = Z_flat[i * model.d + j];

    model.y_phys = y_phys;
    model.y      = y_std;
    model.mu_X   = mu_X;
    model.sd_X   = sd_X;
    model.mu_y   = mu_y;
    model.sd_y   = sd_y;

    // IMPORTANT: HSIC does not give a basis like PCA; store selected dims.
    // (You should add `std::vector<int> hsic_dims;` to Model in the header.)
    model.hsic_dims = hsic_dims;
    model.hsic_scores = hsic_scores;
    model.uses_hsic = true;

    // Debug print (host): top selected HSIC dims + their scores
    static bool printed_once = false;
    if (!printed_once) {
      printed_once = true;
      std::cout << "[HSIC] selected r=" << model.d << " dims: ";
      for (int k = 0; k < model.d; ++k) {
        const int dj = model.hsic_dims[k];
        const double sc = (dj >= 0 && dj < (int)model.hsic_scores.size()) ? model.hsic_scores[dj] : 0.0;
        std::cout << dj << "(s=" << sc << ") ";
      }
      std::cout << "\n";
    }

    // ----------------------------
    // (6) GPU build K(Z,Z), Cholesky, solve alpha (same as trainGP)
    // ----------------------------
    PROFILE_SCOPE("trainGP_HSIC:gpu_buildK_chol_solve");
    using MemSpace = Exec::memory_space;
    using View2D_d = Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace>;

    View2D_d Z_d("Z_d", N, model.d);
    {
        auto Zh = Kokkos::View<const double**, Kokkos::LayoutLeft, Kokkos::HostSpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>(model.X_host.data(), N, model.d);
        Kokkos::deep_copy(Z_d, Zh);
    }

    View2D_d K_d("K_d", N, N);
    gpKernelSymmDevice(Z_d,
                       l,
                       sigma_f,
                       sigma_n,
                       kernelKind,
                       rqAlpha,
                       K_d);
    Kokkos::fence();

    GpuSolverHandle H = nullptr;
    GpuSolverCreate(&H);

    int lwork = 0;
    int* dInfo = nullptr;
    GPU_MALLOC((void**)&dInfo, sizeof(int));
    gpu_potrf_buffersize(H, N, K_d.data(), N, &lwork);

    double* dWork = nullptr;
    GPU_MALLOC((void**)&dWork, sizeof(double) * lwork);

    gpu_potrf(H, N, K_d.data(), N, dWork, lwork, dInfo);
    GPU_DEV_SYNC();

    int hInfo = 0;
    GPU_MEMCPY(&hInfo, dInfo, sizeof(int), GPU_MEMCPY_D2H);
    if (hInfo != 0) {
        std::cerr << "cuSolver potrf failed, info=" << hInfo << std::endl;
        std::exit(EXIT_FAILURE);
    }

    // Solve for alpha using TRSM twice
    View2D_d B("B", N, 1);
    {
        auto Bh = Kokkos::create_mirror_view(B);
        for (int i = 0; i < N; ++i) Bh(i,0) = model.y[i];
        Kokkos::deep_copy(B, Bh);
    }
    KokkosBlas::trsm("L","L","N","N", 1.0, K_d, B);
    KokkosBlas::trsm("L","L","T","N", 1.0, K_d, B);

    model.alpha.assign(N, 0.0);
    {
        auto Bh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);
        for (int i = 0; i < N; ++i) model.alpha[i] = Bh(i,0);
    }

    // Copy L back to host (lower triangle)
    model.L.assign(N, std::vector<double>(N, 0.0));
    {
        auto Kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), K_d);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j <= i; ++j)
                model.L[i][j] = Kh(i,j);
    }

    GPU_FREE(dWork);
    GPU_FREE(dInfo);
    GpuSolverDestroy(H);

    model.logMarginal = 0.0;
    model.sigma_f = sigma_f;
    model.l       = l;
    model.sigma_n = sigma_n;
    model.kernel  = kernelKind;
    model.rq_alpha = rqAlpha;

    return model;
}


















Model updateGP_incremental(const Model& oldModel,
                           const std::vector<double>& newPoints, int new_N,
                           int r, double sigma_f, double l, double sigma_n,
                           int expenseFactor,
                           ObjectiveType objType,
                           FBOGPKernel kernelKind,
                           double rqAlpha)
{
  const int D_orig = (int)(oldModel.X_orig_host.size() / oldModel.N);
  const int N_new_total = oldModel.N + new_N;

  // 1) concat X_orig_host
  std::vector<double> X_all; X_all.reserve((size_t)N_new_total * D_orig);
  X_all.insert(X_all.end(), oldModel.X_orig_host.begin(), oldModel.X_orig_host.end());
  X_all.insert(X_all.end(), newPoints.begin(), newPoints.end());

  // 2) y_all = oldModel.y_phys  + GPU-eval(newPoints)
  std::vector<double> y_all = oldModel.y_phys;
  if (new_N > 0) {
    std::vector<double> y_tail;
    evalObjectiveBatchGPU(newPoints, new_N, D_orig, expenseFactor, objType, y_tail);
    y_all.insert(y_all.end(), y_tail.begin(), y_tail.end());
  }

  // 3) train with precomputed y (skips re-evaluating old points)
  return trainGP_from_precomputedY(X_all,
                                   y_all,
                                   N_new_total,
                                   D_orig,
                                   r,
                                   sigma_f,
                                   l,
                                   sigma_n,
                                   expenseFactor,
                                   objType,
                                   kernelKind,
                                   rqAlpha);
}

Model updateGP_incremental_knownY(
    const Model& old,
    const std::vector<double>& X_new,
    const std::vector<double>& y_new,
    int n_add,
    int d_latent,
    double sigma_f,
    double l,
    double sigma_n,
    FBOGPKernel kernelKind,
    double rqAlpha)
{
    if (old.N <= 0) {
        throw std::runtime_error("old model has N==0");
    }

    const int D_orig = old.D_orig > 0
        ? old.D_orig
        : (int)(old.X_orig_host.size() / old.N);

    if ((int)X_new.size() != n_add * D_orig) {
        throw std::runtime_error("X_new.size() != n_add * D_orig");
    }

    if ((int)y_new.size() != n_add) {
        throw std::runtime_error("y_new.size() != n_add");
    }

    const int N_new_total = old.N + n_add;

    std::vector<double> X_all;
    X_all.reserve((size_t)N_new_total * (size_t)D_orig);
    X_all.insert(X_all.end(), old.X_orig_host.begin(), old.X_orig_host.end());
    X_all.insert(X_all.end(), X_new.begin(), X_new.end());

    std::vector<double> y_all;
    y_all.reserve((size_t)N_new_total);
    y_all.insert(y_all.end(), old.y_phys.begin(), old.y_phys.end());
    y_all.insert(y_all.end(), y_new.begin(), y_new.end());

    if (old.uses_hsic) {
        return trainGP_HSIC_from_precomputedY(
            X_all,
            y_all,
            N_new_total,
            D_orig,
            old.hsic_cfg.r_max,
            old.hsic_cfg.mass_thresh,
            sigma_f,
            l,
            sigma_n,
            old.hsic_cfg.score_type,
            old.hsic_cfg.kernel_var_method,
            kernelKind,
            rqAlpha);
    }

    return trainGP_from_precomputedY(
        X_all,
        y_all,
        N_new_total,
        D_orig,
        d_latent,
        sigma_f,
        l,
        sigma_n,
        1,
        ObjectiveType::Sphere,
        kernelKind,
        rqAlpha);
}

Model trainGP_HSIC_from_precomputedY(
    const std::vector<double>& X_orig_host,
    const std::vector<double>& y_phys,
    int N, int D,
    int r_max,
    double hsic_mass_thresh,
    double sigma_f, double l, double sigma_n,
    const std::string& score_type,
    const std::string& kernel_var_method,
    FBOGPKernel kernelKind,
    double rqAlpha)
{
    PROFILE_SCOPE("trainGP_HSIC_from_precomputedY");

    if ((int)X_orig_host.size() != N * D)
        throw std::runtime_error("trainGP_HSIC_from_precomputedY: X_orig_host.size() != N*D");
    if ((int)y_phys.size() != N)
        throw std::runtime_error("trainGP_HSIC_from_precomputedY: y_phys.size() != N");
    if (N <= 0 || D <= 0)
        throw std::runtime_error("trainGP_HSIC_from_precomputedY: invalid N or D");

    r_max = std::max(1, std::min(r_max, D));
    if (!(hsic_mass_thresh > 0.0 && hsic_mass_thresh <= 1.0)) hsic_mass_thresh = 0.0;

    // (1) Standardize X
    PROFILE_SCOPE("trainGP_HSIC:standardise_X");
    std::vector<double> mu_X(D), sd_X(D);
    std::vector<double> X_std = X_orig_host;
    for (int dim = 0; dim < D; ++dim) {
        std::vector<double> col(N);
        for (int i = 0; i < N; ++i) col[i] = X_std[i * D + dim];
        standardise(col, mu_X[dim], sd_X[dim]);
        for (int i = 0; i < N; ++i) X_std[i * D + dim] = col[i];
    }

    // (2) HSIC embed using provided y_phys
    PROFILE_SCOPE("trainGP_HSIC:hsic_embed");
    std::vector<int> hsic_dims;
    std::vector<double> hsic_scores;
    std::vector<double> Z_flat = hsicEmbedUranie95(
        X_std, y_phys, N, D, r_max, hsic_mass_thresh,
        &hsic_dims, &hsic_scores, score_type, kernel_var_method);

    if (hsic_dims.empty()) hsic_dims = {0};
    const int r = (int)hsic_dims.size();
    if ((int)Z_flat.size() != N * r)
        throw std::runtime_error("trainGP_HSIC_from_precomputedY: Z_flat.size() != N*r");

    // (3) Standardize y
    PROFILE_SCOPE("trainGP_HSIC:standardise_y");
    std::vector<double> y_std = y_phys;
    double mu_y = 0.0, sd_y = 1.0;
    standardise(y_std, mu_y, sd_y);

    // (4) Fill model
    Model model;
    model.N      = N;
    model.d      = r;
    model.D_orig = D;
    model.X_orig_host = X_orig_host;
    model.X_host      = Z_flat;
    model.X.assign(N, std::vector<double>(r));
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < r; ++j)
            model.X[i][j] = Z_flat[i * r + j];

    model.y_phys = y_phys;
    model.y      = y_std;
    model.mu_X   = mu_X;
    model.sd_X   = sd_X;
    model.mu_y   = mu_y;
    model.sd_y   = sd_y;
    model.hsic_dims = hsic_dims;
    model.hsic_scores = hsic_scores;
    model.uses_hsic = true;


// Debug print (host): top selected HSIC dims + their scores
    static bool printed_once = false;
    if (!printed_once) {
      printed_once = true;
      std::cout << "[HSIC] selected r=" << model.d << " dims: ";
      for (int k = 0; k < model.d; ++k) {
        const int dj = model.hsic_dims[k];
        const double sc = (dj >= 0 && dj < (int)model.hsic_scores.size()) ? model.hsic_scores[dj] : 0.0;
        std::cout << dj << "(s=" << sc << ") ";
      }
      std::cout << "\n";
    }



    // (5) GPU build K(Z,Z), chol, solve alpha (same as trainGP_HSIC)
    PROFILE_SCOPE("trainGP_HSIC:gpu_buildK_chol_solve");
    using Exec     = Kokkos::DefaultExecutionSpace;
    using MemSpace = Exec::memory_space;
    using View2D_d = Kokkos::View<double**, Kokkos::LayoutLeft, MemSpace>;

    View2D_d Z_d("Z_d", N, r);
    {
        auto Zh = Kokkos::View<const double**, Kokkos::LayoutLeft, Kokkos::HostSpace,
                               Kokkos::MemoryTraits<Kokkos::Unmanaged>>(model.X_host.data(), N, r);
        Kokkos::deep_copy(Z_d, Zh);
    }

    View2D_d K_d("K_d", N, N);
    gpKernelSymmDevice(Z_d,
                       l,
                       sigma_f,
                       sigma_n,
                       kernelKind,
                       rqAlpha,
                       K_d);
    Kokkos::fence();

    GpuSolverHandle H = nullptr;
    GpuSolverCreate(&H);
    int lwork = 0;
    int* dInfo = nullptr;
    GPU_MALLOC((void**)&dInfo, sizeof(int));
    gpu_potrf_buffersize(H, N, K_d.data(), N, &lwork);
    double* dWork = nullptr;
    GPU_MALLOC((void**)&dWork, sizeof(double) * lwork);
    gpu_potrf(H, N, K_d.data(), N, dWork, lwork, dInfo);
    GPU_DEV_SYNC();
    int hInfo = 0;
    GPU_MEMCPY(&hInfo, dInfo, sizeof(int), GPU_MEMCPY_D2H);
    if (hInfo != 0) {
        std::cerr << "cuSolver potrf failed, info=" << hInfo << std::endl;
        std::exit(EXIT_FAILURE);
    }

    View2D_d B("B", N, 1);
    {
        auto Bh = Kokkos::create_mirror_view(B);
        for (int i = 0; i < N; ++i) Bh(i,0) = model.y[i];
        Kokkos::deep_copy(B, Bh);
    }
    KokkosBlas::trsm("L","L","N","N", 1.0, K_d, B);
    KokkosBlas::trsm("L","L","T","N", 1.0, K_d, B);

    model.alpha.assign(N, 0.0);
    {
        auto Bh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B);
        for (int i = 0; i < N; ++i) model.alpha[i] = Bh(i,0);
    }

    model.L.assign(N, std::vector<double>(N, 0.0));
    {
        auto Kh = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), K_d);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j <= i; ++j)
                model.L[i][j] = Kh(i,j);
    }

    GPU_FREE(dWork);
    GPU_FREE(dInfo);
    GpuSolverDestroy(H);

    model.logMarginal = 0.0;
    model.sigma_f = sigma_f;
    model.l       = l;
    model.sigma_n = sigma_n;
    model.kernel  = kernelKind;
    model.rq_alpha = rqAlpha;
    model.hsic_cfg.r_max = r_max;
    model.hsic_cfg.mass_thresh = hsic_mass_thresh;
    model.hsic_cfg.score_type = score_type;
    model.hsic_cfg.kernel_var_method = kernel_var_method;
    return model;
}

} // end namespace GP
