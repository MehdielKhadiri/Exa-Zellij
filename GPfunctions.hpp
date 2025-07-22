#ifndef GP_FUNCTIONS_HPP
#define GP_FUNCTIONS_HPP

#include <Kokkos_Core.hpp>
#include <vector>
#include "HPCFunctions.hpp"

// Decide which memory space to use for your Kokkos device data
using mem_space  = Kokkos::DefaultExecutionSpace::memory_space;
using exec_space = Kokkos::DefaultExecutionSpace;

namespace GP {

  // --------------------------------------------------------------------------
  // GP Model structure.
  // --------------------------------------------------------------------------
  struct Model {
      // CPU-based data
      std::vector<std::vector<double>> X;  // Training points (N x d)
      std::vector<double>               y;  // Function values at training points
      std::vector<std::vector<double>> L;  // Cholesky factor (N x N)
      std::vector<double>               alpha; // GP weights (length N)

      int N; // number of training points
      int d; // dimension
      double logMarginal;  // (not fully computed here)

      // Flattened X for convenience (size = N*d)
      // Some code uses flattened arrays for easy copying to device
      std::vector<double> X_host;
  };

  // --------------------------------------------------------------------------
  // A device-friendly functor to compute UCB and LCB in parallel for
  // multiple "test" (candidate) points, given your GP model info on device.
  // --------------------------------------------------------------------------
  struct PredictAcquisition {
    // Inputs
    Kokkos::View<const double**, Kokkos::LayoutLeft, mem_space> Xtest;   // [numCandidates, d]
    Kokkos::View<const double**, Kokkos::LayoutLeft, mem_space> Xtrain;  // [N, d]
    Kokkos::View<const double**, Kokkos::LayoutLeft, mem_space> Ltrain;  // [N, N] (Cholesky)
    Kokkos::View<const double*,  Kokkos::LayoutLeft, mem_space> alpha;   // length N
    double sigma_f;
    double l;
    double kappa;
    int    N;  // number of training points
    int    d;  // dimension

    // Outputs
    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> ucb; // length = numCandidates
    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> lcb; // length = numCandidates

    // Constructor
    PredictAcquisition(
      Kokkos::View<const double**, Kokkos::LayoutLeft, mem_space> Xtest_,
      Kokkos::View<const double**, Kokkos::LayoutLeft, mem_space> Xtrain_,
      Kokkos::View<const double**, Kokkos::LayoutLeft, mem_space> Ltrain_,
      Kokkos::View<const double*,  Kokkos::LayoutLeft, mem_space> alpha_,
      double sigma_f_,
      double l_,
      double kappa_,
      int N_,
      int d_)
      : Xtest(Xtest_), Xtrain(Xtrain_), Ltrain(Ltrain_), alpha(alpha_),
        sigma_f(sigma_f_), l(l_), kappa(kappa_), N(N_), d(d_)
    {
      // Allocate the output arrays
      const int numCandidates = Xtest_.extent(0);
      ucb = Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>("UCB", numCandidates);
      lcb = Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>("LCB", numCandidates);
    }

    // The operator that runs in parallel over each candidate index i
    KOKKOS_FUNCTION
    void operator()(const int i) const {
      // 1) Build covariance vector k_star between Xtest[i] and each training point
      double* k_star = (double*)alloca(N*sizeof(double));
      for(int j = 0; j < N; j++){
        // compute squared distance
        double sqdist = 0.0;
        for(int dim = 0; dim < d; dim++){
          double diff = Xtest(i, dim) - Xtrain(j, dim);
          sqdist += diff * diff;
        }
        // kernel = sigma_f^2 * exp(-0.5*sqdist/l^2)
        k_star[j] = sigma_f*sigma_f * exp(-0.5 * sqdist / (l*l));
      }

      // 2) Predictive mean mu = k_star^T alpha
      double mu = 0.0;
      for(int j = 0; j < N; j++){
        mu += k_star[j] * alpha(j);
      }

      // 3) Solve Ltrain * v = k_star  (forward-substitution)
      double* v = (double*)alloca(N*sizeof(double));
      for(int r = 0; r < N; r++){
        double sum = k_star[r];
        for(int c = 0; c < r; c++){
          sum -= Ltrain(r, c)*v[c];
        }
        v[r] = sum / Ltrain(r, r);
      }

      // 4) k_xx = sigma_f^2 for SE kernel at the same point
      double k_xx = sigma_f*sigma_f;

      // 5) sigma^2 = k_xx - norm2(v)
      double v_norm2 = 0.0;
      for(int r = 0; r < N; r++){
        v_norm2 += v[r]*v[r];
      }
      double var = k_xx - v_norm2;
      if(var < 0.0) var = 0.0;
      double sigma = sqrt(var);

      // 6) UCB & LCB
      ucb(i) = mu + kappa*sigma;
      lcb(i) = mu - kappa*sigma;
    }
  };

  // --------------------------------------------------------------------------
  // Latin Hypercube Sampling. Returns a flattened array of size N*d.
  // --------------------------------------------------------------------------
  std::vector<double> latinHypercubeSample(const int N,
                                           const int d,
                                           const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& lowerBounds,
                                           const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& upperBounds,
                                           const unsigned seed = 0);

  // --------------------------------------------------------------------------
  // Train a GP on the Rastrigin function (or whichever) using
  //   X_host: flattened array of training points (size = N*d).
  //   (sigma_f, l, sigma_n): hyperparams
  // Returns a Model with alpha, L, etc. set.
  // --------------------------------------------------------------------------
  Model trainGP(const std::vector<double>& X_host,
                const int N,
                const int d,
                const double sigma_f,
                const double l,
                const double sigma_n,
                int expenseFactor,
                ObjectiveType objType);

  // --------------------------------------------------------------------------
  // Update an existing GP model by adding "new_N" new points (flattened array)
  // and re-training from scratch (naive approach).
  // --------------------------------------------------------------------------
  Model updateGP(const Model& oldModel,
                 const std::vector<double>& newPoints,
                 const int new_N,
                 const double sigma_f,
                 const double l,
                 const double sigma_n,
                 int expenseFactor,
                 ObjectiveType objType);

  // --------------------------------------------------------------------------
  // CPU-based Prediction: For a single test point x_star, compute mu & sigma
  //   using the model's L, alpha, etc.
  // --------------------------------------------------------------------------
  void predictGP(const Model& model,
                 const std::vector<double>& x_star,
                 double sigma_f,
                 double l,
                 double& mu,
                 double& sigma);

  // --------------------------------------------------------------------------
  // CPU-based Overload: Return mu, and also pass out sigma. (Used in acquisitions)
  // --------------------------------------------------------------------------
  double predictGP(const Model& model,
                   const std::vector<double>& x_star,
                   const double kappa,
                   double& sigma_out);

  // --------------------------------------------------------------------------
  // Kernel function: Squared exponential
  //   k(x,x') = sigma_f^2 * exp(-0.5 * ||x - xp||^2 / l^2)
  // --------------------------------------------------------------------------
  double kernel(const std::vector<double>& x,
                const std::vector<double>& xp,
                double l,
                double sigma_f);

  // --------------------------------------------------------------------------
  // Acquisition Functions for Minimization:
  //   UCB(x) = mu(x) + kappa*sigma(x)
  //   LCB(x) = mu(x) - kappa*sigma(x)
  //   EI(x)  = "expected improvement" (one-sided, for minimization)
  // --------------------------------------------------------------------------
  double acquisitionUCB(const Model& model,
                        const std::vector<double>& x,
                        const double kappa);

  double acquisitionLCB(const Model& model,
                        const std::vector<double>& x,
                        const double kappa);

  double acquisitionEI(const Model& model,
                       const std::vector<double>& x,
                       const double f_min);



                       
// --------------------------------------------------------------------------
// q-Expected-Improvement (QEI) pour la minimisation :
//   X_batch  : vecteur de points (q ≥ 1) ; si q == 1 ⇒ QEI = EI
//   f_min    : meilleure valeur observée
//   num_samples : nombre d’échantillons Monte-Carlo (par défaut 10 000)
// --------------------------------------------------------------------------
double acquisitionQEI(const Model&                              model,
                      const std::vector<std::vector<double>>&   X_batch,
                      const double                              f_min,
                      const int                                 num_samples = 10000);


  // --------------------------------------------------------------------------
  // GP-BUCB Batch Selection for Minimization (CPU-based demonstration).
  // Returns a vector of selected candidate points (each a std::vector<double>).
  // --------------------------------------------------------------------------
  std::vector<std::vector<double>> gp_bucb_batch_min(
      const Model& model,
      const Kokkos::View<double**, Kokkos::LayoutLeft>& candidatePoints,
      double kappa,
      int batch_size,
      double sigma_n,
      double sigma_f,
      double l);

  // Embedding functions (PCA or Nonlinear embedding)
  std::vector<double> linearEmbeddingPCA(const std::vector<double>& points, int numPoints, int origDim, int reducedDim);
  std::vector<double> nonlinearEmbedding(const std::vector<double>& points, int numPoints, int origDim, int reducedDim);


  struct DeviceGPModel {
  // 1-D view: length N
  Kokkos::View<double*, Kokkos::LayoutRight, mem_space>  alpha; // (N)

  // 2-D view: N × Dred (same as before)
  Kokkos::View<double**, Kokkos::LayoutRight, mem_space> X;     // (N × Dred)

  double sigma_f, l, sigma_n;
};


/* build a device mirror once after training -------------------------------- */
DeviceGPModel makeDeviceCopy(const GP::Model& h,
                             double sigma_f, double l, double sigma_n);

/* kernel: predict mean & variance for a batch of points -------------------- */
void batchPredictGP(const DeviceGPModel& d_gp,
                    Kokkos::View<const double**, mem_space>  d_Xstar, // (M × Dred)
                    Kokkos::View<double*,      mem_space>   d_mu,     // (M)
                    Kokkos::View<double*,      mem_space>   d_var);   // (M)



double negLogMarginalLikelihood(const Model&     m,
                                  double           sigma_f,
                                  double           l,
                                  double           sigma_n);



} // end namespace GP

#endif // GP_FUNCTIONS_HPP
