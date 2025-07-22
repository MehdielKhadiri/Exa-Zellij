#include <cmath>
#include <algorithm>
#include <random>
#include <iostream>
#include <chrono>
#include <limits>
#include <Eigen/Dense>
#include "GPfunctions.hpp"
#include "HPCFunctions.hpp" 


namespace GP {

  // --------------------------------------------------------------------------
  // kernel function
  // --------------------------------------------------------------------------
  double kernel(const std::vector<double>& x,
                const std::vector<double>& xp,
                double l,
                double sigma_f) {
    double sqdist = 0.0;
    int d = x.size();
    for (int i = 0; i < d; i++){
      double diff = x[i] - xp[i];
      sqdist += diff*diff;
    }
    return sigma_f*sigma_f * std::exp(-0.5 * sqdist / (l*l));
  }

  // --------------------------------------------------------------------------
  // Latin Hypercube Sampling Implementation (host code).
  // Returns a flattened array X_host of length N*d
  // --------------------------------------------------------------------------
  std::vector<double> latinHypercubeSample(const int N,
                                           const int d,
                                           const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& lowerBounds,
                                           const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& upperBounds,
                                           const unsigned seed) {
    std::vector<double> X_host(N*d, 0.0);

    // Copy device->host for bounds
    auto lower_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), lowerBounds);
    auto upper_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), upperBounds);

    // rng
    std::mt19937 gen(seed ? seed : std::random_device{}());
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for(int dim = 0; dim < d; dim++){
      // perm of [0..N-1]
      std::vector<int> perm(N);
      for(int i=0; i<N; i++){
        perm[i] = i;
      }
      std::shuffle(perm.begin(), perm.end(), gen);

      double LB = lower_host(dim);
      double UB = upper_host(dim);

      for(int i=0; i<N; i++){
        double randU = dist(gen); // in [0,1]
        double coord = LB + ((perm[i] + randU)/double(N)) * (UB - LB);
        X_host[i*d + dim] = coord;
      }
    }

    return X_host;
  }


  // --------------------------------------------------------------------------
  // trainGP: builds kernel matrix, does Cholesky, etc.
  // We evaluate f(x) = Rastrigin(x) here for demonstration
  // --------------------------------------------------------------------------
  Model trainGP(const std::vector<double>& X_host,
      const int N,
      const int d,
      const double sigma_f,
      const double l,
      const double sigma_n,
      int expenseFactor,
      ObjectiveType objType) {
    Model model;
    model.N = N;
    model.d = d;
    model.X_host = X_host; // store flattened
    model.y.assign(N, 0.0);

    // Unflatten into 2D
    model.X.resize(N, std::vector<double>(d, 0.0));
    for(int i = 0; i < N; i++){
      for(int j = 0; j < d; j++){
        model.X[i][j] = X_host[i*d + j];
      }
    }

    // Évaluation générique
    for(int i=0; i<N; i++){
      // on réutilise le vecteur plat de la ligne i
      std::vector<double> xi_vec(d);
      for(int j=0; j<d; j++) xi_vec[j] = model.X[i][j];
      // wrapper lambdas pour Kokkos::View-like
      struct View1D_Host {
        double* data; int N;
        double operator()(int k) const { return data[k]; }
        int extent(int) const { return N; }
      } view{ xi_vec.data(), d };
      model.y[i] = computeObjective(view, expenseFactor, objType);
    }

    // Build NxN kernel matrix K
    std::vector<std::vector<double>> K(N, std::vector<double>(N,0.0));
    for(int i = 0; i < N; i++){
      for(int j = 0; j < N; j++){
        K[i][j] = kernel(model.X[i], model.X[j], l, sigma_f);
        if(i == j){
          K[i][j] += sigma_n * sigma_n;
        }
      }
    }

    // Cholesky decomposition
    model.L.resize(N, std::vector<double>(N,0.0));
    for(int i = 0; i < N; i++){
      for(int j = 0; j <= i; j++){
        double sum = 0.0;
        for(int k = 0; k < j; k++){
          sum += model.L[i][k] * model.L[j][k];
        }
        if(i == j) {
          model.L[i][j] = std::sqrt(K[i][i] - sum);
        } else {
          model.L[i][j] = (K[i][j] - sum) / model.L[j][j];
        }
      }
    }

    // Solve L z = y
    std::vector<double> z(N, 0.0);
    for(int i = 0; i < N; i++){
      double sum = 0.0;
      for(int j = 0; j < i; j++){
        sum += model.L[i][j] * z[j];
      }
      z[i] = (model.y[i] - sum) / model.L[i][i];
    }

    // Solve L^T alpha = z
    model.alpha.resize(N, 0.0);
    for(int i = N - 1; i >= 0; i--){
      double sum = 0.0;
      for(int j = i + 1; j < N; j++){
        sum += model.L[j][i] * model.alpha[j];
      }
      model.alpha[i] = (z[i] - sum) / model.L[i][i];
    }

    model.logMarginal = 0.0; // not fully computed in this snippet
    return model;
  }


  // --------------------------------------------------------------------------
  // updateGP: naive approach, merges oldModel's points + newPoints => re-trains
  // --------------------------------------------------------------------------
  Model updateGP(const Model& oldModel,
                 const std::vector<double>& newPoints,
                 const int new_N,
                 const double sigma_f,
                 const double l,
                 const double sigma_n,
                 int expenseFactor,
                 ObjectiveType objType) {
    // Merge old + new
    int oldN = oldModel.N;
    int d    = oldModel.d;
    int totalN = oldN + new_N;

    // flatten old
    std::vector<double> combined;
    combined.reserve(totalN*d);
    // old
    combined.insert(combined.end(),
                    oldModel.X_host.begin(),
                    oldModel.X_host.end());
    // new
    combined.insert(combined.end(),
                    newPoints.begin(),
                    newPoints.end());

    // re-train
    return trainGP(combined, totalN, d, sigma_f, l, sigma_n, expenseFactor, objType);
  }

  // --------------------------------------------------------------------------
  // CPU-based predictGP
  // --------------------------------------------------------------------------
  void predictGP(const Model& model,
                 const std::vector<double>& x_star,
                 double sigma_f,
                 double l,
                 double& mu,
                 double& sigma) {
    int N = model.N;
    int d = model.d;

    // 1) build k_star
    std::vector<double> k_star(N, 0.0);
    for(int i=0; i<N; i++){
      k_star[i] = kernel(x_star, model.X[i], l, sigma_f);
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
    double k_xx = sigma_f*sigma_f;

    // 5) sigma^2 = k_xx - v^T v
    double var = 0.0;
    for(int i=0; i<N; i++){
      var += v[i]*v[i];
    }
    var = k_xx - var;
    if(var < 0.0) var=0.0;
    sigma = std::sqrt(var);
  }


  double predictGP(const Model& model,
                   const std::vector<double>& x_star,
                   const double kappa,
                   double& sigma_out) {
    double mu;
    // for demonstration, assume the same sigma_f,l as in the model
    // but you might keep them in the model struct
    predictGP(model, x_star, 1.0, 1.0, mu, sigma_out);
    return mu;
  }

  // --------------------------------------------------------------------------
  // Acquisitions
  // --------------------------------------------------------------------------
  double acquisitionUCB(const Model& model,
                        const std::vector<double>& x,
                        const double kappa) {
    double sigma;
    double mu = predictGP(model, x, kappa, sigma);
    return mu + kappa*sigma;
  }

  double acquisitionLCB(const Model& model,
                        const std::vector<double>& x,
                        const double kappa) {
    double sigma;
    double mu = predictGP(model, x, kappa, sigma);
    return mu - kappa*sigma;
  }

  double acquisitionEI(const Model& model,
                       const std::vector<double>& x,
                       const double f_min) {
    double sigma;
    double mu = predictGP(model, x, 0.0, sigma);
    if(sigma<1e-12){
      return 0.0;
    }
    double z = (f_min - mu)/sigma;
    double cdf = 0.5*(1.0 + std::erf(z / std::sqrt(2.0)));
    double pdf = (1.0/std::sqrt(2.0*M_PI)) * std::exp(-0.5*z*z);
    double ei = (f_min - mu)*cdf + sigma*pdf;
    if(ei<0.0) ei=0.0;
    return ei;
  }




// --------------------------------------------------------------------------
// Monte-Carlo q-Expected-Improvement  (minimisation; GP posterior gaussien)
// --------------------------------------------------------------------------
double acquisitionQEI(const Model&                            model,
                      const std::vector<std::vector<double>>& X_batch,
                      const double                            f_min,
                      const int                               num_samples)
{
  const int q = static_cast<int>(X_batch.size());
  if (q == 0) return 0.0;
  if (q == 1)                                      // dégénérescence : QEI = EI
      return acquisitionEI(model, X_batch[0], f_min);

  const int N       = model.N;                    // nb. points d’entraînement
  const double ell  = 1.0;                        // longueur-de-corrélation
  const double s_f  = 1.0;                        // variance signal
  /* 1)  moyenne  μ_q  et  matrice-covariance  Σ_q  de la loi jointe
         f_q | 𝔇_n  ~  𝓝( μ_q , Σ_q )                                         */
  std::vector<double>                  mu(q, 0.0);
  std::vector<std::vector<double>>     v(q, std::vector<double>(N, 0.0));
  Eigen::MatrixXd                      Sigma(q, q);

  for (int idx = 0; idx < q; ++idx)
  {
    /* ----- vecteur k_* (N × 1)  ----- */
    std::vector<double> k_star(N);
    for (int j = 0; j < N; ++j)
      k_star[j] = kernel(X_batch[idx], model.X[j], ell, s_f);

    /* ----- μ_i = k_*ᵀ α  ----- */
    double mu_i = 0.0;
    for (int j = 0; j < N; ++j) mu_i += k_star[j] * model.alpha[j];
    mu[idx] = mu_i;

    /* ----- solve  L v_i = k_*  (v_i = K^{-1/2} k_*) ----- */
    std::vector<double>& v_i = v[idx];
    for (int r = 0; r < N; ++r)
    {
      double sum = k_star[r];
      for (int c = 0; c < r; ++c) sum -= model.L[r][c] * v_i[c];
      v_i[r] = sum / model.L[r][r];
    }
  }
  /* ----- Σ_q(i,j) = k(x_i,x_j) − v_iᵀ v_j  ----- */
  for (int i = 0; i < q; ++i)
  {
    for (int j = 0; j < q; ++j)
    {
      double k_xixj = kernel(X_batch[i], X_batch[j], ell, s_f);
      double dot    = 0.0;
      for (int t = 0; t < N; ++t) dot += v[i][t] * v[j][t];
      double cov = k_xixj - dot;
      if (cov < 0.0) cov = 0.0;                   // stabilisation numérique
      Sigma(i, j) = cov;
    }
  }

  /* 2)  Factorisation de Cholesky  Σ_q = LLᵀ  avec jitter si nécessaire  */
  const double jitter = 1e-12;
  Eigen::LLT<Eigen::MatrixXd> llt((Sigma
                                   + jitter * Eigen::MatrixXd::Identity(q, q)));
  Eigen::MatrixXd L = llt.matrixL();

  /* 3)  Monte-Carlo                                    */
  std::mt19937                     gen(std::random_device{}());
  std::normal_distribution<double> dist(0.0, 1.0);

  Eigen::VectorXd mu_vec(q);
  for (int i = 0; i < q; ++i) mu_vec(i) = mu[i];

  double acc_improvement = 0.0;

  for (int s = 0; s < num_samples; ++s)
  {
    Eigen::VectorXd z(q);
    for (int i = 0; i < q; ++i) z(i) = dist(gen); // z ~ 𝓝(0, I)
    Eigen::VectorXd f_s = mu_vec + L * z;         // un tirage de f_q

    const double f_best_s = f_s.minCoeff();       // min_j f^{(j)}
    const double imp_s    = f_min - f_best_s;     // (f* − min)_+
    if (imp_s > 0.0) acc_improvement += imp_s;
  }
  return acc_improvement / static_cast<double>(num_samples);
}






  // --------------------------------------------------------------------------
  // gp_bucb_batch_min
  // --------------------------------------------------------------------------
  std::vector<std::vector<double>> gp_bucb_batch_min(
      const Model& model,
      const Kokkos::View<double**, Kokkos::LayoutLeft>& candidatePoints,
      double kappa,
      int batch_size,
      double sigma_n,
      double sigma_f,
      double l) {

    // Copy candidatePoints to host
    auto candHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), candidatePoints);
    int numCandidates = candHost.extent(0);
    int d = candHost.extent(1);

    // For each candidate, compute mu, sigma on CPU
    struct Candidate {
      int idx;
      std::vector<double> x;
      double mu;
      double sigma;
      double acq; // = mu - kappa*sigma
    };

    std::vector<Candidate> cands;
    cands.reserve(numCandidates);

    // Precompute mu, sigma on CPU
    for(int i=0; i<numCandidates; i++){
      Candidate C;
      C.idx = i;
      C.x.resize(d);
      for(int j=0; j<d; j++){
        C.x[j] = candHost(i,j);
      }
      double mu_, sigma_;
      predictGP(model, C.x, sigma_f, l, mu_, sigma_);
      C.mu = mu_;
      C.sigma = sigma_;
      C.acq = C.mu - kappa*C.sigma; // LCB
      cands.push_back(C);
    }

    std::vector<Candidate> selected;
    selected.reserve(batch_size);

    std::vector<bool> used(numCandidates, false);
    std::vector<double> var(numCandidates, 0.0);
    for(int i=0; i<numCandidates; i++){
      var[i] = cands[i].sigma * cands[i].sigma;
    }

    for(int b=0; b<batch_size; b++){
      // pick min LCB
      int bestIdx = -1;
      double bestVal = std::numeric_limits<double>::infinity();
      for(int i=0; i<numCandidates; i++){
        if(!used[i]){
          if(cands[i].acq < bestVal){
            bestVal = cands[i].acq;
            bestIdx = i;
          }
        }
      }
      if(bestIdx<0) break;
      used[bestIdx] = true;
      selected.push_back(cands[bestIdx]);

      // update variance for all others
      double k_xx = kernel(cands[bestIdx].x, cands[bestIdx].x, l, sigma_f);
      double denom = k_xx + sigma_n*sigma_n;
      for(int i=0; i<numCandidates; i++){
        if(used[i]) continue;
        double k_val = kernel(cands[i].x, cands[bestIdx].x, l, sigma_f);
        double reduc = (k_val*k_val)/denom;
        var[i] = std::max(var[i] - reduc, 0.0);
        double sig = std::sqrt(var[i]);
        cands[i].acq = cands[i].mu - kappa*sig; // LCB
      }
    }

    // return the chosen points
    std::vector<std::vector<double>> batch;
    batch.reserve(selected.size());
    for(auto& c: selected){
      batch.push_back(c.x);
    }
    return batch;
  }

  // --------------------------------------------------------------------------
  // PCA-based embedding
  // --------------------------------------------------------------------------
  std::vector<double> linearEmbeddingPCA(const std::vector<double>& points, int numPoints, int origDim, int reducedDim) {
    Eigen::MatrixXd dataMat(numPoints, origDim);
    for(int i = 0; i < numPoints; i++)
        for(int j = 0; j < origDim; j++)
            dataMat(i, j) = points[i*origDim + j];

    Eigen::MatrixXd centered = dataMat.rowwise() - dataMat.colwise().mean();
    Eigen::MatrixXd cov = (centered.adjoint() * centered) / double(numPoints - 1);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(cov);

    Eigen::MatrixXd eigVec = eig.eigenvectors().rightCols(reducedDim);
    Eigen::MatrixXd reduced = centered * eigVec;

    std::vector<double> embedded(numPoints*reducedDim);
    for(int i = 0; i < numPoints; i++)
        for(int j = 0; j < reducedDim; j++)
            embedded[i*reducedDim + j] = reduced(i,j);

    return embedded;
  }

  // --------------------------------------------------------------------------
  // Nonlinear embedding placeholder
  // --------------------------------------------------------------------------
  std::vector<double> nonlinearEmbedding(const std::vector<double>& points, int numPoints, int origDim, int reducedDim) {
    // Implement or call nonlinear embedding here.
    // For now, just calls linear PCA as placeholder.
    return linearEmbeddingPCA(points, numPoints, origDim, reducedDim);
  }




  // alias the spaces we used in the header
using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space  = exec_space::memory_space;
using layout_t   = Kokkos::LayoutRight;

// ----------------------------------------------------------------------------
// 1) Copy the trained CPU model to device memory once
// ----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// makeDeviceCopy
// -----------------------------------------------------------------------------
DeviceGPModel makeDeviceCopy(const Model& h,
                             double sigma_f, double l, double sigma_n)
{
  DeviceGPModel d;
  d.sigma_f = sigma_f;
  d.l       = l;
  d.sigma_n = sigma_n;

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

  return d;
}
// ----------------------------------------------------------------------------
// 2) Predict mean/variance for a batch of M points that already live on GPU
//    (very plain, O(M·N·d); good enough to get you running)
// ----------------------------------------------------------------------------
void batchPredictGP(const DeviceGPModel& d_gp,
                    Kokkos::View<const double**, mem_space> d_Xstar,
                    Kokkos::View<double*,      mem_space> d_mu,
                    Kokkos::View<double*,      mem_space> d_var)
{
  const int M = d_Xstar.extent(0);
  const int D = d_Xstar.extent(1);
  const int N = d_gp.alpha.extent(0);

  const double sigma_f2 = d_gp.sigma_f * d_gp.sigma_f;
  const double l2       = d_gp.l * d_gp.l;

  Kokkos::parallel_for(
      "batchPredictGP",
      Kokkos::RangePolicy<exec_space>(0, M),
      KOKKOS_LAMBDA(const int i) {

        // --- predictive mean -------------------------------------------------
        double mu = 0.0;

        // --- naive variance (full GP would need extra solves; here we
        //     approximate with prior variance so you have *some* value) ------
        double var = sigma_f2;

        for (int n = 0; n < N; ++n) {
          // squared distance between test-point i and training-point n
          double sqdist = 0.0;
          for (int d = 0; d < D; ++d) {
            const double diff = d_Xstar(i, d) - d_gp.X(n, d);
            sqdist += diff * diff;
          }
          const double k_star = sigma_f2 * exp(-0.5 * sqdist / l2);
          mu  += k_star * d_gp.alpha(n);
          // a full treatment would also update var here
        }

        d_mu(i)  = mu;
        d_var(i) = var;   // coarse approximation
      });
}

// ─── NEW: negative log marginal likelihood (exact GP, SE kernel) ───────────────
double negLogMarginalLikelihood(const Model& m,
                                    double       sigma_f,
                                    double       l,
                                    double       sigma_n)
{
  const int N = m.N;
  // log |K|   →  sum log diag(L)  (L is Cholesky of K)
  double logDetK = 0.0;
  for (int i = 0; i < N; ++i) logDetK += std::log(m.L[i][i]);

  // ½ yᵀ α  (α already stored in the model)
  double quad = 0.0;
  for (int i = 0; i < N; ++i) quad += m.y[i] * m.alpha[i];

  const double nll = 0.5 * quad + logDetK + 0.5 * N * std::log(2.0 * M_PI);
  return nll;
}

} // end namespace GP
