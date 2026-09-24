#ifndef GP_FUNCTIONS_HPP
#define GP_FUNCTIONS_HPP

#include <Kokkos_Core.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <stack>
#include <string>
#include <unordered_map>
#include <vector>

#include "HPCfunctions.hpp"

namespace timing {
    using clock = std::chrono::steady_clock;

    struct Region {
        double total = 0.0;
        double self = 0.0;
    };

    struct StackEntry {
        std::string name;
        clock::time_point t0;
        double child_time = 0.0;
    };

    struct Registry {
        std::unordered_map<std::string, Region> regions;
        std::stack<StackEntry> stack;

        static Registry& instance()
        {
            static Registry reg;
            return reg;
        }

        void enter(const std::string& name)
        {
            stack.push(StackEntry{name, clock::now(), 0.0});
        }

        void exit(const std::string& name)
        {
            if (stack.empty()) return;
            auto entry = stack.top();
            stack.pop();
            const auto dt = std::chrono::duration<double, std::milli>(clock::now() - entry.t0).count();
            const double self_time = dt - entry.child_time;
            regions[name].total += dt;
            regions[name].self += self_time;
            if (!stack.empty()) stack.top().child_time += dt;
        }

        void print(std::ostream& os) const
        {
            os << "\n---------------- EXCLUSIVE Timings (seconds) -----------------\n";
            for (const auto& kv : regions) {
                os << std::setw(34) << kv.first << " : " << kv.second.self / 1e3 << " s\n";
            }
            os << "-------------------------------------------------------------\n";
        }
    };

    class Scoped {
        std::string name;
    public:
        explicit Scoped(const std::string& n) : name(n) { Registry::instance().enter(name); }
        ~Scoped() { Registry::instance().exit(name); }
    };
}

#define TOKENPASTE2(x, y) x##y
#define TOKENPASTE(x, y) TOKENPASTE2(x, y)
#ifndef PROFILE_SCOPE
#define PROFILE_SCOPE(name) timing::Scoped TOKENPASTE(_timer_, __LINE__)(name)
#endif

using layout_t = Kokkos::LayoutLeft;

namespace GP {

using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space  = exec_space::memory_space;
using layout_t   = Kokkos::LayoutLeft;

struct HSICConfig {
    int r_max = 1;
    double mass_thresh = 0.95;
    std::string score_type = "R2HSic";
    std::string kernel_var_method = "median";
};

struct Model {
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    std::vector<std::vector<double>> L;
    std::vector<double> alpha;

    int N = 0;
    int d = 0;
    int D_orig = 0;
    double logMarginal = 0.0;

    std::vector<double> X_host;
    std::vector<double> X_orig_host;

    std::vector<double> mu_X;
    std::vector<double> sd_X;
    double mu_y = 0.0;
    double sd_y = 1.0;
    std::vector<double> y_phys;

    std::vector<double> pca_basis;
    std::vector<int> hsic_dims;
    std::vector<double> hsic_scores;
    bool uses_hsic = false;

    HSICConfig hsic_cfg;

    double sigma_f = 1.0;
    double l = 1.0;
    double sigma_n = 1e-6;
    FBOGPKernel kernel = FBOGPKernel::SquaredExponential;
    double rq_alpha = 1.0;
};

struct DeviceGPModel {
    Kokkos::View<double*,  layout_t, mem_space> alpha;
    Kokkos::View<double**, layout_t, mem_space> X;
    Kokkos::View<double**, layout_t, mem_space> L;

    Kokkos::View<double*,  layout_t, mem_space> mu_X;
    Kokkos::View<double*,  layout_t, mem_space> sd_X;
    Kokkos::View<double**, layout_t, mem_space> B;
    Kokkos::View<int*,     layout_t, mem_space> hsic_dims;

    int Ntrain = 0;
    int d_latent = 0;
    int D_orig = 0;
    double sigma_f = 1.0;
    double l = 1.0;
    double sigma_n = 1e-6;
    FBOGPKernel kernel = FBOGPKernel::SquaredExponential;
    double rq_alpha = 1.0;
    bool uses_hsic = false;
};

void rbfKernelMatrixDevice(Kokkos::View<const double**, layout_t, mem_space> A,
                           Kokkos::View<const double**, layout_t, mem_space> B,
                           double ell,
                           double sigma_f,
                           Kokkos::View<double**, layout_t, mem_space> K_out);

void rbfKernelSymmDevice(Kokkos::View<const double**, layout_t, mem_space> X,
                         double ell,
                         double sigma_f,
                         double sigma_n,
                         Kokkos::View<double**, layout_t, mem_space> K_out);

double mean(const std::vector<double>& v);
double stdev(const std::vector<double>& v, double mu);
void standardise(std::vector<double>& v, double& mu, double& sd);

double evalObjective(const std::vector<double>& x, ObjectiveType obj, int expenseFactor);
double kernel(const std::vector<double>& x, const std::vector<double>& xp, double l, double sigma_f);
double kernel(const std::vector<double>& x,
              const std::vector<double>& xp,
              double l,
              double sigma_f,
              FBOGPKernel kernelKind,
              double rqAlpha);

std::vector<double> pcaEmbed(const std::vector<double>& X_flat,
                             int N,
                             int D,
                             int R,
                             std::vector<double>* out_basis = nullptr);

std::vector<double> pcaEmbed95(const std::vector<double>& X_flat,
                               int N,
                               int D,
                               int R_max,
                               double var_thresh,
                               int* out_R,
                               std::vector<double>* out_basis = nullptr);

std::vector<double> latinHypercubeSample(int N,
                                         int d,
                                         const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& lowerBounds,
                                         const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& upperBounds,
                                         unsigned seed = 0);

Model trainGP(const std::vector<double>& X_host,
              int N,
              int d,
              int r,
              double sigma_f,
              double l,
              double sigma_n,
              int expenseFactor,
              ObjectiveType objType,
              FBOGPKernel kernelKind = FBOGPKernel::SquaredExponential,
              double rqAlpha = 1.0);

Model updateGP(const Model& oldModel,
               const std::vector<double>& newPoints,
               int new_N,
               int r,
               double sigma_f,
               double l,
               double sigma_n,
               int expenseFactor,
               ObjectiveType objType,
               FBOGPKernel kernelKind = FBOGPKernel::SquaredExponential,
               double rqAlpha = 1.0);

void predictGP(const Model& model,
               const std::vector<double>& x_star,
               double sigma_f,
               double l,
               double sigma_n,
               double& mu,
               double& sigma);

double predictGP(const Model& model,
                 const std::vector<double>& x_star,
                 double kappa,
                 double& sigma_out);

DeviceGPModel makeDeviceCopy(const Model& h,
                             double sigma_f,
                             double l,
                             double sigma_n,
                             FBOGPKernel kernelKind = FBOGPKernel::SquaredExponential,
                             double rqAlpha = 1.0);

void batchPredictGP(const DeviceGPModel& d_gp,
                    Kokkos::View<const double**, Kokkos::LayoutLeft, mem_space> d_Xstar,
                    Kokkos::View<double*, mem_space> d_mu,
                    Kokkos::View<double*, mem_space> d_var);

double negLogMarginalLikelihood(const Model& m,
                                double sigma_f,
                                double l,
                                double sigma_n,
                                FBOGPKernel kernelKind = FBOGPKernel::SquaredExponential,
                                double rqAlpha = 1.0);

void evalObjectiveBatchGPU(const std::vector<double>& X_host_tail,
                           int new_N,
                           int D,
                           int expenseFactor,
                           ObjectiveType objType,
                           std::vector<double>& y_tail_out);

Model trainGP_from_precomputedY(const std::vector<double>& X_orig_host,
                                const std::vector<double>& y_phys,
                                int N,
                                int d,
                                int r,
                                double sigma_f,
                                double l,
                                double sigma_n,
                                int expenseFactor,
                                ObjectiveType objType,
                                FBOGPKernel kernelKind = FBOGPKernel::SquaredExponential,
                                double rqAlpha = 1.0);

Model updateGP_incremental(const Model& oldModel,
                           const std::vector<double>& newPoints,
                           int new_N,
                           int r,
                           double sigma_f,
                           double l,
                           double sigma_n,
                           int expenseFactor,
                           ObjectiveType objType,
                           FBOGPKernel kernelKind = FBOGPKernel::SquaredExponential,
                           double rqAlpha = 1.0);

void projectOnDevice(const DeviceGPModel& d_gp,
                     Kokkos::View<const double**, Kokkos::LayoutLeft, mem_space> centers_chunk,
                     Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> d_Xstar);

Model updateGP_incremental_knownY(const Model& old,
                                  const std::vector<double>& X_new,
                                  const std::vector<double>& y_new,
                                  int n_add,
                                  int d_latent,
                                  double sigma_f,
                                  double l,
                                  double sigma_n,
                                  FBOGPKernel kernelKind = FBOGPKernel::SquaredExponential,
                                  double rqAlpha = 1.0);

Model trainGP_HSIC(const std::vector<double>& X_orig_host,
                   int N,
                   int D,
                   int r_max,
                   double hsic_mass_thresh,
                   double sigma_f,
                   double l,
                   double sigma_n,
                   int expenseFactor,
                   ObjectiveType objType,
                   const std::string& score_type = "R2HSic",
                   const std::string& kernel_var_method = "median",
                   FBOGPKernel kernelKind = FBOGPKernel::SquaredExponential,
                   double rqAlpha = 1.0);

Model trainGP_HSIC_from_precomputedY(const std::vector<double>& X_orig_host,
                                     const std::vector<double>& y_phys,
                                     int N,
                                     int D,
                                     int r_max,
                                     double hsic_mass_thresh,
                                     double sigma_f,
                                     double l,
                                     double sigma_n,
                                     const std::string& score_type = "R2HSic",
                                     const std::string& kernel_var_method = "median",
                                     FBOGPKernel kernelKind = FBOGPKernel::SquaredExponential,
                                     double rqAlpha = 1.0);

} // namespace GP

#endif // GP_FUNCTIONS_HPP
