#ifndef HPC_FUNCTIONS_HPP
#define HPC_FUNCTIONS_HPP

/*--------------------------------------------------------------------
 *  Updated header with “neutral‑cost” loops
 *  ---------------------------------------------------------------
 *  – Each computeXYZTerm() now executes an artificial workload that
 *    scales with `expenseFactor`, **but** the contribution is
 *    mathematically neutral (returns the same value as before).
 *  – We rely on `fma(0.0, extra, base)` to guarantee the expensive
 *    summation `extra` is evaluated yet cancels to zero.
 *-------------------------------------------------------------------*/

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <mpi.h>

#include <vector>
#include <utility>
#include <cmath>
#include <numeric>
#include <limits>
#include <algorithm>

// For convenience
using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space  = typename exec_space::memory_space;

// ──────────────────────────────────────────────────────────────────
// Basic data structures
// ──────────────────────────────────────────────────────────────────
using HypersphereID = std::vector<std::pair<int,int>>;

struct Hypersphere {
    HypersphereID id;
    int           dimension;
    int           depth;
    KOKKOS_INLINE_FUNCTION
    Hypersphere(int dim, int d, const HypersphereID& identifier)
      : id(identifier), dimension(dim), depth(d) {}
};

struct SearchSpaceBounds {
    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> lowerBounds;
    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> upperBounds;
    int dimension;

    SearchSpaceBounds(int n,
                      const Kokkos::View<double*,Kokkos::LayoutLeft,mem_space>& l,
                      const Kokkos::View<double*,Kokkos::LayoutLeft,mem_space>& u)
        : lowerBounds(l), upperBounds(u), dimension(n) {}
};

// ──────────────────────────────────────────────────────────────────
// Objective types & dispatcher
// ──────────────────────────────────────────────────────────────────
enum class ObjectiveType { Sphere, Rastrigin, Ellipsoid };

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double computeObjective(const View1D& x, int expenseFactor, ObjectiveType objType);

// ──────────────────────────────────────────────────────────────────
// Heavy‑loop helpers that keep exact same return value
// ──────────────────────────────────────────────────────────────────

// 1) Sphere term   ────────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION
double computeSphereTerm(double x, int expenseFactor)
{
    double extra = 0.0;
    for (int i = 0; i < expenseFactor * 50; ++i) {
        extra += sin(x + double(i));   // heavy numerical work
    }
    // fma guarantees the loop cannot be optimised away
    return fma(0.0, extra, x * x);     // == x²  (value unchanged)
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double sphereObjectiveFunction(const View1D& point, int expenseFactor)
{
    double sum = 0.0;
    const int D = point.extent(0);
    for (int d = 0; d < D; ++d) {
        sum += computeSphereTerm(point(d), expenseFactor);
    }
    return sum;
}

// 2) Rastrigin term ───────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION
double computeRastriginTerm(double x, int expenseFactor)
{
    const double base = x * x - 10.0 * cos(2.0 * M_PI * x) + 10.0;
    double extra = 0.0;
    for (int i = 0; i < expenseFactor * 50; ++i) {
        extra += sin(x + double(i));
    }
    return fma(0.0, extra, base);      // base unchanged, work kept
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double rastriginObjectiveFunction(const View1D& x, int expenseFactor)
{
    double sum = 0.0;
    const int D = x.extent(0);
    for (int i = 0; i < D; ++i) {
        sum += computeRastriginTerm(x(i), expenseFactor);
    }
    return sum;
}

// 3) Ellipsoid term ───────────────────────────────────────────────
KOKKOS_INLINE_FUNCTION
double computeEllipsoidTerm(double xi, int i, int D, int expenseFactor)
{
    const double exponent = (D > 1 ? (6.0 * i) / (D - 1) : 0.0);
    const double weight   = pow(10.0, exponent);

    double extra = 0.0;
    for (int k = 0; k < expenseFactor * 50; ++k) {
        extra += sin(xi + double(k));
    }
    return fma(0.0, extra, weight * xi * xi);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double ellipsoidObjectiveFunction(const View1D& x, int expenseFactor)
{
    const int D = x.extent(0);
    double sum = 0.0;
    for (int i = 0; i < D; ++i) {
        sum += computeEllipsoidTerm(x(i), i, D, expenseFactor);
    }
    return sum;
}

// Generic dispatcher
template<typename View1D>
KOKKOS_INLINE_FUNCTION
double computeObjective(const View1D& x, int expenseFactor, ObjectiveType objType)
{
    switch (objType) {
        case ObjectiveType::Sphere:    return sphereObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Rastrigin: return rastriginObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Ellipsoid: return ellipsoidObjectiveFunction(x, expenseFactor);
        default:                       return sphereObjectiveFunction(x, expenseFactor);
    }
}

// ──────────────────────────────────────────────────────────────────
// Forward declarations (unchanged)
// ──────────────────────────────────────────────────────────────────
std::vector<int> serializeHypersphereID(const HypersphereID& id);
HypersphereID    deserializeHypersphereID(const std::vector<int>& data, size_t& offset);

Kokkos::View<double*, mem_space>
reconstructCenter(const Hypersphere& hs,
                  const Kokkos::View<double*, mem_space>& initCenter,
                  double initRadius);

Kokkos::View<double**, mem_space>
generateRandomPointsAroundCenter(const Kokkos::View<double*, mem_space>& center,
                                 double radius,
                                 int dimension);

void prepareHypersphereData(const std::vector<Hypersphere>& hList,
                            Kokkos::View<int*,    mem_space>& depths,
                            Kokkos::View<size_t*, mem_space>& idOffsets,
                            Kokkos::View<int*,    mem_space>& idDims,
                            Kokkos::View<int*,    mem_space>& idSigns);

void reconstructCentersAndRadii(size_t nHypers,
                                int dimension,
                                double initRadius,
                                const Kokkos::View<int*, mem_space>& depths,
                                const Kokkos::View<size_t*, mem_space>& idOffsets,
                                const Kokkos::View<int*, mem_space>& idDims,
                                const Kokkos::View<int*, mem_space>& idSigns,
                                const Kokkos::View<double*, mem_space>& initCenterDev,
                                Kokkos::View<double**, mem_space>& centers,
                                Kokkos::View<double*,  mem_space>& radii);

void scoreHyperspheres(size_t nHypers,
                       int dimension,
                       const Kokkos::View<double**, mem_space>& centers,
                       const Kokkos::View<double*,  mem_space>& radii,
                       const Kokkos::View<double*,  mem_space>& bestSolutionDev,
                       double bestObjVal,
                       Kokkos::View<double*, mem_space>& scores,
                       int expenseFactor,
                       ObjectiveType objType);

// Host‑side helpers
//-------------------------------------------------------------

double computeObjectiveValue(const Kokkos::View<double*, mem_space>& solution,
                             int expenseFactor,
                             ObjectiveType objType);

double computeEllipsoidObjectiveValue(const Kokkos::View<double*, mem_space>& solution,
                                      int expenseFactor);

// Local search
Kokkos::View<double*, mem_space>
intensiveLocalSearch(const Hypersphere&                    hs,
                     const Kokkos::View<double*, mem_space>& initCenter,
                     double                                 initRadius,
                     int                                    maxIters,
                     double                                 stepSize,
                     double                                 phi,
                     double                                 omegaMin,
                     ObjectiveType                          objType,
                     int                                    expenseFactor);

// Decomposition / utilities
std::vector<Hypersphere> decomposeHypersphere(const Hypersphere& parent);
std::vector<int>         selectBestIndices(const std::vector<double>& scores, int beta);
size_t                   get_available_memory();

// CLI parsing
int parseBeta(int argc, char* argv[], int wrank);
int parseDimension(int argc, char* argv[], int wrank);
int parseBudget(int argc, char* argv[], int rank);
int parseExpenseFactor(int argc, char* argv[], int rank);
ObjectiveType parseObjectiveType(int argc, char* argv[], int idx, int wrank);

#endif // HPC_FUNCTIONS_HPP
