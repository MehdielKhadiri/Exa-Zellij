#ifndef HPC_FUNCTIONS_HPP
#define HPC_FUNCTIONS_HPP

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <mpi.h>

#include "FBOTypes.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iosfwd>
#include <limits>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space  = typename exec_space::memory_space;

// Minimal shared enum.
enum class ObjectiveType {
    Sphere,
    Rastrigin,
    Ellipsoid,
    Rosenbrock,
    Ackley,
    Levy,
    Multimodal,
    VQE,
    ShiftedSphere,
    Schwefel,
    Griewank,
    Zakharov,
    StyblinskiTang,
    DixonPrice,
    BentCigar,
    Discus,
    SharpRidge,
    DifferentPowers
};

inline bool usesExternalObjective(ObjectiveType objType)
{
    return objType == ObjectiveType::VQE;
}

inline bool usesAdditiveDeviceObjective(ObjectiveType objType)
{
    switch (objType) {
        case ObjectiveType::Sphere:
        case ObjectiveType::Rastrigin:
        case ObjectiveType::Ellipsoid:
        case ObjectiveType::Levy:
        case ObjectiveType::Multimodal:
        case ObjectiveType::ShiftedSphere:
        case ObjectiveType::Schwefel:
        case ObjectiveType::StyblinskiTang:
        case ObjectiveType::BentCigar:
        case ObjectiveType::Discus:
            return true;
        default:
            return false;
    }
}

struct ObjectiveInfo {
    ObjectiveType type;
    const char* name;
    const char* regime;
    double defaultLower;
    double defaultUpper;
    bool separable;
    bool additiveFastPath;
    bool external;
    bool hasKnownOptimum;
};

std::vector<ObjectiveInfo> listObjectiveInfos();
const ObjectiveInfo& objectiveInfo(ObjectiveType objType);
std::string objectiveCanonicalName(ObjectiveType objType);
std::string availableObjectiveNames();
double objectiveDefaultLowerBound(ObjectiveType objType);
double objectiveDefaultUpperBound(ObjectiveType objType);
bool objectiveHasKnownOptimum(ObjectiveType objType);
double objectiveReferenceCoordinate(ObjectiveType objType, int dimIndex, int dimension);
double objectiveKnownMinimumValue(ObjectiveType objType, int dimension);
void printObjectiveInfo(ObjectiveType objType, int dimension, std::ostream& os);

using ExternalObjectiveFn     = double(*)(const double*, int);
using ExternalObjectiveManyFn = void(*)(const double*, int, int, double*);

struct ExternalObjective {
    ExternalObjectiveFn fn = nullptr;
    ExternalObjectiveManyFn fn_many = nullptr;
};

inline ExternalObjective& globalExternalObjective()
{
    static ExternalObjective g;
    return g;
}

using HypersphereID = std::vector<std::pair<int, int>>;

struct Hypersphere {
    HypersphereID id;
    int dimension = 0;
    int depth = 0;

    Hypersphere() = default;
    Hypersphere(int dim, int d, const HypersphereID& identifier)
        : id(identifier), dimension(dim), depth(d) {}
    Hypersphere(int dim, int d, HypersphereID&& identifier)
        : id(std::move(identifier)), dimension(dim), depth(d) {}
};

struct SearchSpaceBounds {
    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> lowerBounds;
    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> upperBounds;
    int dimension = 0;

    SearchSpaceBounds(int n,
                      const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& l,
                      const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& u)
        : lowerBounds(l), upperBounds(u), dimension(n) {}
};

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double computeObjective(const View1D& x, int expenseFactor, ObjectiveType objType);

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double computeBuiltInObjective(const View1D& x, int expenseFactor, ObjectiveType objType);

template<typename View2D>
struct ObjectiveRowView {
    View2D X;
    int row = 0;
    int dimension = 0;

    KOKKOS_INLINE_FUNCTION
    ObjectiveRowView(const View2D& X_, int row_, int dimension_)
        : X(X_), row(row_), dimension(dimension_) {}

    KOKKOS_INLINE_FUNCTION
    int extent(int) const { return dimension; }

    KOKKOS_INLINE_FUNCTION
    double operator()(int dim) const { return X(row, dim); }
};

template<typename View2D>
KOKKOS_INLINE_FUNCTION
double computeObjectiveRow(const View2D& X,
                           int row,
                           int dimension,
                           int expenseFactor,
                           ObjectiveType objType)
{
    ObjectiveRowView<View2D> x(X, row, dimension);
    return computeBuiltInObjective(x, expenseFactor, objType);
}

KOKKOS_INLINE_FUNCTION
double computeSphereTerm(double x, int expenseFactor)
{
    double extra = 0.0;
    for (int i = 0; i < expenseFactor * 50; ++i) {
        extra += sin(x + double(i));
    }
    return fma(0.0, extra, x * x);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double sphereObjectiveFunction(const View1D& point, int expenseFactor)
{
    double sum = 0.0;
    const int D = point.extent(0);
    for (int d = 0; d < D; ++d) sum += computeSphereTerm(point(d), expenseFactor);
    return sum;
}

KOKKOS_INLINE_FUNCTION
double computeRastriginTerm(double x, int expenseFactor)
{
    const double base = x * x - 10.0 * cos(2.0 * M_PI * x) + 10.0;
    double extra = 0.0;
    for (int i = 0; i < expenseFactor * 50; ++i) {
        extra += sin(x + double(i));
    }
    return fma(0.0, extra, base);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double rastriginObjectiveFunction(const View1D& x, int expenseFactor)
{
    double sum = 0.0;
    const int D = x.extent(0);
    for (int i = 0; i < D; ++i) sum += computeRastriginTerm(x(i), expenseFactor);
    return sum;
}

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
    for (int i = 0; i < D; ++i) sum += computeEllipsoidTerm(x(i), i, D, expenseFactor);
    return sum;
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double rosenbrockObjectiveFunction(const View1D& x, int expenseFactor)
{
    double sum = 0.0;
    const int D = x.extent(0);
    for (int i = 0; i + 1 < D; ++i) {
        const double xi = x(i);
        const double xj = x(i + 1);
        const double a = xj - xi * xi;
        const double b = 1.0 - xi;
        double extra = 0.0;
        for (int k = 0; k < expenseFactor * 50; ++k) {
            extra += sin(xi + xj + double(k));
        }
        sum += fma(0.0, extra, 100.0 * a * a + b * b);
    }
    return sum;
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double ackleyObjectiveFunction(const View1D& x, int expenseFactor)
{
    const int D = x.extent(0);
    if (D == 0) return 0.0;

    double sq = 0.0;
    double cs = 0.0;
    for (int i = 0; i < D; ++i) {
        const double xi = x(i);
        sq += xi * xi;
        cs += cos(2.0 * M_PI * xi);
    }

    double extra = 0.0;
    for (int k = 0; k < expenseFactor * 50; ++k) {
        extra += sin(double(k));
    }

    const double meanSq = sq / double(D);
    const double meanCos = cs / double(D);
    const double base = -20.0 * exp(-0.2 * sqrt(meanSq))
                      - exp(meanCos)
                      + 20.0
                      + M_E;
    return fma(0.0, extra, base);
}

KOKKOS_INLINE_FUNCTION
double computeLevyTerm(double x, int i, int D, int expenseFactor)
{
    const double w = 1.0 + (x - 1.0) / 4.0;
    double base = 0.0;

    if (i == 0) {
        const double s = sin(M_PI * w);
        base += s * s;
    }

    if (i + 1 < D) {
        const double s = sin(M_PI * w + 1.0);
        const double wm = w - 1.0;
        base += wm * wm * (1.0 + 10.0 * s * s);
    } else {
        const double s = sin(2.0 * M_PI * w);
        const double wm = w - 1.0;
        base += wm * wm * (1.0 + s * s);
    }

    double extra = 0.0;
    for (int k = 0; k < expenseFactor * 50; ++k) {
        extra += sin(x + double(k));
    }
    return fma(0.0, extra, base);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double levyObjectiveFunction(const View1D& x, int expenseFactor)
{
    const int D = x.extent(0);
    double sum = 0.0;
    for (int i = 0; i < D; ++i) sum += computeLevyTerm(x(i), i, D, expenseFactor);
    return sum;
}

KOKKOS_INLINE_FUNCTION
double computeMultiModalTerm(double x, int expenseFactor)
{
    const double base = 20.0
                      + M_E
                      - 20.0 * exp(-0.2 * fabs(x))
                      - exp(cos(2.0 * M_PI * x));
    double extra = 0.0;
    for (int i = 0; i < expenseFactor * 50; ++i) {
        extra += sin(x + double(i));
    }
    return fma(0.0, extra, base);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double multimodalObjectiveFunction(const View1D& x, int expenseFactor)
{
    double sum = 0.0;
    const int D = x.extent(0);
    for (int d = 0; d < D; ++d) sum += computeMultiModalTerm(x(d), expenseFactor);
    return sum;
}

KOKKOS_INLINE_FUNCTION
double computeShiftedSphereTerm(double x, int expenseFactor)
{
    const double y = x - 1.0;
    const double base = y * y;
    double extra = 0.0;
    for (int i = 0; i < expenseFactor * 50; ++i) {
        extra += sin(x + double(i));
    }
    return fma(0.0, extra, base);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double shiftedSphereObjectiveFunction(const View1D& x, int expenseFactor)
{
    double sum = 5.0;
    const int D = x.extent(0);
    for (int d = 0; d < D; ++d) sum += computeShiftedSphereTerm(x(d), expenseFactor);
    return sum;
}

KOKKOS_INLINE_FUNCTION
double computeSchwefelTerm(double x, int expenseFactor)
{
    const double base = 418.9828872724338 - x * sin(sqrt(fabs(x)));
    double extra = 0.0;
    for (int i = 0; i < expenseFactor * 50; ++i) {
        extra += sin(x + double(i));
    }
    return fma(0.0, extra, base);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double schwefelObjectiveFunction(const View1D& x, int expenseFactor)
{
    double sum = 0.0;
    const int D = x.extent(0);
    for (int d = 0; d < D; ++d) sum += computeSchwefelTerm(x(d), expenseFactor);
    return sum;
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double griewankObjectiveFunction(const View1D& x, int expenseFactor)
{
    const int D = x.extent(0);
    double sum = 0.0;
    double prod = 1.0;
    for (int i = 0; i < D; ++i) {
        const double xi = x(i);
        sum += xi * xi / 4000.0;
        prod *= cos(xi / sqrt(double(i + 1)));
    }

    double extra = 0.0;
    for (int k = 0; k < expenseFactor * 50; ++k) {
        extra += sin(double(k));
    }
    return fma(0.0, extra, 1.0 + sum - prod);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double zakharovObjectiveFunction(const View1D& x, int expenseFactor)
{
    const int D = x.extent(0);
    double sumSq = 0.0;
    double linear = 0.0;
    for (int i = 0; i < D; ++i) {
        const double xi = x(i);
        sumSq += xi * xi;
        linear += 0.5 * double(i + 1) * xi;
    }

    double extra = 0.0;
    for (int k = 0; k < expenseFactor * 50; ++k) {
        extra += sin(double(k));
    }
    const double l2 = linear * linear;
    return fma(0.0, extra, sumSq + l2 + l2 * l2);
}

KOKKOS_INLINE_FUNCTION
double computeStyblinskiTangTerm(double x, int expenseFactor)
{
    const double x2 = x * x;
    const double base = 0.5 * (x2 * x2 - 16.0 * x2 + 5.0 * x);
    double extra = 0.0;
    for (int i = 0; i < expenseFactor * 50; ++i) {
        extra += sin(x + double(i));
    }
    return fma(0.0, extra, base);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double styblinskiTangObjectiveFunction(const View1D& x, int expenseFactor)
{
    double sum = 0.0;
    const int D = x.extent(0);
    for (int d = 0; d < D; ++d) sum += computeStyblinskiTangTerm(x(d), expenseFactor);
    return sum;
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double dixonPriceObjectiveFunction(const View1D& x, int expenseFactor)
{
    const int D = x.extent(0);
    if (D == 0) return 0.0;

    const double x0m1 = x(0) - 1.0;
    double sum = x0m1 * x0m1;
    for (int i = 1; i < D; ++i) {
        const double xi = x(i);
        const double xprev = x(i - 1);
        const double a = 2.0 * xi * xi - xprev;
        sum += double(i + 1) * a * a;
    }

    double extra = 0.0;
    for (int k = 0; k < expenseFactor * 50; ++k) {
        extra += sin(double(k));
    }
    return fma(0.0, extra, sum);
}

KOKKOS_INLINE_FUNCTION
double computeBentCigarTerm(double x, int dim, int expenseFactor)
{
    const double weight = dim == 0 ? 1.0 : 1.0e6;
    double extra = 0.0;
    for (int i = 0; i < expenseFactor * 50; ++i) {
        extra += sin(x + double(i));
    }
    return fma(0.0, extra, weight * x * x);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double bentCigarObjectiveFunction(const View1D& x, int expenseFactor)
{
    double sum = 0.0;
    const int D = x.extent(0);
    for (int d = 0; d < D; ++d) sum += computeBentCigarTerm(x(d), d, expenseFactor);
    return sum;
}

KOKKOS_INLINE_FUNCTION
double computeDiscusTerm(double x, int dim, int expenseFactor)
{
    const double weight = dim == 0 ? 1.0e6 : 1.0;
    double extra = 0.0;
    for (int i = 0; i < expenseFactor * 50; ++i) {
        extra += sin(x + double(i));
    }
    return fma(0.0, extra, weight * x * x);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double discusObjectiveFunction(const View1D& x, int expenseFactor)
{
    double sum = 0.0;
    const int D = x.extent(0);
    for (int d = 0; d < D; ++d) sum += computeDiscusTerm(x(d), d, expenseFactor);
    return sum;
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double sharpRidgeObjectiveFunction(const View1D& x, int expenseFactor)
{
    const int D = x.extent(0);
    if (D == 0) return 0.0;

    double tail = 0.0;
    for (int i = 1; i < D; ++i) tail += x(i) * x(i);

    double extra = 0.0;
    for (int k = 0; k < expenseFactor * 50; ++k) {
        extra += sin(double(k));
    }
    const double base = x(0) * x(0) + 100.0 * sqrt(tail);
    return fma(0.0, extra, base);
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double differentPowersObjectiveFunction(const View1D& x, int expenseFactor)
{
    const int D = x.extent(0);
    double sum = 0.0;
    for (int i = 0; i < D; ++i) {
        const double exponent = D > 1
            ? 2.0 + 4.0 * double(i) / double(D - 1)
            : 2.0;
        sum += pow(fabs(x(i)), exponent);
    }

    double extra = 0.0;
    for (int k = 0; k < expenseFactor * 50; ++k) {
        extra += sin(double(k));
    }
    return fma(0.0, extra, sqrt(sum));
}

KOKKOS_INLINE_FUNCTION
double objectiveTerm(ObjectiveType objType, double xi, int dim, int D, int expenseFactor)
{
    switch (objType) {
        case ObjectiveType::Sphere:        return computeSphereTerm(xi, expenseFactor);
        case ObjectiveType::Rastrigin:     return computeRastriginTerm(xi, expenseFactor);
        case ObjectiveType::Ellipsoid:     return computeEllipsoidTerm(xi, dim, D, expenseFactor);
        case ObjectiveType::Levy:          return computeLevyTerm(xi, dim, D, expenseFactor);
        case ObjectiveType::Multimodal:    return computeMultiModalTerm(xi, expenseFactor);
        case ObjectiveType::ShiftedSphere: return computeShiftedSphereTerm(xi, expenseFactor);
        case ObjectiveType::Schwefel:      return computeSchwefelTerm(xi, expenseFactor);
        case ObjectiveType::StyblinskiTang: return computeStyblinskiTangTerm(xi, expenseFactor);
        case ObjectiveType::BentCigar:     return computeBentCigarTerm(xi, dim, expenseFactor);
        case ObjectiveType::Discus:        return computeDiscusTerm(xi, dim, expenseFactor);
        case ObjectiveType::Rosenbrock:    return 0.0;
        case ObjectiveType::Ackley:        return 0.0;
        case ObjectiveType::Griewank:      return 0.0;
        case ObjectiveType::Zakharov:      return 0.0;
        case ObjectiveType::DixonPrice:    return 0.0;
        case ObjectiveType::SharpRidge:    return 0.0;
        case ObjectiveType::DifferentPowers: return 0.0;
        case ObjectiveType::VQE:           return 0.0;
        default:                           return 0.0;
    }
}

KOKKOS_INLINE_FUNCTION
double objectiveConstant(ObjectiveType objType)
{
    switch (objType) {
        case ObjectiveType::ShiftedSphere: return 5.0;
        default:                           return 0.0;
    }
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double vqeObjectiveFunction(const View1D& x)
{
    const int D = x.extent(0);
    const double* ptr = &x(0);
    auto* cb = globalExternalObjective().fn;
    return cb ? cb(ptr, D) : 1e300;
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double computeBuiltInObjective(const View1D& x, int expenseFactor, ObjectiveType objType)
{
    switch (objType) {
        case ObjectiveType::Sphere:        return sphereObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Rastrigin:     return rastriginObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Ellipsoid:     return ellipsoidObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Rosenbrock:    return rosenbrockObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Ackley:        return ackleyObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Levy:          return levyObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Multimodal:    return multimodalObjectiveFunction(x, expenseFactor);
        case ObjectiveType::ShiftedSphere: return shiftedSphereObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Schwefel:      return schwefelObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Griewank:      return griewankObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Zakharov:      return zakharovObjectiveFunction(x, expenseFactor);
        case ObjectiveType::StyblinskiTang: return styblinskiTangObjectiveFunction(x, expenseFactor);
        case ObjectiveType::DixonPrice:    return dixonPriceObjectiveFunction(x, expenseFactor);
        case ObjectiveType::BentCigar:     return bentCigarObjectiveFunction(x, expenseFactor);
        case ObjectiveType::Discus:        return discusObjectiveFunction(x, expenseFactor);
        case ObjectiveType::SharpRidge:    return sharpRidgeObjectiveFunction(x, expenseFactor);
        case ObjectiveType::DifferentPowers: return differentPowersObjectiveFunction(x, expenseFactor);
        default:                           return sphereObjectiveFunction(x, expenseFactor);
    }
}

template<typename View1D>
KOKKOS_INLINE_FUNCTION
double computeObjective(const View1D& x, int expenseFactor, ObjectiveType objType)
{
    switch (objType) {
        case ObjectiveType::VQE:           return vqeObjectiveFunction(x);
        default:                           return computeBuiltInObjective(x, expenseFactor, objType);
    }
}

std::vector<int> serializeHypersphereID(const HypersphereID& id);
HypersphereID deserializeHypersphereID(const std::vector<int>& data, size_t& offset);

Kokkos::View<double*, mem_space>
reconstructCenter(const Hypersphere& hs,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initCenter,
                  double initRadius,
                  const std::vector<double>& alphaDepthHost);

Kokkos::View<double*, mem_space>
reconstructCenter(const Hypersphere& hs,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initCenter,
                  double initRadius,
                  const std::vector<double>& alphaDepthHost,
                  FBODecomposition decomposition);

Kokkos::View<double*, mem_space>
reconstructCenter(const Hypersphere& hs,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initCenter,
                  double initRadius);

Kokkos::View<double*, mem_space>
reconstructCenter(const Hypersphere& hs,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initCenter,
                  double initRadius,
                  FBODecomposition decomposition);

template <typename PoolT>
Kokkos::View<double**, mem_space>
generateRandomPointsAroundCenter(const Kokkos::View<double*, mem_space>& center,
                                 double radius,
                                 int dimension,
                                 PoolT& randPool,
                                 int numPoints = 8);

void prepareHypersphereData(const std::vector<Hypersphere>& hList,
                            Kokkos::View<int*, mem_space>& depths,
                            Kokkos::View<size_t*, mem_space>& idOffsets,
                            Kokkos::View<int*, mem_space>& idDims,
                            Kokkos::View<int*, mem_space>& idSigns);

void reconstructCentersAndRadii(size_t nHypers,
                                int dimension,
                                double initRadius,
                                const Kokkos::View<int*, mem_space>& depths,
                                const Kokkos::View<size_t*, mem_space>& idOffsets,
                                const Kokkos::View<int*, mem_space>& idDims,
                                const Kokkos::View<int*, mem_space>& idSigns,
                                const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initCenterDev,
                                const Kokkos::View<double*, mem_space>& alphaDepthDev,
                                Kokkos::View<double**, mem_space>& centers,
                                Kokkos::View<double*, mem_space>& radii,
                                FBODecomposition decomposition = FBODecomposition::Hypersphere);

void reconstructCentersAndRadii(size_t nHypers,
                                int dimension,
                                double initRadius,
                                const Kokkos::View<int*, mem_space>& depths,
                                const Kokkos::View<size_t*, mem_space>& idOffsets,
                                const Kokkos::View<int*, mem_space>& idDims,
                                const Kokkos::View<int*, mem_space>& idSigns,
                                const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initCenterDev,
                                Kokkos::View<double**, mem_space>& centers,
                                Kokkos::View<double*, mem_space>& radii);

void scoreHyperspheres(size_t nHypers,
                       int dimension,
                       const Kokkos::View<double**, mem_space>& centers,
                       const Kokkos::View<double*, mem_space>& radii,
                       const Kokkos::View<double*, mem_space>& bestSolutionDev,
                       double bestObjVal,
                       Kokkos::View<double*, mem_space>& scores,
                       int expenseFactor,
                       ObjectiveType objType);

double computeObjectiveValue(const Kokkos::View<double*, mem_space>& solution,
                             int expenseFactor,
                             ObjectiveType objType);

double computeEllipsoidObjectiveValue(const Kokkos::View<double*, mem_space>& solution,
                                      int expenseFactor);

Kokkos::View<double*, mem_space>
intensiveLocalSearch(const Hypersphere& hs,
                     const Kokkos::View<double*, mem_space>& initCenter,
                     double initRadius,
                     int maxIters,
                     double stepSize,
                     double phi,
                     double omegaMin,
                     ObjectiveType objType,
                     int expenseFactor);

void intensiveLocalSearchBatch(const Kokkos::View<double**, mem_space>& initCenters,
                               int maxIters,
                               double stepSize,
                               double phi,
                               double omegaMin,
                               ObjectiveType objType,
                               int expenseFactor,
                               Kokkos::View<double*, mem_space>& outBestVals,
                               Kokkos::View<double**, mem_space> outFinalPos);

std::vector<Hypersphere> decomposeHypersphere(const Hypersphere& parent);
std::vector<Hypersphere> decomposeHypersphere(const Hypersphere& parent,
                                              FBODecomposition decomposition);
std::vector<int> selectBestIndices(const std::vector<double>& scores, int beta);
size_t get_available_memory();

int parseBeta(int argc, char* argv[], int wrank);
int parseDimension(int argc, char* argv[], int wrank);
int parseBudget(int argc, char* argv[], int rank);
int parseExpenseFactor(int argc, char* argv[], int rank);
ObjectiveType parseObjectiveType(int argc, char* argv[], int idx, int wrank);
double parseRho(int argc, char** argv, int rank);

inline void external_eval_batch_host(const double* X, int D, int B, double* out)
{
    auto& ext = globalExternalObjective();
    if (ext.fn_many) {
        ext.fn_many(X, D, B, out);
        return;
    }
    for (int b = 0; b < B; ++b) {
        const double* row = X + size_t(b) * size_t(D);
        out[b] = ext.fn ? ext.fn(row, D) : 1e300;
    }
}

#endif // HPC_FUNCTIONS_HPP
