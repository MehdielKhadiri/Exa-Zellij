#include "HPCfunctions.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

#if defined(KOKKOS_ENABLE_CUDA)
  #include <cuda_runtime.h>
#elif defined(KOKKOS_ENABLE_HIP)
  #include <hip/hip_runtime.h>
#endif

namespace {

const ObjectiveInfo kObjectiveInfos[] = {
    {ObjectiveType::Sphere,
     "sphere",
     "smooth-convex-separable",
     -5.12,
     5.12,
     true,
     true,
     false,
     true},
    {ObjectiveType::Rastrigin,
     "rastrigin",
     "separable-multimodal",
     -5.12,
     5.12,
     true,
     true,
     false,
     true},
    {ObjectiveType::Ellipsoid,
     "ellipsoid",
     "ill-conditioned-convex",
     -5.12,
     5.12,
     true,
     true,
     false,
     true},
    {ObjectiveType::Rosenbrock,
     "rosenbrock",
     "curved-valley-nonseparable",
     -5.12,
     5.12,
     false,
     false,
     false,
     true},
    {ObjectiveType::Ackley,
     "ackley",
     "global-structure-multimodal",
     -5.12,
     5.12,
     false,
     false,
     false,
     true},
    {ObjectiveType::Levy,
     "levy",
     "multimodal-shifted-optimum",
     -5.12,
     5.12,
     true,
     true,
     false,
     true},
    {ObjectiveType::Multimodal,
     "multimodal",
     "separable-ackley-like",
     -5.12,
     5.12,
     true,
     true,
     false,
     true},
    {ObjectiveType::ShiftedSphere,
     "shifted-sphere",
     "smooth-convex-shifted",
     -5.12,
     5.12,
     true,
     true,
     false,
     true},
    {ObjectiveType::Schwefel,
     "schwefel",
     "deceptive-separable-multimodal",
     -500.0,
     500.0,
     true,
     true,
     false,
     true},
    {ObjectiveType::Griewank,
     "griewank",
     "weakly-coupled-multimodal",
     -600.0,
     600.0,
     false,
     false,
     false,
     true},
    {ObjectiveType::Zakharov,
     "zakharov",
     "smooth-nonseparable-polynomial",
     -10.0,
     10.0,
     false,
     false,
     false,
     true},
    {ObjectiveType::StyblinskiTang,
     "styblinski-tang",
     "separable-multimodal-negative-optimum",
     -5.0,
     5.0,
     true,
     true,
     false,
     true},
    {ObjectiveType::DixonPrice,
     "dixon-price",
     "chain-coupled-nonseparable",
     -10.0,
     10.0,
     false,
     false,
     false,
     true},
    {ObjectiveType::BentCigar,
     "bent-cigar",
     "ridge-ill-conditioned",
     -100.0,
     100.0,
     true,
     true,
     false,
     true},
    {ObjectiveType::Discus,
     "discus",
     "single-axis-ill-conditioned",
     -100.0,
     100.0,
     true,
     true,
     false,
     true},
    {ObjectiveType::SharpRidge,
     "sharp-ridge",
     "nonsmooth-ridge",
     -100.0,
     100.0,
     false,
     false,
     false,
     true},
    {ObjectiveType::DifferentPowers,
     "different-powers",
     "axis-dependent-conditioning",
     -100.0,
     100.0,
     false,
     false,
     false,
     true},
    {ObjectiveType::VQE,
     "vqe",
     "external-quantum-objective",
     -5.12,
     5.12,
     false,
     false,
     true,
     false}
};

} // namespace

std::vector<ObjectiveInfo> listObjectiveInfos()
{
    return std::vector<ObjectiveInfo>(
        std::begin(kObjectiveInfos),
        std::end(kObjectiveInfos));
}

const ObjectiveInfo& objectiveInfo(ObjectiveType objType)
{
    for (const auto& info : kObjectiveInfos) {
        if (info.type == objType) return info;
    }
    return kObjectiveInfos[0];
}

std::string objectiveCanonicalName(ObjectiveType objType)
{
    return objectiveInfo(objType).name;
}

std::string availableObjectiveNames()
{
    std::string names;
    for (size_t i = 0; i < std::size(kObjectiveInfos); ++i) {
        if (i > 0) names += ", ";
        names += kObjectiveInfos[i].name;
    }
    return names;
}

double objectiveDefaultLowerBound(ObjectiveType objType)
{
    return objectiveInfo(objType).defaultLower;
}

double objectiveDefaultUpperBound(ObjectiveType objType)
{
    return objectiveInfo(objType).defaultUpper;
}

bool objectiveHasKnownOptimum(ObjectiveType objType)
{
    return objectiveInfo(objType).hasKnownOptimum;
}

double objectiveReferenceCoordinate(ObjectiveType objType,
                                    int dimIndex,
                                    int dimension)
{
    (void)dimension;

    switch (objType) {
        case ObjectiveType::Rosenbrock:
        case ObjectiveType::Levy:
        case ObjectiveType::ShiftedSphere:
            return 1.0;
        case ObjectiveType::Schwefel:
            return 420.9687462275036;
        case ObjectiveType::StyblinskiTang:
            return -2.903534027771177;
        case ObjectiveType::DixonPrice: {
            const int oneBased = dimIndex + 1;
            const double exponent =
                -1.0 + std::pow(2.0, 1.0 - static_cast<double>(oneBased));
            return std::pow(2.0, exponent);
        }
        default:
            return 0.0;
    }
}

double objectiveKnownMinimumValue(ObjectiveType objType, int dimension)
{
    if (!objectiveHasKnownOptimum(objType)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    switch (objType) {
        case ObjectiveType::ShiftedSphere:
            return 5.0;
        case ObjectiveType::StyblinskiTang: {
            const double x = -2.903534027771177;
            const double x2 = x * x;
            const double f1 = 0.5 * (x2 * x2 - 16.0 * x2 + 5.0 * x);
            return f1 * static_cast<double>(dimension);
        }
        default:
            return 0.0;
    }
}

void printObjectiveInfo(ObjectiveType objType, int dimension, std::ostream& os)
{
    const ObjectiveInfo& info = objectiveInfo(objType);
    os << "[Objective]\n"
       << "  name          = " << info.name << "\n"
       << "  regime        = " << info.regime << "\n"
       << "  separable     = " << (info.separable ? "yes" : "no") << "\n"
       << "  additive path = " << (info.additiveFastPath ? "yes" : "no") << "\n"
       << "  external      = " << (info.external ? "yes" : "no") << "\n"
       << "  bounds        = [" << info.defaultLower << ", "
       << info.defaultUpper << "]\n";

    if (info.hasKnownOptimum) {
        os << "  known f*      = "
           << objectiveKnownMinimumValue(objType, dimension) << "\n";
    } else {
        os << "  known f*      = unavailable\n";
    }
}

std::vector<int> serializeHypersphereID(const HypersphereID& id)
{
    std::vector<int> s;
    s.reserve(1 + 2 * id.size());
    s.push_back(static_cast<int>(id.size()));
    for (const auto& p : id) {
        s.push_back(p.first);
        s.push_back(p.second);
    }
    return s;
}

HypersphereID deserializeHypersphereID(const std::vector<int>& data, size_t& offset)
{
    if (offset >= data.size()) {
        throw std::runtime_error("deserializeHypersphereID: offset out of range");
    }

    const int idSize = data[offset++];
    if (idSize < 0 || offset + size_t(2 * idSize) > data.size()) {
        throw std::runtime_error("deserializeHypersphereID: malformed serialized ID");
    }

    HypersphereID id;
    id.reserve(idSize);
    for (int i = 0; i < idSize; ++i) {
        const int dim  = data[offset++];
        const int sign = data[offset++];
        id.emplace_back(dim, sign);
    }
    return id;
}

Kokkos::View<double*, mem_space>
reconstructCenter(const Hypersphere& hs,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initCenter,
                  double initRadius,
                  const std::vector<double>& alphaDepthHost)
{
    return reconstructCenter(hs,
                             initCenter,
                             initRadius,
                             alphaDepthHost,
                             FBODecomposition::Hypersphere);
}

Kokkos::View<double*, mem_space>
reconstructCenter(const Hypersphere& hs,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initCenter,
                  double initRadius,
                  const std::vector<double>& alphaDepthHost,
                  FBODecomposition decomposition)
{
    const int D = hs.dimension;
    const int depth = hs.depth;

    if (D < 0 || depth < 0 || int(hs.id.size()) < depth) {
        throw std::runtime_error("reconstructCenter: invalid hypersphere metadata");
    }

    Kokkos::View<double*, Kokkos::HostSpace> hostC("hostC", D);
    auto initCenterHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), initCenter);
    Kokkos::deep_copy(hostC, initCenterHost);

    if (fboUsesIndependentIntervalWidths(decomposition)) {
        std::vector<double> halfWidths(static_cast<size_t>(D), initRadius);
        const double factor = fboRadiusFactor(decomposition);

        for (int d = 0; d < depth; ++d) {
            const int dim = hs.id[size_t(d)].first;
            const int code = hs.id[size_t(d)].second;
            if (dim < 0 || dim >= D) {
                throw std::runtime_error("reconstructCenter: split dimension out of range");
            }
            const double oldHalf = halfWidths[static_cast<size_t>(dim)];
            hostC(dim) += fboChildOffset(oldHalf, decomposition, code);
            halfWidths[static_cast<size_t>(dim)] = oldHalf * factor;
        }
    } else {
        double r = initRadius;
        const double alpha0 = fboRadiusFactor(decomposition);

        for (int d = 0; d < depth; ++d) {
            const double a = (decomposition == FBODecomposition::Hypersphere &&
                              d < int(alphaDepthHost.size()))
                ? alphaDepthHost[size_t(d)]
                : alpha0;
            const double rPrime = a * r;
            const int dim = hs.id[size_t(d)].first;
            const int code = hs.id[size_t(d)].second;
            const double offset = code == 0
                ? 0.0
                : double(code > 0 ? 1 : -1) * (r - rPrime);
            if (code != 0 && (dim < 0 || dim >= D)) {
                throw std::runtime_error("reconstructCenter: split dimension out of range");
            }
            if (dim >= 0 && dim < D) hostC(dim) += offset;
            r = rPrime;
        }
    }

    Kokkos::View<double*, mem_space> c("reconstructed_center", D);
    Kokkos::deep_copy(c, hostC);
    return c;
}

Kokkos::View<double*, mem_space>
reconstructCenter(const Hypersphere& hs,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initCenter,
                  double initRadius)
{
    static const std::vector<double> a0 = {1.0 / (1.0 + std::sqrt(2.0))};
    return reconstructCenter(hs, initCenter, initRadius, a0);
}

Kokkos::View<double*, mem_space>
reconstructCenter(const Hypersphere& hs,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initCenter,
                  double initRadius,
                  FBODecomposition decomposition)
{
    static const std::vector<double> a0 = {1.0 / (1.0 + std::sqrt(2.0))};
    return reconstructCenter(hs, initCenter, initRadius, a0, decomposition);
}

template <typename PoolT>
Kokkos::View<double**, mem_space>
generateRandomPointsAroundCenter(const Kokkos::View<double*, mem_space>& center,
                                 double radius,
                                 int dimension,
                                 PoolT& randPool,
                                 int numPoints)
{
    Kokkos::View<double**, mem_space> points("points", numPoints, dimension);
    using policy = Kokkos::TeamPolicy<exec_space>;

    Kokkos::parallel_for("GenRandPoints", policy(numPoints, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename policy::member_type& team) {
            const int i = team.league_rank();
            auto state = randPool.get_state();

            double norm2 = 0.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, dimension),
                [&](int d, double& s) {
                    const double u = 2.0 * Kokkos::rand<typename PoolT::generator_type, double>::draw(state) - 1.0;
                    points(i, d) = u;
                    s += u * u;
                }, norm2);

            const double inv_norm = rsqrt(norm2 + 1e-30);
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dimension),
                [&](int d) { points(i, d) *= inv_norm; });
            team.team_barrier();

            const double U = Kokkos::rand<typename PoolT::generator_type, double>::draw(state);
            const double scale = radius * pow(U + 1e-16, 1.0 / double(dimension));

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dimension),
                [&](int d) { points(i, d) = center(d) + scale * points(i, d); });
            team.team_barrier();

            randPool.free_state(state);
        });

    return points;
}

void prepareHypersphereData(const std::vector<Hypersphere>& hList,
                            Kokkos::View<int*, mem_space>& depths,
                            Kokkos::View<size_t*, mem_space>& idOffsets,
                            Kokkos::View<int*, mem_space>& idDims,
                            Kokkos::View<int*, mem_space>& idSigns)
{
    const size_t n = hList.size();
    depths = Kokkos::View<int*, mem_space>("depths", n);
    idOffsets = Kokkos::View<size_t*, mem_space>("idOffsets", n + 1);

    size_t totalIDSize = 0;
    for (const auto& hs : hList) totalIDSize += hs.id.size();

    idDims = Kokkos::View<int*, mem_space>("idDims", totalIDSize);
    idSigns = Kokkos::View<int*, mem_space>("idSigns", totalIDSize);

    auto dHost = Kokkos::create_mirror_view(depths);
    auto offHost = Kokkos::create_mirror_view(idOffsets);
    auto dimHost = Kokkos::create_mirror_view(idDims);
    auto signHost = Kokkos::create_mirror_view(idSigns);

    size_t currentOff = 0;
    for (size_t i = 0; i < n; ++i) {
        dHost(i) = hList[i].depth;
        offHost(i) = currentOff;
        for (const auto& p : hList[i].id) {
            dimHost(currentOff) = p.first;
            signHost(currentOff) = p.second;
            ++currentOff;
        }
    }
    offHost(n) = currentOff;

    Kokkos::deep_copy(depths, dHost);
    Kokkos::deep_copy(idOffsets, offHost);
    Kokkos::deep_copy(idDims, dimHost);
    Kokkos::deep_copy(idSigns, signHost);
}

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
                                FBODecomposition decomposition)
{
    centers = Kokkos::View<double**, mem_space>("centers", nHypers, dimension);
    radii = Kokkos::View<double*, mem_space>("radii", nHypers);

    using MD2 = Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_space>;

    Kokkos::parallel_for("RC_copy_init",
        MD2({0, 0}, {static_cast<int>(nHypers), dimension}),
        KOKKOS_LAMBDA(int i, int d) {
            centers(i, d) = initCenterDev(d);
        });

    Kokkos::parallel_for("RC_apply_moves",
        Kokkos::RangePolicy<exec_space>(0, static_cast<int>(nHypers)),
        KOKKOS_LAMBDA(int i) {
            const int depth = depths(i);
            const size_t st = idOffsets(i);
            double r = initRadius;
            const double alpha0 = fboRadiusFactor(decomposition);

            for (int dd = 0; dd < depth; ++dd) {
                const int dim = idDims(st + dd);
                const double off =
                    fboPathMoveOffset(initRadius,
                                      decomposition,
                                      idDims,
                                      idSigns,
                                      st,
                                      dd,
                                      alphaDepthDev);
                if (dim >= 0 && dim < dimension) centers(i, dim) += off;
                if (!fboUsesIndependentIntervalWidths(decomposition)) {
                    const double a =
                        (decomposition == FBODecomposition::Hypersphere &&
                         dd < static_cast<int>(alphaDepthDev.extent(0)))
                            ? alphaDepthDev(dd)
                            : alpha0;
                    r *= a;
                }
            }
            radii(i) = fboUsesIndependentIntervalWidths(decomposition)
                ? initRadius
                : r;
        });
}

void reconstructCentersAndRadii(size_t nHypers,
                                int dimension,
                                double initRadius,
                                const Kokkos::View<int*, mem_space>& depths,
                                const Kokkos::View<size_t*, mem_space>& idOffsets,
                                const Kokkos::View<int*, mem_space>& idDims,
                                const Kokkos::View<int*, mem_space>& idSigns,
                                const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initCenterDev,
                                Kokkos::View<double**, mem_space>& centers,
                                Kokkos::View<double*, mem_space>& radii)
{
    Kokkos::View<double*, mem_space> alphaDepthDev("alphaDepthDev_const", 1);
    auto hA = Kokkos::create_mirror_view(alphaDepthDev);
    hA(0) = 1.0 / (1.0 + std::sqrt(2.0));
    Kokkos::deep_copy(alphaDepthDev, hA);

    reconstructCentersAndRadii(nHypers, dimension, initRadius,
                               depths, idOffsets, idDims, idSigns,
                               initCenterDev, alphaDepthDev,
                               centers, radii,
                               FBODecomposition::Hypersphere);
}

void scoreHyperspheres(size_t nHypers,
                       int dimension,
                       const Kokkos::View<double**, mem_space>& centers,
                       const Kokkos::View<double*, mem_space>& radii,
                       const Kokkos::View<double*, mem_space>& bestSolutionDev,
                       double bestObjVal,
                       Kokkos::View<double*, mem_space>& scores,
                       int expenseFactor,
                       ObjectiveType objType)
{
    Kokkos::deep_copy(scores, 0.0);

    if (usesAdditiveDeviceObjective(objType)) {
        using policy = Kokkos::TeamPolicy<exec_space>;
        const int numPoints = 3;

        Kokkos::parallel_for("ScoreHypers_device", policy(static_cast<int>(nHypers), Kokkos::AUTO),
            KOKKOS_LAMBDA(const typename policy::member_type& team) {
                const int i = team.league_rank();
                const double r = radii(i);
                double totalScore = 0.0;

                for (int p = 0; p < numPoints; ++p) {
                    double termSum = 0.0;
                    double dist = 0.0;

                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, dimension),
                        [&](int d, double& loc) {
                            const double xi = centers(i, d) + 0.5 * r;
                            loc += objectiveTerm(objType, xi, d, dimension, expenseFactor);
                        }, termSum);
                    const double fx = objectiveConstant(objType) + termSum;

                    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, dimension),
                        [&](int d, double& loc) {
                            const double xi = centers(i, d) + 0.5 * r;
                            const double diff = xi - bestSolutionDev(d);
                            loc += diff * diff;
                        }, dist);

                    dist = sqrt(dist);
                    if (dist == 0.0) dist = 1e-10;
                    totalScore += (fx - bestObjVal) / dist;
                }

                scores(i) = totalScore / double(numPoints);
            });
        return;
    }

    auto centers_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), centers);
    auto radii_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), radii);
    auto best_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bestSolutionDev);

    const int H = static_cast<int>(nHypers);
    std::vector<double> X(size_t(H) * size_t(dimension));
    for (int i = 0; i < H; ++i) {
        const double off = 0.5 * radii_h(i);
        double* row = X.data() + size_t(i) * size_t(dimension);
        for (int d = 0; d < dimension; ++d) row[d] = centers_h(i, d) + off;
    }

    std::vector<double> fx(H);
    if (usesExternalObjective(objType)) {
        external_eval_batch_host(X.data(), dimension, H, fx.data());
    } else {
        for (int i = 0; i < H; ++i) {
            Kokkos::View<double*, Kokkos::HostSpace> xh("score_obj_host_x", dimension);
            const double* row = X.data() + size_t(i) * size_t(dimension);
            for (int d = 0; d < dimension; ++d) xh(d) = row[d];
            fx[i] = computeBuiltInObjective(xh, expenseFactor, objType);
        }
    }

    Kokkos::View<double*, Kokkos::HostSpace> scores_h("scores_ext_host", H);
    for (int i = 0; i < H; ++i) {
        const double* row = X.data() + size_t(i) * size_t(dimension);
        double dist2 = 0.0;
        for (int d = 0; d < dimension; ++d) {
            const double diff = row[d] - best_h(d);
            dist2 += diff * diff;
        }
        double dist = std::sqrt(dist2);
        if (dist == 0.0) dist = 1e-10;
        scores_h(i) = (fx[i] - bestObjVal) / dist;
    }
    Kokkos::deep_copy(scores, scores_h);
}

Kokkos::View<double*, mem_space>
intensiveLocalSearch(const Hypersphere& hs,
                     const Kokkos::View<double*, mem_space>& initCenter,
                     double initRadius,
                     int maxIters,
                     double stepSize,
                     double phi,
                     double omegaMin,
                     ObjectiveType objType,
                     int expenseFactor)
{
    auto c_dev = reconstructCenter(hs, initCenter, initRadius);
    const int D = hs.dimension;

    if (!usesAdditiveDeviceObjective(objType)) {
        const double sigma = initRadius;
        double omega = std::min(stepSize, 0.75 * sigma);
        const double wmin = std::max(omegaMin, 1e-6 * sigma);
        phi = std::max(0.0, std::min(phi, 0.999));

        auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), c_dev);
        Kokkos::View<double*, Kokkos::HostSpace> y_host("ils_ext_y", D);

        auto clamp_box = [&](double v) -> double {
            if (v < -sigma) return -sigma;
            if (v >  sigma) return  sigma;
            return v;
        };

        auto eval_host_point = [&](const Kokkos::View<double*, Kokkos::HostSpace>& hvec) -> double {
            Kokkos::View<double*, mem_space> dvec("ils_ext_tmp", D);
            Kokkos::deep_copy(dvec, hvec);
            return computeObjectiveValue(dvec, expenseFactor, objType);
        };

        for (int d = 0; d < D; ++d) x_host(d) = clamp_box(x_host(d));
        double f_best = eval_host_point(x_host);

        int it = 0;
        while (it < maxIters && omega >= wmin) {
            bool improved = false;
            for (int d = 0; d < D; ++d) {
                for (int i = 0; i < D; ++i) y_host(i) = x_host(i);
                y_host(d) = clamp_box(y_host(d) + omega);
                const double f_p = eval_host_point(y_host);
                if (f_p < f_best) {
                    for (int i = 0; i < D; ++i) x_host(i) = y_host(i);
                    f_best = f_p;
                    improved = true;
                    continue;
                }

                for (int i = 0; i < D; ++i) y_host(i) = x_host(i);
                y_host(d) = clamp_box(y_host(d) - omega);
                const double f_m = eval_host_point(y_host);
                if (f_m < f_best) {
                    for (int i = 0; i < D; ++i) x_host(i) = y_host(i);
                    f_best = f_m;
                    improved = true;
                }
            }
            if (!improved) omega *= phi;
            ++it;
        }

        Kokkos::View<double*, mem_space> x_dev("localsearch_ext_x", D);
        Kokkos::deep_copy(x_dev, x_host);
        return x_dev;
    }

    Kokkos::View<double*, mem_space> bestSol("bestSol", D);
    Kokkos::deep_copy(bestSol, c_dev);

    using policy = Kokkos::TeamPolicy<exec_space>;
    Kokkos::View<double*, mem_space> localB("localB", D);
    Kokkos::View<double*, mem_space> newVals("newVals", D);
    Kokkos::View<double*, mem_space> termD("termD", D);
    Kokkos::View<bool*, mem_space> improved("improved", 1);

    Kokkos::parallel_for("ILS", policy(1, Kokkos::AUTO),
        KOKKOS_LAMBDA(const typename policy::member_type& team) {
            auto partial = [&](double x, int dim) -> double {
                return objectiveTerm(objType, x, dim, D, expenseFactor);
            };

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, D),
                [&](int d) { localB(d) = bestSol(d); });
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, D),
                [&](int d) { termD(d) = partial(localB(d), d); });
            team.team_barrier();

            double bestVal = 0.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, D),
                [&](int d, double& s) { s += termD(d); }, bestVal);
            team.team_barrier();

            double omega = stepSize;
            for (int iter = 0; iter < maxIters; ++iter) {
                if (team.team_rank() == 0) improved(0) = false;
                team.team_barrier();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, D),
                    [&](int d) {
                        const double x = localB(d);
                        const double xp = x + omega;
                        const double xm = x - omega;
                        const double oldP = termD(d);
                        const double p1 = partial(xp, d);
                        const double p2 = partial(xm, d);

                        double bestP = oldP;
                        double bestX = x;
                        if (p1 < bestP) { bestP = p1; bestX = xp; improved(0) = true; }
                        if (p2 < bestP) { bestP = p2; bestX = xm; improved(0) = true; }

                        newVals(d) = bestX;
                        termD(d) = bestP;
                    });
                team.team_barrier();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, D),
                    [&](int d) { localB(d) = newVals(d); });
                team.team_barrier();

                double newVal = 0.0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, D),
                    [&](int d, double& s) { s += termD(d); }, newVal);
                bestVal = newVal;
                team.team_barrier();

                if (team.team_rank() == 0 && !improved(0)) {
                    omega *= phi;
                    if (omega < omegaMin) iter = maxIters;
                }
                team.team_barrier();
            }

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, D),
                [&](int d) { bestSol(d) = localB(d); });
        });

    return bestSol;
}

std::vector<Hypersphere> decomposeHypersphere(const Hypersphere& parent)
{
    return decomposeHypersphere(parent, FBODecomposition::Hypersphere);
}

std::vector<Hypersphere> decomposeHypersphere(const Hypersphere& parent,
                                              FBODecomposition decomposition)
{
    const int D = parent.dimension;
    const int depth = parent.depth;
    const int nChildren = fboChildrenPerParent(D, decomposition);

    std::vector<Hypersphere> result;
    result.reserve(nChildren);

    for (int child = 0; child < nChildren; ++child) {
        HypersphereID cid = parent.id;

        switch (decomposition) {
            case FBODecomposition::Hypercube: {
                const int active = fboActiveHypercubeDims(D, 12);
                const int start = D > 0 ? depth % D : 0;
                for (int bit = 0; bit < active; ++bit) {
                    const int dim = (start + bit) % D;
                    const int code = ((child >> bit) & 1) ? +1 : -1;
                    cid.emplace_back(dim, code);
                }
                break;
            }
            case FBODecomposition::Bisection: {
                const int dim = D > 0 ? depth % D : 0;
                cid.emplace_back(dim, child == 0 ? -1 : +1);
                break;
            }
            case FBODecomposition::Trisection: {
                const int dim = D > 0 ? depth % D : 0;
                const int code = child == 0 ? -1 : (child == 1 ? 0 : +1);
                cid.emplace_back(dim, code);
                break;
            }
            case FBODecomposition::Voronoi:
                if (child == 0) {
                    cid.emplace_back(0, 0);
                } else {
                    const int q = child - 1;
                    cid.emplace_back((q / 2) % D, (q & 1) ? +1 : -1);
                }
                break;
            case FBODecomposition::Simplex:
                if (child == 0) {
                    cid.emplace_back(0, 0);
                } else {
                    cid.emplace_back((child - 1) % D, +1);
                }
                break;
            case FBODecomposition::Hypersphere:
            default:
                cid.emplace_back(child / 2, (child & 1) ? +1 : -1);
                break;
        }

        result.emplace_back(D, static_cast<int>(cid.size()), std::move(cid));
    }
    return result;
}

std::vector<int> selectBestIndices(const std::vector<double>& scores, int beta)
{
    std::vector<int> inds(scores.size());
    std::iota(inds.begin(), inds.end(), 0);
    beta = std::min(beta, static_cast<int>(scores.size()));
    std::partial_sort(inds.begin(), inds.begin() + beta, inds.end(),
        [&](int a, int b) { return scores[a] < scores[b]; });
    inds.resize(beta);
    return inds;
}

size_t get_available_memory()
{
#if defined(KOKKOS_ENABLE_CUDA)
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return free_b;
#elif defined(KOKKOS_ENABLE_HIP)
    size_t free_b = 0, total_b = 0;
    hipDeviceSynchronize();
    if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return 0;
    return free_b;
#else
    return 0;
#endif
}

int parseBeta(int argc, char* argv[], int wrank)
{
    int beta = 50;
    if (argc > 1) {
        try {
            beta = std::stoi(argv[1]);
            if (beta <= 0) {
                if (wrank == 0) std::cerr << "Invalid beta. using 50.\n";
                beta = 50;
            }
        } catch (...) {
            if (wrank == 0) std::cerr << "Invalid beta arg. using 50\n";
        }
    }
    return beta;
}

int parseDimension(int argc, char* argv[], int wrank)
{
    int dim = 20000;
    if (argc > 2) {
        try {
            dim = std::stoi(argv[2]);
        } catch (...) {
            if (wrank == 0) std::cerr << "Invalid dimension arg. using 20000.\n";
        }
    }
    return dim;
}

int parseBudget(int argc, char* argv[], int rank)
{
    if (argc < 4) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0]
                      << " <beta> <dimension> <budget> [expenseFactor] [objective]\n";
        }
        MPI_Finalize();
        std::exit(EXIT_FAILURE);
    }
    return std::stoi(argv[3]);
}

int parseExpenseFactor(int argc, char* argv[], int rank)
{
    int expenseFactor = 1;
    if (argc > 4) {
        try {
            expenseFactor = std::stoi(argv[4]);
            if (expenseFactor < 0) expenseFactor = 1;
        } catch (...) {
            if (rank == 0) std::cerr << "Invalid expense factor. Using default=1.\n";
        }
    }
    return expenseFactor;
}

ObjectiveType parseObjectiveType(int argc, char* argv[], int idx, int wrank)
{
    std::string raw;
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--objective") {
            raw = argv[i + 1];
            break;
        }
    }

    if (raw.empty()) {
        if (argc <= idx) return ObjectiveType::Sphere;
        raw = argv[idx];
    }

    std::string s(raw);
    for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));

    if (s == "sphere") return ObjectiveType::Sphere;
    if (s == "rastrigin") return ObjectiveType::Rastrigin;
    if (s == "ellipsoid") return ObjectiveType::Ellipsoid;
    if (s == "rosenbrock") return ObjectiveType::Rosenbrock;
    if (s == "ackley") return ObjectiveType::Ackley;
    if (s == "levy") return ObjectiveType::Levy;
    if (s == "vqe") return ObjectiveType::VQE;
    if (s == "shiftedsphere" || s == "shifted_sphere" ||
        s == "shifted-sphere") {
        return ObjectiveType::ShiftedSphere;
    }
    if (s == "multimodal") return ObjectiveType::Multimodal;
    if (s == "schwefel") return ObjectiveType::Schwefel;
    if (s == "griewank") return ObjectiveType::Griewank;
    if (s == "zakharov") return ObjectiveType::Zakharov;
    if (s == "styblinski" || s == "styblinski-tang" ||
        s == "styblinski_tang") {
        return ObjectiveType::StyblinskiTang;
    }
    if (s == "dixon-price" || s == "dixon_price" || s == "dixonprice") {
        return ObjectiveType::DixonPrice;
    }
    if (s == "bent-cigar" || s == "bent_cigar" || s == "bentcigar") {
        return ObjectiveType::BentCigar;
    }
    if (s == "discus") return ObjectiveType::Discus;
    if (s == "sharp-ridge" || s == "sharp_ridge" || s == "sharpridge") {
        return ObjectiveType::SharpRidge;
    }
    if (s == "different-powers" || s == "different_powers" ||
        s == "differentpowers") {
        return ObjectiveType::DifferentPowers;
    }

    if (wrank == 0) {
        std::cerr << "Unknown objective \"" << raw
                  << "\" - falling back to Sphere.\n"
                  << "Available objectives: "
                  << availableObjectiveNames() << "\n";
    }
    return ObjectiveType::Sphere;
}

double parseRho(int argc, char** argv, int rank)
{
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--rho") return std::stod(argv[i + 1]);
    }
    if (rank == 0) std::cout << "[Info] rho not specified; defaulting to 0.8\n";
    return 0.8;
}

double computeEllipsoidObjectiveValue(const Kokkos::View<double*, mem_space>& solution, int expenseFactor)
{
    auto h_sol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution);
    const int D = static_cast<int>(h_sol.extent(0));
    double sum = 0.0;
    for (int i = 0; i < D; ++i) sum += computeEllipsoidTerm(h_sol(i), i, D, expenseFactor);
    return sum;
}

double computeObjectiveValue(const Kokkos::View<double*, mem_space>& solution,
                             int expenseFactor,
                             ObjectiveType objType)
{
    auto hostSol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution);
    if (usesExternalObjective(objType)) {
        std::vector<double> x(static_cast<size_t>(hostSol.extent(0)));
        for (size_t i = 0; i < x.size(); ++i) x[i] = hostSol(i);
        double y = 1e300;
        external_eval_batch_host(x.data(), static_cast<int>(x.size()), 1, &y);
        return y;
    }
    return computeBuiltInObjective(hostSol, expenseFactor, objType);
}

void intensiveLocalSearchBatch(const Kokkos::View<double**, mem_space>& initCenters,
                               int maxIters,
                               double stepSize,
                               double phi,
                               double omegaMin,
                               ObjectiveType objType,
                               int expenseFactor,
                               Kokkos::View<double*, mem_space>& outBestVals,
                               Kokkos::View<double**, mem_space> outFinalPos)
{
    const int H = static_cast<int>(initCenters.extent(0));
    const int D = static_cast<int>(initCenters.extent(1));
    if (H == 0) return;

    Kokkos::View<double**, mem_space> localB("ils_localB", H, D);
    Kokkos::View<double**, mem_space> newVal("ils_newVal", H, D);
    Kokkos::View<double**, mem_space> termD("ils_termD", H, D);

    if (!usesAdditiveDeviceObjective(objType)) {
        auto centers_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), initCenters);
        std::vector<double> X_flat(size_t(H) * size_t(D));
        for (int i = 0; i < H; ++i) {
            for (int d = 0; d < D; ++d) X_flat[size_t(i) * size_t(D) + size_t(d)] = centers_h(i, d);
        }

        std::vector<double> vals(H);
        if (usesExternalObjective(objType)) {
            external_eval_batch_host(X_flat.data(), D, H, vals.data());
        } else {
            for (int i = 0; i < H; ++i) {
                Kokkos::View<double*, Kokkos::HostSpace> xh("ils_batch_obj_host_x", D);
                for (int d = 0; d < D; ++d) {
                    xh(d) = X_flat[size_t(i) * size_t(D) + size_t(d)];
                }
                vals[i] = computeBuiltInObjective(xh, expenseFactor, objType);
            }
        }

        auto h_bestVals = Kokkos::create_mirror_view(outBestVals);
        for (int i = 0; i < H; ++i) h_bestVals(i) = vals[i];
        Kokkos::deep_copy(outBestVals, h_bestVals);

        if (outFinalPos.data() != nullptr) Kokkos::deep_copy(outFinalPos, initCenters);
        return;
    }

    using policy = Kokkos::TeamPolicy<exec_space>;
    Kokkos::parallel_for("ILS_batch", policy(H, 128, 1),
        KOKKOS_LAMBDA(const typename policy::member_type& team) {
            const int i = team.league_rank();
            const double c0 = objectiveConstant(objType);

            auto partial = [&](double x, int dim) -> double {
                return objectiveTerm(objType, x, dim, D, expenseFactor);
            };

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, D),
                [&](int d) { localB(i, d) = initCenters(i, d); });
            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, D),
                [&](int d) { termD(i, d) = partial(localB(i, d), d); });
            team.team_barrier();

            double bestVal = 0.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, D),
                [&](int d, double& s) { s += termD(i, d); }, bestVal);
            team.team_barrier();

            double omega = stepSize;
            for (int it = 0; it < maxIters; ++it) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, D),
                    [&](int d) {
                        const double x = localB(i, d);
                        const double xp = x + omega;
                        const double xm = x - omega;
                        const double cur = termD(i, d);
                        const double p1 = partial(xp, d);
                        const double p2 = partial(xm, d);

                        double bestP = cur;
                        double bestX = x;
                        if (p1 < bestP) { bestP = p1; bestX = xp; }
                        if (p2 < bestP) { bestP = p2; bestX = xm; }

                        newVal(i, d) = bestX;
                        termD(i, d) = bestP;
                    });
                team.team_barrier();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, D),
                    [&](int d) { localB(i, d) = newVal(i, d); });
                team.team_barrier();

                double curVal = 0.0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, D),
                    [&](int d, double& s) { s += termD(i, d); }, curVal);
                team.team_barrier();

                if (team.team_rank() == 0) {
                    if (curVal < bestVal) {
                        bestVal = curVal;
                    } else {
                        omega *= phi;
                        if (omega < omegaMin) it = maxIters;
                    }
                }
                team.team_barrier();
            }

            if (team.team_rank() == 0) outBestVals(i) = bestVal + c0;

            if (outFinalPos.data() != nullptr) {
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, D),
                    [&](int d) { outFinalPos(i, d) = localB(i, d); });
            }
        });
}

template Kokkos::View<double**, mem_space>
generateRandomPointsAroundCenter<Kokkos::Random_XorShift64_Pool<exec_space>>(
    const Kokkos::View<double*, mem_space>&,
    double,
    int,
    Kokkos::Random_XorShift64_Pool<exec_space>&,
    int);
