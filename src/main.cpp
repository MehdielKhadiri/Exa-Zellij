#include "FQOMpi.hpp"

#include <Kokkos_Core.hpp>
#include <Kokkos_Sort.hpp>
#include <KokkosBlas3_gemm.hpp>

#include "HPCfunctions.hpp"
#include "GPfunctions.hpp"
#include "FBOConfig.hpp"
#include "FBDConfig.hpp"
#include "FBOSurrogateModels.hpp"
#include "py_vqe_shim.hpp"
#include "FQOStdioObjective.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <stdexcept>
#include <type_traits>

static double medianHost(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(),
                     values.begin() + static_cast<std::ptrdiff_t>(middle),
                     values.end());
    const double upper = values[middle];
    if ((values.size() & 1u) != 0u) return upper;
    const double lower =
        *std::max_element(values.begin(),
                          values.begin() +
                              static_cast<std::ptrdiff_t>(middle));
    return 0.5 * (lower + upper);
}

#ifdef KOKKOS_ENABLE_HIP
  #include <hip/hip_runtime.h>
#endif

#ifdef KOKKOS_ENABLE_CUDA
  #include <cuda_runtime.h>
#endif

// ============================================================================
// Reduction helpers
// ============================================================================

struct MinAcqIndex {
    double value;
    int index;

    KOKKOS_INLINE_FUNCTION
    MinAcqIndex() : value(DBL_MAX), index(-1) {}

    KOKKOS_INLINE_FUNCTION
    MinAcqIndex(const double val, const int idx) : value(val), index(idx) {}

    KOKKOS_INLINE_FUNCTION
    bool operator<(const MinAcqIndex& rhs) const {
        return value < rhs.value;
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const MinAcqIndex& src) {
        if (src.value < value) {
            value = src.value;
            index = src.index;
        }
    }
};

namespace Kokkos {
template <>
struct reduction_identity<MinAcqIndex> {
    KOKKOS_FORCEINLINE_FUNCTION static MinAcqIndex max() {
        return MinAcqIndex(-DBL_MAX, -1);
    }

    KOKKOS_FORCEINLINE_FUNCTION static MinAcqIndex min() {
        return MinAcqIndex(DBL_MAX, -1);
    }
};
}

struct AcqIdx {
    double val;
    int idx;

    KOKKOS_INLINE_FUNCTION
    AcqIdx() : val(DBL_MAX), idx(-1) {}

    KOKKOS_INLINE_FUNCTION
    AcqIdx(double v, int i) : val(v), idx(i) {}

    KOKKOS_INLINE_FUNCTION
    bool operator<(const AcqIdx& rhs) const {
        if (val < rhs.val) return true;
        if (val > rhs.val) return false;
        return idx < rhs.idx;
    }
};

struct MaxValIndex {
    double value;
    int index;

    KOKKOS_INLINE_FUNCTION
    MaxValIndex() : value(-DBL_MAX), index(-1) {}

    KOKKOS_INLINE_FUNCTION
    MaxValIndex(double v, int i) : value(v), index(i) {}

    KOKKOS_INLINE_FUNCTION
    bool operator>(const MaxValIndex& rhs) const {
        if (value > rhs.value) return true;
        if (value < rhs.value) return false;
        return index >= 0 && rhs.index >= 0 ? (index < rhs.index) : (index > rhs.index);
    }

    KOKKOS_INLINE_FUNCTION
    bool operator<(const MaxValIndex& rhs) const {
        if (value < rhs.value) return true;
        if (value > rhs.value) return false;
        return index >= 0 && rhs.index >= 0 ? (index > rhs.index) : (index < rhs.index);
    }

    KOKKOS_INLINE_FUNCTION
    void operator+=(const MaxValIndex& rhs) {
        if (rhs.value > value) {
            value = rhs.value;
            index = rhs.index;
        }
    }
};

namespace Kokkos {
template <>
struct reduction_identity<MaxValIndex> {
    KOKKOS_FORCEINLINE_FUNCTION static MaxValIndex max() {
        return MaxValIndex(-DBL_MAX, -1);
    }

    KOKKOS_FORCEINLINE_FUNCTION static MaxValIndex min() {
        return MaxValIndex(DBL_MAX, -1);
    }
};
}

// ============================================================================
// Selection utilities
// ============================================================================

std::vector<int> selectCoverage(const std::vector<double>& ei, double rho) {
    const double total = std::accumulate(ei.begin(), ei.end(), 0.0);
    const double target = rho * total;

    std::vector<int> order(ei.size());
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return ei[a] > ei[b];
    });

    std::vector<int> chosen;
    double running = 0.0;

    for (int idx : order) {
        running += ei[idx];
        chosen.push_back(idx);
        if (running >= target) break;
    }

    return chosen;
}

static std::vector<int> selectCoverageGPU(const std::vector<double>& h_ei,
                                          double rho,
                                          int beta) {
    using ExecSpace = Kokkos::DefaultExecutionSpace;
    using MemSpace  = typename ExecSpace::memory_space;

    const int N = static_cast<int>(h_ei.size());
    std::vector<int> out;

    if (N == 0 || beta <= 0) return out;

    Kokkos::View<double*, MemSpace> d_ei("eiSphere_dev", N);
    Kokkos::View<unsigned char*, MemSpace> d_taken("ei_taken", N);

    {
        auto h = Kokkos::create_mirror_view(d_ei);
        for (int i = 0; i < N; ++i) h(i) = h_ei[i];
        Kokkos::deep_copy(d_ei, h);
    }

    Kokkos::deep_copy(d_taken, static_cast<unsigned char>(0));

    double total = 0.0;
    Kokkos::parallel_reduce(
        "rho_local_total",
        Kokkos::RangePolicy<ExecSpace>(0, N),
        KOKKOS_LAMBDA(int i, double& upd) {
            upd += d_ei(i);
        },
        total);

    const double target = rho * total;

    out.reserve(std::min(beta, N));

    if (target <= 0.0) {
        const int k = std::min(beta, N);

        for (int pick = 0; pick < k; ++pick) {
            MaxValIndex best;

            Kokkos::parallel_reduce(
                "rho_topk_fallback",
                Kokkos::RangePolicy<ExecSpace>(0, N),
                KOKKOS_LAMBDA(int i, MaxValIndex& upd) {
                    if (!d_taken(i)) {
                        const double v = d_ei(i);
                        if (v > upd.value) {
                            upd.value = v;
                            upd.index = i;
                        }
                    }
                },
                Kokkos::Max<MaxValIndex>(best));

            if (best.index < 0) break;

            Kokkos::parallel_for(
                "rho_mark_fallback",
                Kokkos::RangePolicy<ExecSpace>(0, 1),
                KOKKOS_LAMBDA(int) {
                    d_taken(best.index) = 1;
                });

            Kokkos::fence();
            out.push_back(best.index);
        }

        if (out.empty() && N > 0) out.push_back(0);
        return out;
    }

    double covered = 0.0;

    for (int pick = 0; pick < beta && covered < target; ++pick) {
        MaxValIndex best;

        Kokkos::parallel_reduce(
            "rho_local_argmax",
            Kokkos::RangePolicy<ExecSpace>(0, N),
            KOKKOS_LAMBDA(int i, MaxValIndex& upd) {
                if (!d_taken(i)) {
                    const double v = d_ei(i);
                    if (v > upd.value) {
                        upd.value = v;
                        upd.index = i;
                    }
                }
            },
            Kokkos::Max<MaxValIndex>(best));

        if (best.index < 0 || !(best.value > 0.0)) break;

        Kokkos::parallel_for(
            "rho_local_mark",
            Kokkos::RangePolicy<ExecSpace>(0, 1),
            KOKKOS_LAMBDA(int) {
                d_taken(best.index) = 1;
            });

        Kokkos::fence();

        covered += best.value;
        out.push_back(best.index);
    }

    if (out.empty() && N > 0) out.push_back(0);
    return out;
}

KOKKOS_INLINE_FUNCTION
static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
}

KOKKOS_INLINE_FUNCTION
static double hashUnit01(uint64_t x)
{
    return static_cast<double>(mix64(x) & 0x1FFFFFFFFFFFFFull) /
           static_cast<double>(0x20000000000000ull);
}

KOKKOS_INLINE_FUNCTION
static double fbdProbeOffset(FBDExploration exploration,
                             int probe,
                             int candidate,
                             int d,
                             int dimension,
                             double radius,
                             int seed,
                             int depth)
{
    if (exploration == FBDExploration::Center) {
        return 0.0;
    }

    if (exploration == FBDExploration::CenterAxial) {
        if (probe <= 0) return 0.0;
        if (dimension <= 0) return 0.0;
        const int axisStep = (probe - 1) / 2;
        const int axis = static_cast<int>(
            (static_cast<uint64_t>(candidate + 1) * 0x9E3779B97F4A7C15ull +
             static_cast<uint64_t>(seed + depth + axisStep)) %
            static_cast<uint64_t>(dimension));
        if (d != axis) return 0.0;
        const double sign = ((probe - 1) % 2 == 0) ? 1.0 : -1.0;
        return sign * 0.5 * radius;
    }

    if (exploration == FBDExploration::CenterRandom && probe <= 0) {
        return 0.0;
    }

    uint64_t h = static_cast<uint64_t>(candidate + 1) *
                 0x9E3779B97F4A7C15ull;
    h ^= static_cast<uint64_t>(d + 17) * 0xBF58476D1CE4E5B9ull;
    h ^= static_cast<uint64_t>(seed + 1) * 0x94D049BB133111EBull;
    h ^= static_cast<uint64_t>(depth + 31) * 0xD6E8FEB86659FD93ull;
    h ^= static_cast<uint64_t>(probe + 101) * 0xA0761D6478BD642Full;

    if (exploration == FBDExploration::Corners) {
        return ((mix64(h) & 1ull) ? 0.5 : -0.5) * radius;
    }

    const double centered = 2.0 * hashUnit01(h) - 1.0;
    const double scale =
        0.5 / sqrt(static_cast<double>(dimension > 0 ? dimension : 1));
    return centered * scale * radius;
}

static std::vector<int> selectTopWeights(const std::vector<double>& weights,
                                         int beta)
{
    std::vector<int> order(weights.size());
    std::iota(order.begin(), order.end(), 0);

    beta = std::min(beta, static_cast<int>(weights.size()));
    if (beta <= 0) return {};

    std::partial_sort(order.begin(),
                      order.begin() + beta,
                      order.end(),
                      [&](int a, int b) {
                          if (weights[a] != weights[b]) {
                              return weights[a] > weights[b];
                          }
                          return a < b;
                      });
    order.resize(beta);
    return order;
}

static std::vector<int> selectFBDFrontier(const std::vector<double>& weights,
                                          double rho,
                                          int beta,
                                          const FBDConfig& cfg,
                                          uint64_t salt)
{
    const int n = static_cast<int>(weights.size());
    const int cap = std::min(beta > 0 ? beta : n, n);
    if (cap <= 0) return {};

    switch (cfg.treeSearch) {
        case FBDTreeSearch::BreadthFirst: {
            std::vector<int> out(cap);
            std::iota(out.begin(), out.end(), 0);
            return out;
        }
        case FBDTreeSearch::DepthFirst: {
            std::vector<int> out;
            out.reserve(cap);
            for (int i = 0; i < cap; ++i) out.push_back(n - 1 - i);
            return out;
        }
        case FBDTreeSearch::EpsilonGreedy: {
            std::vector<int> out = selectTopWeights(weights, cap);
            if (n > cap && cfg.epsilon > 0.0 && !out.empty()) {
                const double u =
                    static_cast<double>(mix64(salt) & 0x1FFFFFFFFFFFFFull) /
                    static_cast<double>(0x20000000000000ull);
                if (u < cfg.epsilon) {
                    const int candidate =
                        static_cast<int>(mix64(salt ^ 0xA0761D6478BD642Full) %
                                         static_cast<uint64_t>(n));
                    if (std::find(out.begin(), out.end(), candidate) ==
                        out.end()) {
                        out.back() = candidate;
                    }
                }
            }
            return out;
        }
        case FBDTreeSearch::CyclicBest:
        case FBDTreeSearch::BestFirst:
        case FBDTreeSearch::Beam:
        default:
            (void)rho;
            return selectTopWeights(weights, cap);
    }
}

static std::vector<double> fbdWeightsFromScores(
    const Kokkos::View<double*, mem_space>& d_scores)
{
    auto h_scores =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_scores);

    const int n = static_cast<int>(h_scores.extent(0));
    std::vector<double> weights(static_cast<size_t>(n), 1e-12);

    double bestScore = std::numeric_limits<double>::infinity();
    for (int i = 0; i < n; ++i) {
        const double s = h_scores(i);
        if (std::isfinite(s) && s < bestScore) bestScore = s;
    }
    if (!std::isfinite(bestScore)) return weights;

    for (int i = 0; i < n; ++i) {
        const double s = h_scores(i);
        if (!std::isfinite(s)) continue;
        const double delta = std::max(0.0, s - bestScore);
        weights[static_cast<size_t>(i)] = 1.0 / (1.0 + delta);
    }

    return weights;
}

// ============================================================================
// Projection helper kept for compatibility/debugging
// ============================================================================

std::vector<double> projectCandidatesPCA(const std::vector<double>& X_host,
                                         const std::vector<double>& mu_X,
                                         const std::vector<double>& sd_X,
                                         const std::vector<double>& pca_basis,
                                         int numCandidates,
                                         int origDim,
                                         int reducedDim) {
    std::vector<double> out(static_cast<size_t>(numCandidates) *
                            static_cast<size_t>(reducedDim),
                            0.0);

    for (int i = 0; i < numCandidates; ++i) {
        std::vector<double> z(origDim);

        for (int d = 0; d < origDim; ++d) {
            z[d] = (X_host[static_cast<size_t>(i) * origDim + d] - mu_X[d]) / sd_X[d];
        }

        for (int k = 0; k < reducedDim; ++k) {
            double acc = 0.0;

            for (int d = 0; d < origDim; ++d) {
                acc += z[d] * pca_basis[d + origDim * k];
            }

            out[static_cast<size_t>(i) * reducedDim + k] = acc;
        }
    }

    return out;
}

// ============================================================================
// MPI/Kokkos timing helpers
// ============================================================================

inline void mpi_progress_some(MPI_Request* reqs, int n) {
    int index = MPI_UNDEFINED;
    int flag  = 0;

    if (n > 0) {
        MPI_Testany(n, reqs, &index, &flag, MPI_STATUS_IGNORE);
    }
}

struct ScopedSyncTimer {
    const char* name;
    MPI_Comm comm;
    int rank = 0;
    int size = 1;
    double t0 = 0.0;

    ScopedSyncTimer(const char* n, MPI_Comm c)
        : name(n), comm(c) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);

        Kokkos::fence();

#ifdef KOKKOS_ENABLE_CUDA
        cudaDeviceSynchronize();
#endif

#ifdef KOKKOS_ENABLE_HIP
        hipDeviceSynchronize();
#endif

        MPI_Barrier(comm);
        t0 = MPI_Wtime();
    }

    ~ScopedSyncTimer() {
        Kokkos::fence();

#ifdef KOKKOS_ENABLE_CUDA
        cudaDeviceSynchronize();
#endif

#ifdef KOKKOS_ENABLE_HIP
        hipDeviceSynchronize();
#endif

        MPI_Barrier(comm);

        const double t1 = MPI_Wtime();
        const double dt = t1 - t0;

        double dt_max = 0.0;
        double dt_min = 0.0;
        double dt_sum = 0.0;

        MPI_Reduce(&dt, &dt_max, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        MPI_Reduce(&dt, &dt_min, 1, MPI_DOUBLE, MPI_MIN, 0, comm);
        MPI_Reduce(&dt, &dt_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

        if (rank == 0) {
            std::cout << std::fixed << std::setprecision(6)
                      << "[verify] " << name
                      << "  max=" << dt_max << "s"
                      << "  min=" << dt_min << "s"
                      << "  avg=" << (dt_sum / size) << "s"
                      << std::endl;
        }
    }
};

#define VERIFY_SCOPE_MAX_COMM(name, comm) ScopedSyncTimer TOKENPASTE(_vts_, __LINE__)(name, comm)

// ============================================================================
// Fractal virtual-child helpers
// ============================================================================

static int childCountForConfig(int dimension, const FBOConfig& cfg)
{
    return fboChildrenPerParent(dimension,
                                cfg.decomposition,
                                cfg.hypercubeActiveDims,
                                cfg.voronoiSites,
                                cfg.simplexChildren);
}

static uint64_t hashFractalID(const Hypersphere& parent, uint64_t seed)
{
    uint64_t h = seed ^ 0x9E3779B97F4A7C15ull;
    h ^= static_cast<uint64_t>(parent.depth + 1) * 0xBF58476D1CE4E5B9ull;
    for (const auto& move : parent.id) {
        h ^= static_cast<uint64_t>(move.first + 0x9E37) +
             (static_cast<uint64_t>(move.second + 17) << 32);
        h ^= h >> 30;
        h *= 0xBF58476D1CE4E5B9ull;
        h ^= h >> 27;
        h *= 0x94D049BB133111EBull;
        h ^= h >> 31;
    }
    return h;
}

static int chooseSplitDimension(const Hypersphere& parent,
                                int dimension,
                                const FBOConfig& cfg)
{
    if (dimension <= 1) return 0;

    if (cfg.splitPolicy == FBOSplitPolicy::Random) {
        return static_cast<int>(
            hashFractalID(parent, 0xD6E8FEB86659FD93ull) %
            static_cast<uint64_t>(dimension));
    }

    if (cfg.splitPolicy == FBOSplitPolicy::LongestSide) {
        std::vector<int> splitCounts(dimension, 0);
        for (const auto& move : parent.id) {
            if (move.first >= 0 && move.first < dimension) {
                ++splitCounts[move.first];
            }
        }

        int bestDim = 0;
        int bestCount = splitCounts[0];
        for (int d = 1; d < dimension; ++d) {
            if (splitCounts[d] < bestCount) {
                bestCount = splitCounts[d];
                bestDim = d;
            }
        }
        return bestDim;
    }

    return parent.depth % dimension;
}

static std::vector<int> chooseHypercubeActiveDims(const Hypersphere& parent,
                                                  int dimension,
                                                  const FBOConfig& cfg)
{
    const int active =
        fboActiveHypercubeDims(dimension, cfg.hypercubeActiveDims);

    std::vector<int> dims;
    dims.reserve(static_cast<size_t>(active));

    if (active <= 0) return dims;

    if (cfg.splitPolicy == FBOSplitPolicy::LongestSide) {
        std::vector<int> splitCounts(dimension, 0);
        for (const auto& move : parent.id) {
            if (move.first >= 0 && move.first < dimension) {
                ++splitCounts[move.first];
            }
        }

        std::vector<int> order(dimension);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b) {
            if (splitCounts[a] != splitCounts[b]) {
                return splitCounts[a] < splitCounts[b];
            }
            return a < b;
        });

        dims.insert(dims.end(), order.begin(), order.begin() + active);
        return dims;
    }

    int start = parent.depth % dimension;
    if (cfg.splitPolicy == FBOSplitPolicy::Random) {
        start = static_cast<int>(
            hashFractalID(parent, 0xA0761D6478BD642Full) %
            static_cast<uint64_t>(dimension));
    }

    for (int i = 0; i < active; ++i) {
        dims.push_back((start + i) % dimension);
    }
    return dims;
}

static std::vector<std::pair<int, int>>
childMovesForLocalIndex(const Hypersphere& parent,
                        int localChild,
                        int dimension,
                        const FBOConfig& cfg)
{
    std::vector<std::pair<int, int>> moves;

    switch (cfg.decomposition) {
        case FBODecomposition::Hypercube: {
            const std::vector<int> dims =
                chooseHypercubeActiveDims(parent, dimension, cfg);
            moves.reserve(dims.size());
            for (int bit = 0; bit < static_cast<int>(dims.size()); ++bit) {
                const int code = ((localChild >> bit) & 1) ? +1 : -1;
                moves.emplace_back(dims[bit], code);
            }
            break;
        }

        case FBODecomposition::Bisection: {
            const int dim = chooseSplitDimension(parent, dimension, cfg);
            moves.emplace_back(dim, localChild == 0 ? -1 : +1);
            break;
        }

        case FBODecomposition::Trisection: {
            const int dim = chooseSplitDimension(parent, dimension, cfg);
            const int code = localChild == 0 ? -1 : (localChild == 1 ? 0 : +1);
            moves.emplace_back(dim, code);
            break;
        }

        case FBODecomposition::Voronoi:
            if (localChild == 0) {
                moves.emplace_back(0, 0);
            } else {
                const int q = localChild - 1;
                moves.emplace_back((q / 2) % dimension, (q & 1) ? +1 : -1);
            }
            break;

        case FBODecomposition::Simplex:
            if (localChild == 0) {
                moves.emplace_back(0, 0);
            } else {
                moves.emplace_back((localChild - 1) % dimension, +1);
            }
            break;

        case FBODecomposition::Hypersphere:
        default:
            moves.emplace_back(localChild / 2, (localChild & 1) ? +1 : -1);
            break;
    }

    return moves;
}

static Hypersphere makeChildHypersphereFromGlobal(size_t globalChild,
                                                  const std::vector<Hypersphere>& parents,
                                                  int dimension,
                                                  const FBOConfig& cfg)
{
    const int children = childCountForConfig(dimension, cfg);
    const size_t parentIndex =
        children > 0 ? globalChild / static_cast<size_t>(children) : 0;
    const int localChild =
        children > 0 ? static_cast<int>(globalChild %
                                        static_cast<size_t>(children))
                     : 0;

    if (parentIndex >= parents.size()) {
        throw std::runtime_error("makeChildHypersphereFromGlobal: parent index out of range");
    }

    HypersphereID id = parents[parentIndex].id;
    const auto moves =
        childMovesForLocalIndex(parents[parentIndex],
                                localChild,
                                dimension,
                                cfg);
    id.insert(id.end(), moves.begin(), moves.end());
    return Hypersphere(dimension, static_cast<int>(id.size()), std::move(id));
}

static std::vector<Hypersphere> decomposeFractalCell(const Hypersphere& parent,
                                                     int dimension,
                                                     const FBOConfig& cfg)
{
    const int children = childCountForConfig(dimension, cfg);
    std::vector<Hypersphere> out;
    out.reserve(static_cast<size_t>(children));

    std::vector<Hypersphere> parents{parent};
    for (int child = 0; child < children; ++child) {
        out.push_back(makeChildHypersphereFromGlobal(
            static_cast<size_t>(child),
            parents,
            dimension,
            cfg));
    }
    return out;
}

template <class MemSpace>
void prepareChildChunkFromParents(const std::vector<Hypersphere>& parents,
                                  size_t childStart,
                                  size_t childCount,
                                  int D,
                                  const FBOConfig& cfg,
                                  Kokkos::View<int*, MemSpace>& depths,
                                  Kokkos::View<size_t*, MemSpace>& offs,
                                  Kokkos::View<int*, MemSpace>& dims,
                                  Kokkos::View<int*, MemSpace>& signs) {
    size_t total_ints = 0;

    for (size_t j = 0; j < childCount; ++j) {
        Hypersphere child =
            makeChildHypersphereFromGlobal(childStart + j, parents, D, cfg);
        total_ints += child.id.size();
    }

    depths = Kokkos::View<int*, MemSpace>("depths_chunk", childCount);
    offs   = Kokkos::View<size_t*, MemSpace>("offs_chunk", childCount);
    dims   = Kokkos::View<int*, MemSpace>("idDims_chunk", total_ints);
    signs  = Kokkos::View<int*, MemSpace>("idSigns_chunk", total_ints);

    auto h_depths = Kokkos::create_mirror_view(depths);
    auto h_offs   = Kokkos::create_mirror_view(offs);
    auto h_dims   = Kokkos::create_mirror_view(dims);
    auto h_signs  = Kokkos::create_mirror_view(signs);

    size_t cursor = 0;

    for (size_t j = 0; j < childCount; ++j) {
        Hypersphere child =
            makeChildHypersphereFromGlobal(childStart + j, parents, D, cfg);
        const auto& pid = child.id;
        const int depth = static_cast<int>(pid.size());

        h_depths(j) = depth;
        h_offs(j)   = cursor;

        for (int dd = 0; dd < depth; ++dd) {
            h_dims(cursor + dd)  = pid[dd].first;
            h_signs(cursor + dd) = pid[dd].second;
        }

        cursor += static_cast<size_t>(depth);
    }

    Kokkos::deep_copy(depths, h_depths);
    Kokkos::deep_copy(offs, h_offs);
    Kokkos::deep_copy(dims, h_dims);
    Kokkos::deep_copy(signs, h_signs);
}

struct FBDProbeView {
    Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> centers;
    Kokkos::View<double*, mem_space> radii;
    FBDExploration exploration;
    int candidate, probe, dimension, seed, depth;
    double lower, upper;
    KOKKOS_INLINE_FUNCTION int extent(int) const { return dimension; }
    KOKKOS_INLINE_FUNCTION double operator()(int d) const {
        double x = centers(candidate, d) + fbdProbeOffset(
            exploration, probe, candidate, d, dimension, radii(candidate), seed, depth);
        return x < lower ? lower : (x > upper ? upper : x);
    }
};

static void computeFBDScores(
    int numCandidates,
    int dimension,
    const Kokkos::View<double**, Kokkos::LayoutLeft, mem_space>& centers,
    const Kokkos::View<double*, mem_space>& radii,
    const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& bestSolution,
    double bestObjectiveValue,
    double objectiveLowerBound,
    double objectiveUpperBound,
    const FBDConfig& cfg,
    int seed,
    int depth,
    int expenseFactor,
    ObjectiveType objType,
    Kokkos::View<double*, mem_space>& scores,
    Kokkos::View<double*, mem_space>& bestValues,
    Kokkos::View<int*, mem_space>& bestProbes)
{
    Kokkos::realloc(scores, numCandidates);
    Kokkos::realloc(bestValues, numCandidates);
    Kokkos::realloc(bestProbes, numCandidates);

    const FBDExploration exploration = cfg.exploration;
    const FBDScoring scoring = cfg.scoring;
    const int pointsPerCell = std::max(1, cfg.pointsPerCell);

    if (!usesExternalObjective(objType)) {
        const bool additive = usesAdditiveDeviceObjective(objType);
        recordObjectiveEvaluations(static_cast<long long>(numCandidates) * pointsPerCell, true);
        using team_policy = Kokkos::TeamPolicy<exec_space>;
        using member_type = team_policy::member_type;

        Kokkos::View<double*, mem_space> probeFx("fbd_probe_fx",
                                                 numCandidates * pointsPerCell);
        Kokkos::View<double*, mem_space> probeDist("fbd_probe_dist",
                                                   numCandidates * pointsPerCell);

        Kokkos::parallel_for(
            "fbd_probe_eval_device",
            team_policy(numCandidates * pointsPerCell, Kokkos::AUTO),
            KOKKOS_LAMBDA(const member_type& team) {
                const int flat = team.league_rank();
                const int i = flat / pointsPerCell;
                const int p = flat - i * pointsPerCell;
                const double r = radii(i);

                double termSum = 0.0;
                double dist2 = 0.0;

                if (additive) Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, dimension),
                    [&](int d, double& acc) {
                        double x = centers(i, d) +
                                   fbdProbeOffset(exploration,
                                                  p,
                                                  i,
                                                  d,
                                                  dimension,
                                                  r,
                                                  seed,
                                                  depth);

                        if (x < objectiveLowerBound) x = objectiveLowerBound;
                        if (x > objectiveUpperBound) x = objectiveUpperBound;

                        acc += objectiveTerm(objType,
                                             x,
                                             d,
                                             dimension,
                                             expenseFactor);
                    },
                    termSum);
                else Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, 1),
                    [&](int, double& acc) {
                        FBDProbeView point{centers, radii, exploration, i, p,
                            dimension, seed, depth, objectiveLowerBound, objectiveUpperBound};
                        acc += computeBuiltInObjective(point, expenseFactor, objType);
                    }, termSum);

                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team, dimension),
                    [&](int d, double& acc) {
                        double x = centers(i, d) +
                                   fbdProbeOffset(exploration,
                                                  p,
                                                  i,
                                                  d,
                                                  dimension,
                                                  r,
                                                  seed,
                                                  depth);

                        if (x < objectiveLowerBound) x = objectiveLowerBound;
                        if (x > objectiveUpperBound) x = objectiveUpperBound;

                        const double diff = x - bestSolution(d);
                        acc += diff * diff;
                    },
                    dist2);

                Kokkos::single(Kokkos::PerTeam(team), [&]() {
                    probeFx(flat) = objectiveConstant(objType) + termSum;
                    probeDist(flat) = sqrt(dist2);
                });
            });

        Kokkos::parallel_for(
            "fbd_score_reduce_probes",
            Kokkos::RangePolicy<exec_space>(0, numCandidates),
            KOKKOS_LAMBDA(int i) {
                const double r = radii(i);
                double bestFx = DBL_MAX;
                double bestDist = DBL_MAX;
                int bestProbe = 0;
                double meanFx = 0.0;

                for (int p = 0; p < pointsPerCell; ++p) {
                    const int flat = i * pointsPerCell + p;
                    const double fx = probeFx(flat);
                    meanFx += fx;
                    if (fx < bestFx) {
                        bestFx = fx;
                        bestDist = probeDist(flat);
                        bestProbe = p;
                    }
                }

                meanFx /= static_cast<double>(pointsPerCell);
                double dist = bestDist;
                if (dist <= 1e-12) dist = 1e-12;

                const double improvement = bestFx - bestObjectiveValue;
                double score = bestFx;

                switch (scoring) {
                    case FBDScoring::MeanValue:
                        score = meanFx;
                        break;
                    case FBDScoring::Improvement:
                        score = improvement;
                        break;
                    case FBDScoring::ImprovementDistance:
                        score = improvement / dist;
                        break;
                    case FBDScoring::Optimistic:
                        score = bestFx -
                                0.05 * r *
                                    sqrt(static_cast<double>(
                                        dimension > 0 ? dimension : 1));
                        break;
                    case FBDScoring::BestValue:
                    default:
                        score = bestFx;
                        break;
                }

                scores(i) = score;
                bestValues(i) = bestFx;
                bestProbes(i) = bestProbe;
            });

        Kokkos::fence();
        return;
    }

    auto h_centers =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), centers);
    auto h_radii =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), radii);
    auto h_best =
        Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bestSolution);
    auto h_scores = Kokkos::create_mirror_view(scores);
    auto h_bestValues = Kokkos::create_mirror_view(bestValues);
    auto h_bestProbes = Kokkos::create_mirror_view(bestProbes);

    Kokkos::View<double*, Kokkos::HostSpace> x("fbd_score_x", dimension);

    for (int i = 0; i < numCandidates; ++i) {
        const double r = h_radii(i);
        double bestFx = std::numeric_limits<double>::max();
        double bestDist = std::numeric_limits<double>::max();
        int bestProbe = 0;
        double meanFx = 0.0;

        for (int p = 0; p < pointsPerCell; ++p) {
            double dist2 = 0.0;

            for (int d = 0; d < dimension; ++d) {
                double xd = h_centers(i, d) +
                    fbdProbeOffset(exploration,
                                   p,
                                   i,
                                   d,
                                   dimension,
                                   r,
                                   seed,
                                   depth);

                if (xd < objectiveLowerBound) xd = objectiveLowerBound;
                if (xd > objectiveUpperBound) xd = objectiveUpperBound;
                x(d) = xd;

                const double diff = xd - h_best(d);
                dist2 += diff * diff;
            }

            double fx = 0.0;
            if (usesExternalObjective(objType)) {
                external_eval_batch_host(x.data(), dimension, 1, &fx);
            } else {
                fx = computeBuiltInObjective(x, expenseFactor, objType);
            }
            const double dist = std::sqrt(dist2);
            meanFx += fx;
            if (fx < bestFx) {
                bestFx = fx;
                bestDist = dist;
                bestProbe = p;
            }
        }

        meanFx /= static_cast<double>(pointsPerCell);
        double dist = bestDist;
        if (dist <= 1e-12) dist = 1e-12;

        const double improvement = bestFx - bestObjectiveValue;
        double score = bestFx;

        switch (scoring) {
            case FBDScoring::MeanValue:
                score = meanFx;
                break;
            case FBDScoring::Improvement:
                score = improvement;
                break;
            case FBDScoring::ImprovementDistance:
                score = improvement / dist;
                break;
            case FBDScoring::Optimistic:
                score = bestFx -
                        0.05 * r *
                            std::sqrt(static_cast<double>(
                                dimension > 0 ? dimension : 1));
                break;
            case FBDScoring::BestValue:
            default:
                score = bestFx;
                break;
        }

        h_scores(i) = score;
        h_bestValues(i) = bestFx;
        h_bestProbes(i) = bestProbe;
    }

    Kokkos::deep_copy(scores, h_scores);
    Kokkos::deep_copy(bestValues, h_bestValues);
    Kokkos::deep_copy(bestProbes, h_bestProbes);
    Kokkos::fence();
}

// ============================================================================
// CLI helpers specific to this main
// ============================================================================

int parseExploitMult(int argc, char** argv, int wrank) {
    int m = 4;

    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--exploit-mult") {
            try {
                m = std::max(1, std::stoi(argv[i + 1]));
            } catch (...) {
            }
            break;
        }
    }

    if (wrank == 0) {
        std::cout << "[Info] exploit multiplier = " << m << "\n";
    }

    return m;
}

int parseExploitBudget(int argc, char** argv, int wrank) {
    int b = -1;

    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == "--exploit-budget") {
            try {
                b = std::max(-1, std::stoi(argv[i + 1]));
            } catch (...) {
            }
            break;
        }
    }

    if (wrank == 0) {
        if (b >= 0) {
            std::cout << "[Info] exploitation budget = " << b << " (global)\n";
        } else {
            std::cout << "[Info] exploitation budget = auto from --exploit-frac\n";
        }
    }

    return b;
}


static int parseOptionalIntArg(int argc,
                               char** argv,
                               const std::string& flag,
                               int defaultValue)
{
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) {
            return std::stoi(argv[i + 1]);
        }
    }
    return defaultValue;
}

static bool hasOptionalArg(int argc,
                           char** argv,
                           const std::string& flag)
{
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}






static double parseOptionalDoubleArg(int argc,
                                     char** argv,
                                     const std::string& flag,
                                     double defaultValue)
{
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) {
            return std::stod(argv[i + 1]);
        }
    }
    return defaultValue;
}






static std::string parseOptionalStringArg(int argc,
                                          char** argv,
                                          const std::string& flag,
                                          const std::string& defaultValue)
{
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) {
            return std::string(argv[i + 1]);
        }
    }
    return defaultValue;
}

// ============================================================================
// Portable depth-barrier checkpointing
// ============================================================================

struct FTCheckpointState {
    int version = 2;
    int worldSize = 1;
    int worldRank = 0;
    int completedDepth = 0;
    int dimension = 0;
    int budget = 0;
    int seed = 0;
    int nInit = 0;
    int optimizerFamily = 0;
    int surrogate = 0;
    int doeDistribution =
        static_cast<int>(FBODoeDistribution::Replicated);
    double bestObjective = std::numeric_limits<double>::infinity();
    std::vector<double> bestPoint;
    std::vector<Hypersphere> frontier;
    std::vector<double> trainingPoints;
    std::vector<double> trainingY;
};

static std::string ftRankPath(const std::string& base, int rank)
{
    std::ostringstream os;
    os << base << ".rank" << std::setw(5) << std::setfill('0') << rank;
    return os.str();
}

static std::string ftDepthBase(const std::string& base, int depth)
{
    std::ostringstream os;
    os << base << ".depth" << std::setw(6) << std::setfill('0') << depth;
    return os.str();
}

static bool ftRankCheckpointExists(const std::string& base, int rank)
{
    if (base.empty()) {
        return false;
    }
    std::error_code ec;
    return std::filesystem::exists(ftRankPath(base, rank), ec);
}

static bool ftEndsWith(const std::string& value, const std::string& suffix)
{
    return value.size() >= suffix.size() &&
           value.compare(value.size() - suffix.size(),
                         suffix.size(),
                         suffix) == 0;
}

static std::uint64_t ftFileSizeBytes(const std::string& path)
{
    std::error_code ec;
    const auto size = std::filesystem::file_size(path, ec);
    if (ec) {
        throw std::runtime_error("could not stat checkpoint " + path +
                                 ": " + ec.message());
    }
    return static_cast<std::uint64_t>(size);
}

static std::uint64_t ftFileFnv1a64(const std::string& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("could not open checkpoint for checksum: " +
                                 path);
    }

    constexpr std::uint64_t offset = 14695981039346656037ull;
    constexpr std::uint64_t prime = 1099511628211ull;
    std::uint64_t hash = offset;

    char buffer[1 << 15];
    while (in) {
        in.read(buffer, sizeof(buffer));
        const std::streamsize got = in.gcount();
        for (std::streamsize i = 0; i < got; ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]);
            hash *= prime;
        }
    }
    return hash;
}

static std::unordered_map<std::string, std::string>
ftReadManifest(const std::string& base)
{
    std::unordered_map<std::string, std::string> values;
    std::ifstream in(base + ".manifest");
    std::string line;
    while (std::getline(in, line)) {
        const auto pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        values[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return values;
}

static void ftPublishTextFileAtomically(const std::string& path,
                                        const std::string& text)
{
    namespace fs = std::filesystem;
    const std::string tmpPath = path + ".tmp";
    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }

    std::ofstream out(tmpPath, std::ios::trunc);
    if (!out) {
        throw std::runtime_error("could not open manifest for writing: " +
                                 tmpPath);
    }
    out << text;
    out.close();
    if (!out) {
        throw std::runtime_error("failed while writing manifest: " + tmpPath);
    }

    std::error_code ec;
    fs::rename(tmpPath, path, ec);
    if (ec) {
        fs::remove(path, ec);
        ec.clear();
        fs::rename(tmpPath, path, ec);
    }
    if (ec) {
        throw std::runtime_error("could not publish manifest " + path +
                                 ": " + ec.message());
    }
}

static std::string ftReadLatestCheckpointBase(const std::string& base)
{
    std::ifstream in(base + ".latest");
    std::string line;
    if (!std::getline(in, line)) {
        return "";
    }
    return line;
}

static void ftAppendUnique(std::vector<std::string>& values,
                           const std::string& value)
{
    if (value.empty()) {
        return;
    }
    if (std::find(values.begin(), values.end(), value) == values.end()) {
        values.push_back(value);
    }
}

static std::vector<std::string> ftCheckpointCandidateBases(
    const std::string& base)
{
    namespace fs = std::filesystem;
    std::vector<std::string> candidates;
    ftAppendUnique(candidates, ftReadLatestCheckpointBase(base));

    const fs::path basePath(base);
    const fs::path dir = basePath.parent_path().empty()
        ? fs::path(".")
        : basePath.parent_path();
    const std::string prefix = basePath.filename().string() + ".depth";
    const std::string suffix = ".manifest";

    std::vector<std::string> generationBases;
    std::error_code ec;
    if (fs::exists(dir, ec)) {
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (ec) {
                break;
            }
            if (!entry.is_regular_file(ec)) {
                continue;
            }
            const std::string name = entry.path().filename().string();
            if (name.rfind(prefix, 0) == 0 && ftEndsWith(name, suffix)) {
                const std::string stem =
                    name.substr(0, name.size() - suffix.size());
                generationBases.push_back((dir / stem).string());
            }
        }
    }

    std::sort(generationBases.begin(), generationBases.end());
    std::reverse(generationBases.begin(), generationBases.end());
    for (const auto& generationBase : generationBases) {
        ftAppendUnique(candidates, generationBase);
    }

    ftAppendUnique(candidates, base);
    return candidates;
}

static void ftRemoveGenerationFiles(const std::string& generationBase)
{
    namespace fs = std::filesystem;
    const fs::path basePath(generationBase);
    const fs::path dir = basePath.parent_path().empty()
        ? fs::path(".")
        : basePath.parent_path();
    const std::string prefix = basePath.filename().string();

    std::error_code ec;
    if (!fs::exists(dir, ec)) {
        return;
    }
    for (const auto& entry : fs::directory_iterator(dir, ec)) {
        if (ec) {
            break;
        }
        if (!entry.is_regular_file(ec)) {
            continue;
        }
        const std::string name = entry.path().filename().string();
        if (name == prefix + ".manifest" ||
            name.rfind(prefix + ".rank", 0) == 0) {
            fs::remove(entry.path(), ec);
            ec.clear();
        }
    }
}

static void ftPruneOldGenerations(const std::string& base, int keep)
{
    if (keep <= 0) {
        return;
    }
    auto candidates = ftCheckpointCandidateBases(base);
    std::vector<std::string> generations;
    const std::string marker = ".depth";
    for (const auto& candidate : candidates) {
        if (candidate.find(marker) != std::string::npos) {
            ftAppendUnique(generations, candidate);
        }
    }
    std::sort(generations.begin(), generations.end());
    std::reverse(generations.begin(), generations.end());
    for (size_t i = static_cast<size_t>(keep); i < generations.size(); ++i) {
        ftRemoveGenerationFiles(generations[i]);
    }
}

static std::uint64_t ftParseU64Manifest(
    const std::unordered_map<std::string, std::string>& values,
    const std::string& key)
{
    const auto it = values.find(key);
    if (it == values.end()) {
        throw std::runtime_error("manifest missing key " + key);
    }
    return static_cast<std::uint64_t>(std::stoull(it->second));
}

static volatile std::sig_atomic_t g_ftSignalRequested = 0;
static volatile std::sig_atomic_t g_ftSignalNumber = 0;

static void ftSignalHandler(int signum)
{
    g_ftSignalRequested = 1;
    g_ftSignalNumber = signum;
}

static void installFTSignalHandlers()
{
    std::signal(SIGTERM, ftSignalHandler);
    std::signal(SIGINT, ftSignalHandler);
#ifdef SIGUSR1
    std::signal(SIGUSR1, ftSignalHandler);
#endif
}

static bool ftSignalRequested()
{
    return g_ftSignalRequested != 0;
}

static int ftRequestedSignalNumber()
{
    return static_cast<int>(g_ftSignalNumber);
}

template <typename T>
static void writeScalar(std::ostream& out, const T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
static void readScalar(std::istream& in, T& value)
{
    static_assert(std::is_trivially_copyable_v<T>);
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
}

template <typename T>
static void writeVector(std::ostream& out, const std::vector<T>& values)
{
    static_assert(std::is_trivially_copyable_v<T>);
    const std::uint64_t n = static_cast<std::uint64_t>(values.size());
    writeScalar(out, n);
    if (n > 0) {
        out.write(reinterpret_cast<const char*>(values.data()),
                  static_cast<std::streamsize>(n * sizeof(T)));
    }
}

template <typename T>
static void readVector(std::istream& in, std::vector<T>& values)
{
    static_assert(std::is_trivially_copyable_v<T>);
    std::uint64_t n = 0;
    readScalar(in, n);
    values.resize(static_cast<size_t>(n));
    if (n > 0) {
        in.read(reinterpret_cast<char*>(values.data()),
                static_cast<std::streamsize>(n * sizeof(T)));
    }
}

static void saveFTCheckpoint(const std::string& base,
                             const FTCheckpointState& state)
{
    namespace fs = std::filesystem;
    const std::string path = ftRankPath(base, state.worldRank);
    const std::string tmpPath = path + ".tmp";

    const fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) {
        fs::create_directories(parent);
    }

    std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error("could not open checkpoint for writing: " +
                                 tmpPath);
    }

    const char magic[16] = {'A','S','T','R','A','F','T','C','K','P','T','0','0','1','\0','\0'};
    out.write(magic, sizeof(magic));

    writeScalar(out, state.version);
    writeScalar(out, state.worldSize);
    writeScalar(out, state.worldRank);
    writeScalar(out, state.completedDepth);
    writeScalar(out, state.dimension);
    writeScalar(out, state.budget);
    writeScalar(out, state.seed);
    writeScalar(out, state.nInit);
    writeScalar(out, state.optimizerFamily);
    writeScalar(out, state.surrogate);
    writeScalar(out, state.doeDistribution);
    writeScalar(out, state.bestObjective);
    writeVector(out, state.bestPoint);

    const std::uint64_t nFrontier =
        static_cast<std::uint64_t>(state.frontier.size());
    writeScalar(out, nFrontier);
    for (const auto& hs : state.frontier) {
        auto serialized = serializeHypersphereID(hs.id);
        writeVector(out, serialized);
    }

    writeVector(out, state.trainingPoints);
    writeVector(out, state.trainingY);
    out.close();

    if (!out) {
        throw std::runtime_error("failed while writing checkpoint: " + tmpPath);
    }

    std::error_code ec;
    fs::rename(tmpPath, path, ec);
    if (ec) {
        fs::remove(path, ec);
        ec.clear();
        fs::rename(tmpPath, path, ec);
    }
    if (ec) {
        throw std::runtime_error("could not publish checkpoint " + path +
                                 ": " + ec.message());
    }
}

static FTCheckpointState loadFTCheckpoint(const std::string& base, int rank)
{
    FTCheckpointState state;
    const std::string path = ftRankPath(base, rank);
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("could not open checkpoint for reading: " +
                                 path);
    }

    char magic[16] = {};
    in.read(magic, sizeof(magic));
    const std::string expected("ASTRAFTCKPT001");
    if (std::string(magic, magic + expected.size()) != expected) {
        throw std::runtime_error("invalid checkpoint magic in " + path);
    }

    readScalar(in, state.version);
    readScalar(in, state.worldSize);
    readScalar(in, state.worldRank);
    readScalar(in, state.completedDepth);
    readScalar(in, state.dimension);
    readScalar(in, state.budget);
    readScalar(in, state.seed);
    readScalar(in, state.nInit);
    readScalar(in, state.optimizerFamily);
    readScalar(in, state.surrogate);
    if (state.version >= 2) {
        readScalar(in, state.doeDistribution);
    } else {
        state.doeDistribution =
            static_cast<int>(FBODoeDistribution::Replicated);
    }
    readScalar(in, state.bestObjective);
    readVector(in, state.bestPoint);

    std::uint64_t nFrontier = 0;
    readScalar(in, nFrontier);
    state.frontier.clear();
    state.frontier.reserve(static_cast<size_t>(nFrontier));
    for (std::uint64_t i = 0; i < nFrontier; ++i) {
        std::vector<int> serialized;
        readVector(in, serialized);
        size_t offset = 0;
        HypersphereID id = deserializeHypersphereID(serialized, offset);
        state.frontier.emplace_back(state.dimension,
                                    static_cast<int>(id.size()),
                                    std::move(id));
    }

    readVector(in, state.trainingPoints);
    readVector(in, state.trainingY);
    if (!in) {
        throw std::runtime_error("truncated or corrupt checkpoint: " + path);
    }

    return state;
}

static bool ftRankCheckpointUsable(const std::string& base,
                                   int rank,
                                   int expectedWorldSize,
                                   std::string* reason)
{
    if (base.empty()) {
        if (reason) {
            *reason = "empty checkpoint base";
        }
        return false;
    }

    const std::string path = ftRankPath(base, rank);
    if (!ftRankCheckpointExists(base, rank)) {
        if (reason) {
            *reason = "missing rank checkpoint " + path;
        }
        return false;
    }

    try {
        const auto manifest = ftReadManifest(base);
        if (!manifest.empty()) {
            const auto completeIt = manifest.find("complete");
            if (completeIt == manifest.end() || completeIt->second != "1") {
                if (reason) {
                    *reason = "checkpoint manifest is incomplete";
                }
                return false;
            }

            const auto worldIt = manifest.find("world_size");
            if (worldIt != manifest.end() &&
                std::stoi(worldIt->second) != expectedWorldSize) {
                if (reason) {
                    *reason = "checkpoint world size does not match";
                }
                return false;
            }

            std::ostringstream prefix;
            prefix << "rank" << rank << "_";
            const std::uint64_t expectedBytes =
                ftParseU64Manifest(manifest, prefix.str() + "bytes");
            const std::uint64_t expectedHash =
                ftParseU64Manifest(manifest, prefix.str() + "fnv1a64");

            const std::uint64_t actualBytes = ftFileSizeBytes(path);
            if (actualBytes != expectedBytes) {
                if (reason) {
                    *reason = "checkpoint size mismatch";
                }
                return false;
            }

            const std::uint64_t actualHash = ftFileFnv1a64(path);
            if (actualHash != expectedHash) {
                if (reason) {
                    *reason = "checkpoint checksum mismatch";
                }
                return false;
            }
        }

        const FTCheckpointState state = loadFTCheckpoint(base, rank);
        if (state.worldRank != rank ||
            state.worldSize != expectedWorldSize) {
            if (reason) {
                *reason = "checkpoint rank metadata does not match";
            }
            return false;
        }
    } catch (const std::exception& e) {
        if (reason) {
            *reason = e.what();
        }
        return false;
    }

    return true;
}

static int ftCheckpointWorldSize(const std::string& base)
{
    const auto manifest = ftReadManifest(base);
    const auto it = manifest.find("world_size");
    if (it != manifest.end()) {
        return std::stoi(it->second);
    }
    FTCheckpointState rank0 = loadFTCheckpoint(base, 0);
    return rank0.worldSize;
}

static bool ftCheckpointSetUsable(const std::string& base,
                                  int* checkpointWorldSize,
                                  std::string* reason)
{
    if (base.empty()) {
        if (reason) {
            *reason = "empty checkpoint base";
        }
        return false;
    }

    int oldWorldSize = 0;
    try {
        oldWorldSize = ftCheckpointWorldSize(base);
    } catch (const std::exception& e) {
        if (reason) {
            *reason = e.what();
        }
        return false;
    }

    if (oldWorldSize <= 0) {
        if (reason) {
            *reason = "checkpoint has invalid world size";
        }
        return false;
    }

    for (int r = 0; r < oldWorldSize; ++r) {
        std::string rankReason;
        if (!ftRankCheckpointUsable(base, r, oldWorldSize, &rankReason)) {
            if (reason) {
                std::ostringstream os;
                os << "rank " << r << ": " << rankReason;
                *reason = os.str();
            }
            return false;
        }
    }

    if (checkpointWorldSize) {
        *checkpointWorldSize = oldWorldSize;
    }
    return true;
}

template <typename HostView>
static double squaredDistanceToObjectiveReference(const HostView& point,
                                                  ObjectiveType objTy)
{
    if (!objectiveHasKnownOptimum(objTy)) {
        return std::numeric_limits<double>::quiet_NaN();
    }

    const int dimension = static_cast<int>(point.extent(0));
    double dist2 = 0.0;
    for (int k = 0; k < dimension; ++k) {
        const double ref = objectiveReferenceCoordinate(objTy, k, dimension);
        const double diff = point(k) - ref;
        dist2 += diff * diff;
    }
    return dist2;
}





// ============================================================================
// Global benchmark budget plan
// ============================================================================

struct BudgetPlan {
    int total = 0;
    int doe = 0;
    int exploration = 0;
    int exploitation = 0;
    double exploitFraction = 0.10;
    double depthGamma = 1.0;
    std::vector<int> depthBudgets;
};

static long long globalObjectiveEvaluations()
{
    const long long local = objectiveEvaluationAudit().device.load() +
                            objectiveEvaluationAudit().host.load();
    long long global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return global;
}

static std::vector<int> splitBudgetByWeights(int total,
                                              const std::vector<double>& weights)
{
    std::vector<int> out(weights.size(), 0);

    if (total <= 0 || weights.empty()) {
        return out;
    }

    const double wsum = std::accumulate(weights.begin(), weights.end(), 0.0);

    if (!(wsum > 0.0)) {
        out[0] = total;
        return out;
    }

    std::vector<double> frac(weights.size(), 0.0);
    int assigned = 0;

    for (size_t i = 0; i < weights.size(); ++i) {
        const double exact =
            static_cast<double>(total) * weights[i] / wsum;

        out[i] = static_cast<int>(std::floor(exact));
        frac[i] = exact - static_cast<double>(out[i]);
        assigned += out[i];
    }

    int remainder = total - assigned;

    std::vector<int> order(weights.size());
    std::iota(order.begin(), order.end(), 0);

    std::sort(order.begin(), order.end(), [&](int a, int b) {
        if (frac[a] > frac[b]) return true;
        if (frac[a] < frac[b]) return false;

        // Tie-break: give remaining evaluations to deeper levels.
        return a > b;
    });

    for (int r = 0; r < remainder; ++r) {
        out[order[r % order.size()]] += 1;
    }

    // Low-budget runs must not create a positive budget only at a later
    // depth, because the tree cannot reach that depth without evaluating
    // earlier levels.
    for (size_t i = 0; i < out.size(); ++i) {
        if (out[i] > 0) continue;
        auto donor = std::find_if(out.begin() + static_cast<std::ptrdiff_t>(i + 1),
                                  out.end(),
                                  [](int v) { return v > 0; });
        if (donor == out.end()) break;
        --(*donor);
        ++out[i];
    }

    return out;
}

static std::vector<double> makeDepthWeights(int maxDepth,
                                            double gamma)
{
    if (maxDepth <= 0) {
        return {};
    }

    if (gamma < 0.0) {
        gamma = 0.0;
    }

    std::vector<double> weights(maxDepth, 1.0);

    for (int d = 1; d <= maxDepth; ++d) {
        weights[d - 1] = std::pow(static_cast<double>(d), gamma);
    }

    return weights;
}

static BudgetPlan makeBudgetPlan(int globalBudget,
                                 int nInitGlobal,
                                 int maxDepth,
                                 int userExploitBudget,
                                 double exploitFraction,
                                 double depthGamma,
                                 int worldRank)
{
    BudgetPlan plan;

    plan.total = globalBudget;
    plan.doe = nInitGlobal;
    plan.exploitFraction = exploitFraction;
    plan.depthGamma = depthGamma;

    if (globalBudget <= 0) {
        if (worldRank == 0) {
            std::cerr << "[Error] global budget must be positive.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (maxDepth <= 0) {
        if (worldRank == 0) {
            std::cerr << "[Error] maxDepth must be positive.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (nInitGlobal < 0) {
        if (worldRank == 0) {
            std::cerr << "[Error] n_init cannot be negative.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (nInitGlobal > globalBudget) {
        if (worldRank == 0) {
            std::cerr << "[Error] n_init cannot exceed global budget. "
                      << "n_init=" << nInitGlobal
                      << ", budget=" << globalBudget << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    if (exploitFraction < 0.0) exploitFraction = 0.0;
    if (exploitFraction > 1.0) exploitFraction = 1.0;

    const int remainingAfterDoe = globalBudget - nInitGlobal;

    int exploitBudget = 0;

    if (userExploitBudget >= 0) {
        exploitBudget = userExploitBudget;
    } else {
        exploitBudget = static_cast<int>(
            std::llround(exploitFraction *
                         static_cast<double>(remainingAfterDoe))
        );
    }

    exploitBudget = std::max(0, std::min(exploitBudget, remainingAfterDoe));

    const int explorationBudget = remainingAfterDoe - exploitBudget;

    if (explorationBudget < maxDepth && worldRank == 0) {
        std::cerr << "[Warning] exploration budget is smaller than maxDepth; "
                  << "some depths may receive zero candidate evaluations.\n"
                  << "          explorationBudget=" << explorationBudget
                  << ", maxDepth=" << maxDepth << "\n";
    }

    if (explorationBudget <= 0) {
        if (worldRank == 0) {
            std::cerr << "[Error] exploration budget is zero after DOE and "
                      << "exploitation allocation. Increase budget, lower "
                      << "DOE/exploitation fractions, or use a smaller n_init.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    plan.exploitation = exploitBudget;
    plan.exploration = explorationBudget;

    const std::vector<double> depthWeights =
        makeDepthWeights(maxDepth, depthGamma);

    plan.depthBudgets =
        splitBudgetByWeights(explorationBudget, depthWeights);

    if (worldRank == 0) {
        std::cout << "\n[BudgetPlan]\n";
        std::cout << "  total global budget = " << plan.total << "\n";
        std::cout << "  DOE budget          = " << plan.doe << "\n";
        std::cout << "  remaining budget    = " << remainingAfterDoe << "\n";
        std::cout << "  exploration budget  = " << plan.exploration << "\n";
        std::cout << "  exploitation budget = " << plan.exploitation << "\n";
        std::cout << "  exploit fraction    = " << exploitFraction << "\n";
        std::cout << "  depth gamma         = " << depthGamma << "\n";

        int checkSum = 0;

        for (int d = 0; d < maxDepth; ++d) {
            checkSum += plan.depthBudgets[d];

            const double frac =
                plan.exploration > 0
                    ? 100.0 * static_cast<double>(plan.depthBudgets[d]) /
                          static_cast<double>(plan.exploration)
                    : 0.0;

            std::cout << "  depth " << (d + 1)
                      << " exploration budget = "
                      << plan.depthBudgets[d]
                      << "  (" << std::fixed << std::setprecision(2)
                      << frac << "%)\n";
        }

        std::cout << "  exploration check   = " << checkSum
                  << " / " << plan.exploration << "\n";
        std::cout << "  total check         = "
                  << plan.doe + plan.exploration + plan.exploitation
                  << " / " << plan.total << "\n\n";
    }

    return plan;
}

static BudgetPlan makeBudgetPlanFixedFractions(int globalBudget,
                                               int nInitGlobal,
                                               bool nInitWasExplicit,
                                               int maxDepth,
                                               int userExploitBudget,
                                               double doeFraction,
                                               double exploreFraction,
                                               double exploitFraction,
                                               double depthGamma,
                                               int worldRank)
{
    (void)nInitWasExplicit;

    if (doeFraction < 0.0) doeFraction = 0.25;
    if (exploreFraction < 0.0) exploreFraction = 0.65;
    if (exploitFraction < 0.0) exploitFraction = 0.10;

    const double sum = doeFraction + exploreFraction + exploitFraction;
    if (!(sum > 0.0)) {
        doeFraction = 0.25;
        exploreFraction = 0.65;
        exploitFraction = 0.10;
    } else {
        doeFraction /= sum;
        exploreFraction /= sum;
        exploitFraction /= sum;
    }

    nInitGlobal = std::max(0, std::min(nInitGlobal, globalBudget));

    int exploitBudget = userExploitBudget >= 0
        ? userExploitBudget
        : static_cast<int>(
              std::llround(exploitFraction *
                           static_cast<double>(globalBudget)));

    exploitBudget = std::max(0, std::min(exploitBudget, globalBudget - nInitGlobal));

    int explorationBudget = globalBudget - nInitGlobal - exploitBudget;

    if (exploreFraction > 0.0 && userExploitBudget >= 0) {
        const int targetExplore = static_cast<int>(
            std::llround(exploreFraction *
                         static_cast<double>(globalBudget)));
        explorationBudget = std::min(explorationBudget,
                                     std::max(0, targetExplore));
        exploitBudget = globalBudget - nInitGlobal - explorationBudget;
    }

    BudgetPlan plan;
    plan.total = globalBudget;
    plan.doe = nInitGlobal;
    plan.exploration = std::max(0, explorationBudget);
    plan.exploitation = std::max(0, exploitBudget);
    plan.exploitFraction = exploitFraction;
    plan.depthGamma = depthGamma;

    if (plan.exploration <= 0) {
        if (worldRank == 0) {
            std::cerr << "[Error] fixed fractions leave no exploration budget. "
                      << "Reduce --doe-frac/--exploit-frac or increase budget.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    plan.depthBudgets = splitBudgetByWeights(
        plan.exploration,
        makeDepthWeights(maxDepth, depthGamma));

    if (worldRank == 0) {
        std::cout << "\n[BudgetPlan]\n";
        std::cout << "  mode                = fixed-fractions\n";
        std::cout << "  total global budget = " << plan.total << "\n";
        std::cout << "  DOE budget          = " << plan.doe
                  << " (frac=" << doeFraction << ")\n";
        std::cout << "  exploration budget  = " << plan.exploration
                  << " (frac=" << exploreFraction << ")\n";
        std::cout << "  exploitation budget = " << plan.exploitation
                  << " (frac=" << exploitFraction << ")\n";
        std::cout << "  depth gamma         = " << depthGamma << "\n";

        int checkSum = 0;
        for (int d = 0; d < maxDepth; ++d) {
            checkSum += plan.depthBudgets[d];
            std::cout << "  depth " << (d + 1)
                      << " exploration budget = "
                      << plan.depthBudgets[d] << "\n";
        }
        std::cout << "  exploration check   = " << checkSum
                  << " / " << plan.exploration << "\n";
        std::cout << "  total check         = "
                  << plan.doe + plan.exploration + plan.exploitation
                  << " / " << plan.total << "\n\n";
    }

    return plan;
}











static int defaultNInitForDim(int d)
{
    if (d <= 100) return 100;
    if (d <= 1000) return 250;
    return 500;
}

static std::vector<double> readDoeCsvRowMajor(const std::string& path,
                                              int nRows,
                                              int dim)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open DOE file: " + path);
    }

    std::vector<double> X;
    X.reserve((size_t)nRows * (size_t)dim);

    std::string line;
    while (std::getline(in, line)) {
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);

        double v;
        while (iss >> v) {
            X.push_back(v);
        }
    }

    const size_t expected = (size_t)nRows * (size_t)dim;
    if (X.size() != expected) {
        std::ostringstream oss;
        oss << "DOE file has wrong number of values. Expected "
            << expected << " = " << nRows << " * " << dim
            << ", got " << X.size()
            << ". File: " << path;
        throw std::runtime_error(oss.str());
    }

    return X;
}

static std::vector<double> readDoeCsvShardRowMajor(const std::string& path,
                                                   int firstRow,
                                                   int nRows,
                                                   int dim)
{
    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error("Could not open DOE file: " + path);
    }

    std::vector<double> X;
    X.reserve(static_cast<size_t>(nRows) * static_cast<size_t>(dim));

    const int endRow = firstRow + nRows;
    std::string line;
    for (int row = 0; row < endRow; ++row) {
        if (!std::getline(in, line)) {
            std::ostringstream oss;
            oss << "DOE file ended before row " << endRow
                << ". File: " << path;
            throw std::runtime_error(oss.str());
        }
        if (row < firstRow) {
            continue;
        }

        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream iss(line);
        double value = 0.0;
        int valuesInRow = 0;
        while (iss >> value) {
            X.push_back(value);
            ++valuesInRow;
        }
        if (valuesInRow != dim) {
            std::ostringstream oss;
            oss << "DOE row " << row << " has " << valuesInRow
                << " values; expected " << dim
                << ". File: " << path;
            throw std::runtime_error(oss.str());
        }
    }

    return X;
}

static int doeRowsForRank(int nInitGlobal, int worldSize, int worldRank)
{
    const int base = nInitGlobal / worldSize;
    const int extra = nInitGlobal % worldSize;
    return base + (worldRank < extra ? 1 : 0);
}

static int doeRowStartForRank(int nInitGlobal, int worldSize, int worldRank)
{
    const int base = nInitGlobal / worldSize;
    const int extra = nInitGlobal % worldSize;
    return worldRank * base + std::min(worldRank, extra);
}

static std::vector<double> placeInitialDoe(
    MPI_Comm comm,
    int worldRank,
    int worldSize,
    std::vector<double> globalX,
    int nInitGlobal,
    int dim,
    FBODoeDistribution distribution)
{
    const size_t expected =
        static_cast<size_t>(nInitGlobal) * static_cast<size_t>(dim);
    int rootHasExpectedDesign =
        worldRank != 0 || globalX.size() == expected ? 1 : 0;
    MPI_Bcast(&rootHasExpectedDesign, 1, MPI_INT, 0, comm);
    if (!rootHasExpectedDesign) {
        if (worldRank == 0) {
            std::cerr << "[DOE] internal error: generated design has "
                      << globalX.size() << " values; expected "
                      << expected << "\n";
        }
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (expected > static_cast<size_t>(std::numeric_limits<int>::max())) {
        if (worldRank == 0) {
            std::cerr << "[DOE] MPI transfer exceeds INT_MAX values; "
                      << "reduce n_init or dimension.\n";
        }
        MPI_Abort(comm, EXIT_FAILURE);
    }

    if (distribution == FBODoeDistribution::Replicated) {
        if (worldRank != 0) {
            globalX.resize(expected);
        }
        if (!globalX.empty()) {
            MPI_Bcast(globalX.data(),
                      static_cast<int>(globalX.size()),
                      MPI_DOUBLE,
                      0,
                      comm);
        }
        return globalX;
    }

    const int localRows =
        doeRowsForRank(nInitGlobal, worldSize, worldRank);
    std::vector<double> localX(
        static_cast<size_t>(localRows) * static_cast<size_t>(dim));
    std::vector<int> sendCounts(static_cast<size_t>(worldSize), 0);
    std::vector<int> displacements(static_cast<size_t>(worldSize), 0);

    int displacement = 0;
    for (int rank = 0; rank < worldSize; ++rank) {
        const long long count =
            static_cast<long long>(
                doeRowsForRank(nInitGlobal, worldSize, rank)) *
            static_cast<long long>(dim);
        if (count > std::numeric_limits<int>::max()) {
            if (worldRank == 0) {
                std::cerr << "[DOE] rank " << rank
                          << " shard exceeds INT_MAX values.\n";
            }
            MPI_Abort(comm, EXIT_FAILURE);
        }
        sendCounts[static_cast<size_t>(rank)] =
            static_cast<int>(count);
        displacements[static_cast<size_t>(rank)] = displacement;
        displacement += static_cast<int>(count);
    }

    MPI_Scatterv(worldRank == 0 && !globalX.empty()
                     ? globalX.data()
                     : nullptr,
                 sendCounts.data(),
                 displacements.data(),
                 MPI_DOUBLE,
                 localX.empty() ? nullptr : localX.data(),
                 static_cast<int>(localX.size()),
                 MPI_DOUBLE,
                 0,
                 comm);

    return localX;
}

static std::vector<double> makeRandomDoeHost(int nRows,
                                             int dim,
                                             const std::vector<double>& lower,
                                             const std::vector<double>& upper,
                                             int seed,
                                             int globalRowStart = 0)
{
    std::vector<double> X((size_t)nRows * (size_t)dim);
    std::mt19937_64 rng(static_cast<unsigned long long>(seed));
    rng.discard(static_cast<unsigned long long>(globalRowStart) *
                static_cast<unsigned long long>(dim));

    for (int i = 0; i < nRows; ++i) {
        for (int d = 0; d < dim; ++d) {
            std::uniform_real_distribution<double>
                dist(lower[static_cast<size_t>(d)],
                     upper[static_cast<size_t>(d)]);
            X[static_cast<size_t>(i) * dim + d] = dist(rng);
        }
    }

    return X;
}

static bool isPrimeForDoe(int n)
{
    if (n < 2) return false;
    if (n == 2) return true;
    if (n % 2 == 0) return false;
    for (int p = 3; p * p <= n; p += 2) {
        if (n % p == 0) return false;
    }
    return true;
}

static std::vector<int> firstPrimesForDoe(int count)
{
    std::vector<int> primes;
    primes.reserve(static_cast<size_t>(std::max(0, count)));
    for (int n = 2; static_cast<int>(primes.size()) < count; ++n) {
        if (isPrimeForDoe(n)) primes.push_back(n);
    }
    return primes;
}

static double radicalInverseForDoe(uint64_t index, int base)
{
    double invBase = 1.0 / static_cast<double>(base);
    double factor = invBase;
    double result = 0.0;

    while (index > 0) {
        const uint64_t digit = index % static_cast<uint64_t>(base);
        result += static_cast<double>(digit) * factor;
        index /= static_cast<uint64_t>(base);
        factor *= invBase;
    }

    return result;
}

static std::vector<double> makeLowDiscrepancyDoeHost(
    int nRows,
    int nRowsGlobal,
    int globalRowStart,
    int dim,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    int seed,
    bool hammersley)
{
    std::vector<double> X(static_cast<size_t>(nRows) * dim);
    const int primeCount = hammersley ? std::max(0, dim - 1) : dim;
    const std::vector<int> primes = firstPrimesForDoe(primeCount);
    const uint64_t skip = seed > 0 ? static_cast<uint64_t>(seed) : 0ull;

    for (int i = 0; i < nRows; ++i) {
        const int globalRow = globalRowStart + i;
        const uint64_t idx =
            static_cast<uint64_t>(globalRow) + 1ull + skip;
        for (int d = 0; d < dim; ++d) {
            double u = 0.0;
            if (hammersley && d == 0) {
                u = (static_cast<double>(globalRow) + 0.5) /
                    static_cast<double>(std::max(1, nRowsGlobal));
            } else {
                const int primeIndex = hammersley ? d - 1 : d;
                u = radicalInverseForDoe(idx,
                                         primes[static_cast<size_t>(primeIndex)]);
            }

            const double lo = lower[static_cast<size_t>(d)];
            const double hi = upper[static_cast<size_t>(d)];
            X[static_cast<size_t>(i) * dim + d] = lo + u * (hi - lo);
        }
    }

    return X;
}

static std::uint64_t doeMix64(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

static std::vector<double> makeDeterministicLhsRowsHost(
    int nRows,
    int nRowsGlobal,
    int globalRowStart,
    int dim,
    const std::vector<double>& lower,
    const std::vector<double>& upper,
    int seed)
{
    std::vector<double> X(
        static_cast<size_t>(nRows) * static_cast<size_t>(dim));
    const std::uint64_t seedBits =
        static_cast<std::uint64_t>(static_cast<std::uint32_t>(seed));

    for (int d = 0; d < dim; ++d) {
        const std::uint64_t dimKey =
            doeMix64(seedBits ^ (static_cast<std::uint64_t>(d) << 32));
        int multiplier =
            nRowsGlobal <= 1
                ? 1
                : 1 + static_cast<int>(
                    dimKey % static_cast<std::uint64_t>(nRowsGlobal - 1));
        while (std::gcd(multiplier, nRowsGlobal) != 1) {
            ++multiplier;
            if (multiplier >= nRowsGlobal) multiplier = 1;
        }
        const int offset =
            nRowsGlobal <= 1
                ? 0
                : static_cast<int>(
                    doeMix64(dimKey) %
                    static_cast<std::uint64_t>(nRowsGlobal));

        for (int i = 0; i < nRows; ++i) {
            const int globalRow = globalRowStart + i;
            const int stratum =
                nRowsGlobal <= 1
                    ? 0
                    : static_cast<int>(
                        (static_cast<long long>(multiplier) * globalRow +
                         offset) %
                        nRowsGlobal);
            const std::uint64_t jitterBits =
                doeMix64(dimKey ^
                         static_cast<std::uint64_t>(globalRow));
            const double jitter =
                (static_cast<double>(jitterBits >> 11) + 0.5) *
                (1.0 / 9007199254740992.0);
            const double u =
                (static_cast<double>(stratum) + jitter) /
                static_cast<double>(std::max(1, nRowsGlobal));
            const double lo = lower[static_cast<size_t>(d)];
            const double hi = upper[static_cast<size_t>(d)];
            X[static_cast<size_t>(i) * dim + d] =
                lo + u * (hi - lo);
        }
    }
    return X;
}

static std::vector<double> copyHostBounds(
    const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& bounds)
{
    auto host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bounds);
    std::vector<double> out(host.extent(0));
    for (size_t i = 0; i < out.size(); ++i) out[i] = host(i);
    return out;
}

static std::vector<double> makeInitialDoe(MPI_Comm comm,
                                          int worldRank,
                                          int worldSize,
                                          const FBOConfig& fbo,
                                          int nInitGlobal,
                                          int dim,
                                          int seed,
                                          const std::string& doeFile,
                                          const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& lowerBounds,
                                          const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& upperBounds)
{
    if (nInitGlobal <= 0) {
        return {};
    }

    if (fbo.doe == FBODoe::Sobol || fbo.doe == FBODoe::DoeFile) {
        if (doeFile.empty()) {
            if (worldRank == 0) {
                std::cerr << "[Error] Missing --doe-file for --fbo-doe "
                          << toString(fbo.doe) << ".\n"
                          << "        Generate a Sobol DOE with make_doe.py "
                          << "or choose --fbo-doe random/lhs/halton/hammersley.\n";
            }
            MPI_Abort(comm, EXIT_FAILURE);
        }

        if (worldRank == 0) {
            std::cout << "[DOE] using " << toString(fbo.doe)
                      << " file: " << doeFile << "\n";
        }

        if (fbo.doeDistribution == FBODoeDistribution::Distributed) {
            const int localRows =
                doeRowsForRank(nInitGlobal, worldSize, worldRank);
            const int firstRow =
                doeRowStartForRank(nInitGlobal, worldSize, worldRank);

            std::vector<double> localX;
            std::string localError;
            int localReadOk = 1;
            try {
                localX = readDoeCsvShardRowMajor(doeFile,
                                                 firstRow,
                                                 localRows,
                                                 dim);
            } catch (const std::exception& e) {
                localReadOk = 0;
                localError = e.what();
            }

            int allReadsOk = 0;
            MPI_Allreduce(&localReadOk,
                          &allReadsOk,
                          1,
                          MPI_INT,
                          MPI_MIN,
                          comm);
            if (!allReadsOk) {
                if (!localReadOk) {
                    std::cerr << "[rank " << worldRank
                              << "] DOE shard read failed: "
                              << localError << "\n";
                }
                MPI_Abort(comm, EXIT_FAILURE);
            }

            if (worldRank == 0) {
                const int smallestShard = nInitGlobal / worldSize;
                const int largestShard =
                    smallestShard +
                    (nInitGlobal % worldSize != 0 ? 1 : 0);
                std::cout
                    << "[DOE] every rank read its own Sobol shard from "
                    << "the node-local verified file; no DOE point "
                    << "communication\n"
                    << "[DOE] local rows/rank = "
                    << smallestShard << ".." << largestShard << "\n";
            }
            return localX;
        }

        std::vector<double> X;
        if (worldRank == 0) {
            X = readDoeCsvRowMajor(doeFile, nInitGlobal, dim);
        }
        return placeInitialDoe(comm,
                               worldRank,
                               worldSize,
                               std::move(X),
                               nInitGlobal,
                               dim,
                               fbo.doeDistribution);
    }

    const std::vector<double> lower = copyHostBounds(lowerBounds);
    const std::vector<double> upper = copyHostBounds(upperBounds);

    if (fbo.doeDistribution == FBODoeDistribution::Distributed) {
        const int localRows =
            doeRowsForRank(nInitGlobal, worldSize, worldRank);
        const int globalRowStart =
            doeRowStartForRank(nInitGlobal, worldSize, worldRank);
        std::vector<double> localX;

        if (fbo.doe == FBODoe::LatinHypercube) {
            localX = makeDeterministicLhsRowsHost(localRows,
                                                  nInitGlobal,
                                                  globalRowStart,
                                                  dim,
                                                  lower,
                                                  upper,
                                                  seed);
        } else if (fbo.doe == FBODoe::Halton ||
                   fbo.doe == FBODoe::Hammersley) {
            localX = makeLowDiscrepancyDoeHost(
                localRows,
                nInitGlobal,
                globalRowStart,
                dim,
                lower,
                upper,
                seed,
                fbo.doe == FBODoe::Hammersley);
        } else {
            localX = makeRandomDoeHost(localRows,
                                       dim,
                                       lower,
                                       upper,
                                       seed,
                                       globalRowStart);
        }

        if (worldRank == 0) {
            const int smallestShard = nInitGlobal / worldSize;
            const int largestShard =
                smallestShard + (nInitGlobal % worldSize != 0 ? 1 : 0);
            std::cout << "[DOE] generated distributed "
                      << toString(fbo.doe)
                      << " shards directly on each rank; "
                      << "no DOE point communication\n"
                      << "[DOE] local rows/rank = "
                      << smallestShard << ".." << largestShard << "\n";
        }
        return localX;
    }

    std::vector<double> X;

    if (worldRank == 0) {
        if (fbo.doe == FBODoe::LatinHypercube) {
            X = GP::latinHypercubeSample(nInitGlobal,
                                         dim,
                                         lowerBounds,
                                         upperBounds,
                                         static_cast<unsigned>(seed));
        } else if (fbo.doe == FBODoe::Halton ||
                   fbo.doe == FBODoe::Hammersley) {
            X = makeLowDiscrepancyDoeHost(
                nInitGlobal,
                nInitGlobal,
                0,
                dim,
                lower,
                upper,
                seed,
                fbo.doe == FBODoe::Hammersley);
        } else {
            X = makeRandomDoeHost(nInitGlobal, dim, lower, upper, seed);
        }
    }

    if (worldRank == 0) {
        std::cout << "[DOE] generated " << toString(fbo.doe)
                  << " design in-process\n";
    }

    return placeInitialDoe(comm,
                           worldRank,
                           worldSize,
                           std::move(X),
                           nInitGlobal,
                           dim,
                           fbo.doeDistribution);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    if (argc==1 || hasOptionalArg(argc,argv,"--help")) {
        std::cout << "Fractal Quantum Optimizer " << FQO_VERSION << "\n"
          << "Usage: fractal_optimizer BEAM DIMENSION BUDGET EXPENSE OBJECTIVE [options]\n"
          << "  --optimizer-family fbd|fbo   Select configurable family\n"
          << "  --list-objectives           Show objective names\n"
          << "  --result-json PATH          Save the final point/value as JSON\n"
          << "  --seed N --max-depth N --lower-bound L --upper-bound U\n"
          << "Python API: from fqo import minimize, Config\n"
          << "Component options: docs/configuration.md\n";
        return 0;
    }
    if (hasOptionalArg(argc,argv,"--version")) {
        std::cout << FQO_VERSION << '\n'; return 0;
    }
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    const bool have_thread_multiple = (provided == MPI_THREAD_MULTIPLE);

    auto level_str = [](int lvl) {
        switch (lvl) {
            case MPI_THREAD_SINGLE:
                return "MPI_THREAD_SINGLE";
            case MPI_THREAD_FUNNELED:
                return "MPI_THREAD_FUNNELED";
            case MPI_THREAD_SERIALIZED:
                return "MPI_THREAD_SERIALIZED";
            case MPI_THREAD_MULTIPLE:
                return "MPI_THREAD_MULTIPLE";
            default:
                return "UNKNOWN";
        }
    };

    int world_size = 1;
    int world_rank = 0;

    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank == 0) {
        std::cout << "[MPI] Requested MPI_THREAD_MULTIPLE, provided = "
                  << level_str(provided) << std::endl;
    }

    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD,
                        MPI_COMM_TYPE_SHARED,
                        world_rank,
                        MPI_INFO_NULL,
                        &node_comm);

    int node_rank = 0;
    int node_size = 1;

    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    // ------------------------------------------------------------------------
    // Keep your GPU selection behavior unchanged.
    // ------------------------------------------------------------------------
    int dev = 0;

    MPI_Comm node_leader_comm;

    if (node_rank == 0) {
        MPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, &node_leader_comm);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, world_rank, &node_leader_comm);
    }

    int num_nodes = 0;
    int node_id   = -1;

    if (node_rank == 0) {
        MPI_Comm_size(node_leader_comm, &num_nodes);
        MPI_Comm_rank(node_leader_comm, &node_id);
    }

    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, node_comm);
    MPI_Bcast(&node_id, 1, MPI_INT, 0, node_comm);

    if (hasOptionalArg(argc, argv, "--list-objectives")) {
        if (world_rank == 0) {
            std::cout << "Available objectives:\n";
            for (const auto& info : listObjectiveInfos()) {
                std::cout << "  " << info.name
                          << "  regime=" << info.regime
                          << "  bounds=[" << info.defaultLower
                          << ", " << info.defaultUpper << "]"
                          << "  additive="
                          << (info.additiveFastPath ? "yes" : "no")
                          << "  external="
                          << (info.external ? "yes" : "no")
                          << "\n";
            }
        }
        if (node_rank == 0) {
            MPI_Comm_free(&node_leader_comm);
        }
        MPI_Comm_free(&node_comm);
        MPI_Finalize();
        return 0;
    }

    const int beta = parseBeta(argc, argv, world_rank);
    int dimension  = parseDimension(argc, argv, world_rank);
    int budget     = parseBudget(argc, argv, world_rank);

    int expenseFactor = parseExpenseFactor(argc, argv, world_rank);
    if (dimension<=0 || budget<=0 || beta<=0) {
        if(world_rank==0) std::cerr << "Dimension, budget and beam must be positive.\n";
        MPI_Abort(MPI_COMM_WORLD,1);
    }

    const ObjectiveType objTy = parseObjectiveType(argc, argv, 5, world_rank);

    const double rho         = parseRho(argc, argv, world_rank);
    const int exploit_mult   = parseExploitMult(argc, argv, world_rank);
    const int exploit_budget = parseExploitBudget(argc, argv, world_rank);

    (void)exploit_mult;

    FBOConfig fboConfig;
    FBDConfig fbdConfig;
    OptimizerFamily optimizerFamily = OptimizerFamily::FBO;
    try {
        optimizerFamily = parseOptimizerFamily(argc, argv, world_rank);
        fboConfig = parseFBOConfig(argc, argv, world_rank);
        fbdConfig = parseFBDConfig(argc, argv, world_rank);
    } catch (...) {
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const bool useFBD = optimizerFamily == OptimizerFamily::FBD;

    if (useFBD) {
        if (fbdConfig.pointsPerCell <= 0) {
            fbdConfig.pointsPerCell =
                defaultFBDPointsPerCell(fbdConfig.exploration);
        }

        fboConfig.decomposition = fbdConfig.decomposition;
        fboConfig.splitPolicy = fbdConfig.splitPolicy;
        fboConfig.doe = fbdConfig.initialDesign;
        fboConfig.surrogate = FBOSurrogate::None;
        fboConfig.exploitation = fbdConfig.exploitation;
        fboConfig.hypercubeActiveDims = fbdConfig.hypercubeActiveDims;
        fboConfig.voronoiSites = fbdConfig.voronoiSites;
        fboConfig.simplexChildren = fbdConfig.simplexChildren;
    }

    double defaultLowerBound = objectiveDefaultLowerBound(objTy);
    double defaultUpperBound = objectiveDefaultUpperBound(objTy);

    if (objTy == ObjectiveType::External) {
        const auto module=parseOptionalStringArg(argc,argv,"--python-module","");
        if (!module.empty()) {
            const auto directory=parseOptionalStringArg(argc,argv,"--python-dir","");
            if (init_python_objective(directory.c_str(),module.c_str(),dimension)) {
                std::cerr << "Python objective initialization failed: " << vqe_last_error() << '\n';
                MPI_Abort(MPI_COMM_WORLD,1);
            }
            globalExternalObjective().fn=&vqe_eval_trampoline;
            globalExternalObjective().fn_many=&vqe_eval_many_trampoline;
        } else {
            if (world_size!=1 || !hasOptionalArg(argc,argv,"--objective-stdio")) {
                if(world_rank==0) std::cerr << "Use the fqo Python API for pipe callbacks (one rank), or --python-module for MPI jobs.\n";
                MPI_Abort(MPI_COMM_WORLD,1);
            }
            globalExternalObjective().fn=&fqo_stdio_one;
            globalExternalObjective().fn_many=&fqo_stdio_many;
        }
    }

    if (objTy == ObjectiveType::VQE) {
        const std::string molecule =
            parseOptionalStringArg(argc, argv, "--vqe-mol", "LiH");
        const std::string basis =
            parseOptionalStringArg(argc, argv, "--vqe-basis", "STO-3G");
        const std::string pythonDirectory =
            parseOptionalStringArg(argc, argv, "--vqe-python-dir", "");

        const int initStatus =
            init_vqe_bridge(pythonDirectory.c_str(),
                            molecule.c_str(),
                            basis.c_str());
        if (initStatus != 0 || !vqe_is_ready()) {
            if (world_rank == 0) {
                std::cerr
                    << "[Error] VQE bridge initialization failed for "
                    << molecule << "/" << basis << ": "
                    << vqe_last_error() << "\n";
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        const int vqeDimension = vqe_num_params();
        if (vqeDimension <= 0) {
            if (world_rank == 0) {
                std::cerr << "[Error] VQE bridge returned an invalid "
                          << "parameter count: " << vqeDimension << "\n";
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (world_rank == 0) {
            std::cout << "[VQE] molecule=" << molecule
                      << " basis=" << basis
                      << " parameters=" << vqeDimension << "\n";
            if (dimension != vqeDimension) {
                std::cout << "[VQE] overriding requested dimension "
                          << dimension << " with ansatz dimension "
                          << vqeDimension << "\n";
            }
        }
        dimension = vqeDimension;

        // Dimension-aware trust box used by the LUMI-G VQE path.
        constexpr double pi = 3.14159265358979323846;
        const double delta = 0.30;
        const double sigma =
            std::min(std::sqrt(3.0 / static_cast<double>(dimension)) * delta,
                     pi / 8.0);
        defaultLowerBound = -sigma;
        defaultUpperBound = sigma;

        auto& ext = globalExternalObjective();
        ext.fn = &vqe_eval_trampoline;
        ext.fn_many = &vqe_eval_many_trampoline;
    }

    const double objectiveLowerBound =
        parseOptionalDoubleArg(argc,
                               argv,
                               "--lower-bound",
                               defaultLowerBound);
    const double objectiveUpperBound =
        parseOptionalDoubleArg(argc,
                               argv,
                               "--upper-bound",
                               defaultUpperBound);

    if (!(objectiveLowerBound < objectiveUpperBound)) {
        if (world_rank == 0) {
            std::cerr << "[Error] invalid objective bounds: lower="
                      << objectiveLowerBound
                      << " upper=" << objectiveUpperBound << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    const FBODecomposition decompositionKind = fboConfig.decomposition;
    const bool useGPSurrogate = FBO::isGPSurrogate(fboConfig.surrogate);
    const bool useSimpleSurrogate =
        FBO::isSimpleSurrogate(fboConfig.surrogate);
    const bool useSurrogate = useGPSurrogate || useSimpleSurrogate;

    if (world_rank == 0) {
        std::cout << "\n[Optimizer]\n"
                  << "  family = " << toString(optimizerFamily) << "\n";
        printObjectiveInfo(objTy, dimension, std::cout);
        std::cout << "  active bounds = [" << objectiveLowerBound
                  << ", " << objectiveUpperBound << "]\n"
                  << std::endl;

        if (useFBD) {
            printFBDConfig(fbdConfig, std::cout);
        } else {
            printFBOConfig(fboConfig, std::cout);
        }
        std::cout << "[Fractal]\n"
                  << "  effective children/parent = "
                  << childCountForConfig(dimension, fboConfig) << "\n";
        if (fboConfig.decomposition == FBODecomposition::Hypercube) {
            std::cout << "  effective hypercube dims  = "
                      << fboActiveHypercubeDims(
                             dimension,
                             fboConfig.hypercubeActiveDims)
                      << "\n";
        }
        std::cout << std::endl;
    }

    if (usesExternalObjective(objTy) &&
        globalExternalObjective().fn == nullptr &&
        globalExternalObjective().fn_many == nullptr) {
        if (world_rank == 0) {
            std::cerr << "[Error] External objective is not registered in this executable.\n";
        }

        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

#if defined(KOKKOS_VERSION) && (KOKKOS_VERSION >= 40000)
    {
        Kokkos::InitializationSettings settings;
        settings.set_device_id(dev);
        Kokkos::initialize(settings);
    }
#else
    Kokkos::initialize(argc, argv);
#endif

    if (world_rank == 0) {
        std::cout << "[Kokkos] configuration:\n";
        Kokkos::print_configuration(std::cout);
    }

    {
#if defined(KOKKOS_ENABLE_CUDA)
        {
            int cur = -1;
            cudaGetDevice(&cur);

            std::cout << "[rank " << world_rank
                      << " | node_rank " << node_rank
                      << "] CUDA device = " << cur << std::endl;
        }
#elif defined(KOKKOS_ENABLE_HIP)
        {
            int cur = -1;
            hipGetDevice(&cur);

            const char* hipv  = std::getenv("HIP_VISIBLE_DEVICES");
            const char* rocrv = std::getenv("ROCR_VISIBLE_DEVICES");

            std::cout << "[rank " << world_rank
                      << " | node_rank " << node_rank
                      << "] HIP device = " << cur
                      << "  HIP_VISIBLE_DEVICES=" << (hipv ? hipv : "")
                      << "  ROCR_VISIBLE_DEVICES=" << (rocrv ? rocrv : "")
                      << std::endl;
        }
#endif

        auto start = std::chrono::high_resolution_clock::now();

        using ExecSpace  = Kokkos::DefaultExecutionSpace;
        using MemSpace   = ExecSpace::memory_space;
        using mem_space  = MemSpace;
        using exec_space = ExecSpace;

        PROFILE_SCOPE("initial_bounds");

        Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
            lowerBounds("lowerBounds", dimension);

        Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
            upperBounds("upperBounds", dimension);

        Kokkos::deep_copy(lowerBounds, objectiveLowerBound);
        Kokkos::deep_copy(upperBounds, objectiveUpperBound);

        SearchSpaceBounds searchSpace(dimension, lowerBounds, upperBounds);
        (void)searchSpace;

        // --------------------------------------------------------------------
        // Initial GP design.
        //
        // Important fix:
        // N < 2 breaks PCA covariance/Gram scaling and can lead to a bad GP.
        // --------------------------------------------------------------------
        const int reducedDimension = std::min(100, dimension);
const double sigma_f = fboConfig.gpSigmaF;
const double l = fboConfig.gpLengthScale;
const double sigma_n = fboConfig.gpNoise;
const FBOGPKernel gpKernelKind = fboConfig.gpKernel;
const double gpRQAlpha = fboConfig.gpRQAlpha;

// Same seed as Python benchmark.
const int seed = parseOptionalIntArg(argc, argv, "--seed", 0);

// Same n_init rule as Python sweep, unless explicitly overridden.
const bool nInitWasExplicit = hasOptionalArg(argc, argv, "--n-init");
int nInitGlobal = parseOptionalIntArg(
    argc,
    argv,
    "--n-init",
    defaultNInitForDim(dimension)
);

if (!nInitWasExplicit && fboConfig.budgetMode == FBOBudgetMode::FixedFractions) {
    double df = fboConfig.doeFraction >= 0.0 ? fboConfig.doeFraction : 0.25;
    double xf = fboConfig.exploreFraction >= 0.0 ? fboConfig.exploreFraction : 0.65;
    double ef = fboConfig.exploitFraction >= 0.0 ? fboConfig.exploitFraction : 0.10;
    if (df < 0.0) df = 0.0;
    if (xf < 0.0) xf = 0.0;
    if (ef < 0.0) ef = 0.0;
    const double fsum = df + xf + ef;
    if (fsum > 0.0) df /= fsum;
    nInitGlobal = static_cast<int>(
        std::llround(df * static_cast<double>(budget)));
}

if (useFBD && !nInitWasExplicit) {
    nInitGlobal = std::max(1, fbdConfig.initialPoints);
}

if (useGPSurrogate && nInitGlobal < 2) {
    nInitGlobal = std::min(budget, 2);
}

if (useSimpleSurrogate && nInitGlobal < 1) {
    nInitGlobal = std::min(budget, 1);
}

if (useGPSurrogate && nInitGlobal < 2) {
    if (world_rank == 0) {
        std::cerr << "[Error] n_init must be at least 2 for GP-based surrogates.\n";
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

if (useSimpleSurrogate && nInitGlobal < 1) {
    if (world_rank == 0) {
        std::cerr << "[Error] n_init must be at least 1 for "
                  << toString(fboConfig.surrogate)
                  << " surrogate.\n";
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

if (fboConfig.doeDistribution == FBODoeDistribution::Distributed) {
    const int smallestLocalDoe = nInitGlobal / world_size;
    const int requiredLocalDoe = useGPSurrogate
        ? 2
        : (useSimpleSurrogate ? 1 : 0);
    if (smallestLocalDoe < requiredLocalDoe) {
        if (world_rank == 0) {
            std::cerr
                << "[Error] distributed DOE gives some ranks only "
                << smallestLocalDoe << " points, but "
                << toString(fboConfig.surrogate) << " requires at least "
                << requiredLocalDoe << " local points per rank.\n"
                << "        Increase --n-init to at least "
                << requiredLocalDoe * world_size
                << " or use --fbo-doe-distribution replicated.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

const std::string doeFile = parseOptionalStringArg(argc, argv, "--doe-file", "");
const std::string ftCheckpointBase =
    parseOptionalStringArg(argc, argv, "--ft-checkpoint", "");
const std::string ftCheckpointReplicaBase =
    parseOptionalStringArg(argc, argv, "--ft-checkpoint-replica", "");
const std::string ftResumeBase =
    parseOptionalStringArg(argc, argv, "--ft-resume", "");
const bool ftAutoResume =
    hasOptionalArg(argc, argv, "--ft-auto-resume");
const bool ftElasticResume =
    hasOptionalArg(argc, argv, "--ft-elastic-resume");
const bool ftHandleSignals =
    !ftCheckpointBase.empty() &&
    !hasOptionalArg(argc, argv, "--ft-no-signal-handler");
const int ftStopAfterDepth =
    parseOptionalIntArg(argc, argv, "--ft-stop-after-depth", -1);
const int ftKeepCheckpoints =
    parseOptionalIntArg(argc, argv, "--ft-keep-checkpoints", 1);

if (ftKeepCheckpoints < 1) {
    if (world_rank == 0) {
        std::cerr << "[FT] --ft-keep-checkpoints must be >= 1.\n";
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}
const bool ftRollingCheckpoints = ftKeepCheckpoints > 1;

if (ftHandleSignals) {
    installFTSignalHandlers();
}

std::string ftResumeBaseEffective = ftResumeBase;
if (ftAutoResume && ftResumeBaseEffective.empty()) {
    if (ftCheckpointBase.empty()) {
        if (world_rank == 0) {
            std::cerr << "[FT] --ft-auto-resume requires --ft-checkpoint "
                      << "when --ft-resume is not provided.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    auto chooseUsableCheckpointBase =
        [&](const std::string& base, int* anyFound) -> std::string {
        if (anyFound) {
            *anyFound = 0;
        }
        if (base.empty()) {
            return "";
        }

        for (const auto& candidate : ftCheckpointCandidateBases(base)) {
            if (ftElasticResume) {
                int oldWorldSize = 0;
                std::string setReason;
                const int usableSet =
                    ftCheckpointSetUsable(candidate,
                                          &oldWorldSize,
                                          &setReason) ? 1 : 0;
                const int localExists =
                    ftRankCheckpointExists(candidate, 0) ? 1 : 0;

                int allAgreeUsable = 0;
                int anyExists = 0;
                MPI_Allreduce(&usableSet,
                              &allAgreeUsable,
                              1,
                              MPI_INT,
                              MPI_MIN,
                              MPI_COMM_WORLD);
                MPI_Allreduce(&localExists,
                              &anyExists,
                              1,
                              MPI_INT,
                              MPI_MAX,
                              MPI_COMM_WORLD);
                if (anyFound && anyExists) {
                    *anyFound = 1;
                }
                if (allAgreeUsable) {
                    return candidate;
                }
                continue;
            }

            const int localUsable =
                ftRankCheckpointUsable(candidate,
                                       world_rank,
                                       world_size,
                                       nullptr) ? 1 : 0;
            const int localExists =
                ftRankCheckpointExists(candidate, world_rank) ? 1 : 0;

            int allUsable = 0;
            int anyExists = 0;
            MPI_Allreduce(&localUsable,
                          &allUsable,
                          1,
                          MPI_INT,
                          MPI_MIN,
                          MPI_COMM_WORLD);
            MPI_Allreduce(&localExists,
                          &anyExists,
                          1,
                          MPI_INT,
                          MPI_MAX,
                          MPI_COMM_WORLD);

            if (anyFound && anyExists) {
                *anyFound = 1;
            }
            if (allUsable) {
                return candidate;
            }
        }
        return "";
    };

    int anyHasCheckpoint = 0;
    int replicaAnyHasCheckpoint = 0;
    const std::string primaryResumeBase =
        chooseUsableCheckpointBase(ftCheckpointBase, &anyHasCheckpoint);
    const std::string replicaResumeBase =
        chooseUsableCheckpointBase(ftCheckpointReplicaBase,
                                   &replicaAnyHasCheckpoint);

    if (!primaryResumeBase.empty()) {
        ftResumeBaseEffective = primaryResumeBase;
    } else if (!replicaResumeBase.empty()) {
        ftResumeBaseEffective = replicaResumeBase;
        if (world_rank == 0) {
            std::cout << "[FT] auto-resume falling back to checkpoint "
                      << "replica " << replicaResumeBase
                      << " because the primary set is missing or failed "
                      << "validation\n";
        }
    } else if (anyHasCheckpoint) {
        if (world_rank == 0) {
            std::cerr << "[FT] incomplete or corrupt checkpoint set for base "
                      << ftCheckpointBase
                      << "; refusing automatic resume.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    } else if (replicaAnyHasCheckpoint) {
        if (world_rank == 0) {
            std::cerr << "[FT] partial checkpoint replica set found for base "
                      << ftCheckpointReplicaBase
                      << "; refusing automatic resume.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    } else if (world_rank == 0) {
        std::cout << "[FT] auto-resume found no checkpoint; starting fresh.\n";
    }
}

if (world_rank == 0) {
    std::cout << "[DOE] n_init_global = " << nInitGlobal << "\n";
    std::cout << "[DOE] seed = " << seed << "\n";
    std::cout << "[DOE] execution = "
              << toString(fboConfig.doeDistribution) << "\n";
    if (!ftCheckpointBase.empty()) {
        std::cout << "[FT] checkpoint base = " << ftCheckpointBase << "\n";
    }
    if (!ftCheckpointReplicaBase.empty()) {
        std::cout << "[FT] checkpoint replica base = "
                  << ftCheckpointReplicaBase << "\n";
    }
    if (!ftResumeBaseEffective.empty()) {
        std::cout << "[FT] resume base = " << ftResumeBaseEffective << "\n";
    }
    if (ftAutoResume) {
        std::cout << "[FT] auto-resume = on\n";
    }
    if (ftElasticResume) {
        std::cout << "[FT] elastic resume = on\n";
    }
    if (ftRollingCheckpoints) {
        std::cout << "[FT] rolling checkpoints = on; keep last "
                  << ftKeepCheckpoints << " generations\n";
    }
    if (ftHandleSignals) {
        std::cout << "[FT] signal-aware checkpointing = on "
                  << "(SIGTERM/SIGINT";
#ifdef SIGUSR1
        std::cout << "/SIGUSR1";
#endif
        std::cout << ")\n";
    }
    if (ftStopAfterDepth > 0) {
        std::cout << "[FT] intentional stop after depth = "
                  << ftStopAfterDepth << "\n";
    }
}

const bool ftWillResume = !ftResumeBaseEffective.empty();

// Replicated mode gives every rank the complete DOE. Distributed mode gives
// each rank a deterministic, disjoint shard and therefore a local initial GP.
std::vector<double> trainingPoints;
if (!ftWillResume) {
    trainingPoints = makeInitialDoe(
        MPI_COMM_WORLD,
        world_rank,
        world_size,
        fboConfig,
        nInitGlobal,
        dimension,
        seed,
        doeFile,
        lowerBounds,
        upperBounds
    );
} else if (world_rank == 0) {
    std::cout << "[FT] skipping initial DOE/model build before resume\n";
}

const int numGPPoints = ftWillResume
    ? 0
    : static_cast<int>(trainingPoints.size() /
                       static_cast<size_t>(dimension));
const int pcaRankCap = std::min(100, dimension);

// Include replicated DOE work and the initial-center call on each rank in
// the physical evaluation cap. Reject an impossible initialization before
// any objective calls, while leaving FBD's structural stopping rule intact.
if (!useFBD && !ftWillResume) {
    const long long localInitialization = static_cast<long long>(numGPPoints) + 1;
    long long initialization = 0;
    MPI_Allreduce(&localInitialization, &initialization, 1, MPI_LONG_LONG,
                  MPI_SUM, MPI_COMM_WORLD);
    if (initialization >= budget) {
        if (world_rank == 0) {
            std::cerr << "[Error] FBO initialization requires " << initialization
                      << " evaluations, leaving no search budget within " << budget
                      << ". Reduce --n-init or the rank count, or use distributed DOE.\n";
        }
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}

if (!ftWillResume) {
    std::cout << "[rank " << world_rank
              << "] DOE local points = " << numGPPoints << "\n";
}

        GP::Model gpModel;
        GP::DeviceGPModel d_gpModel;
        Kokkos::View<double*, mem_space> d_mu_X;
        Kokkos::View<double*, mem_space> d_sd_X;
        Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> d_B;
        FBO::SimpleSurrogateDevice simpleSurrogateModel;
        std::vector<double> trainingY;

        if (!ftWillResume && useGPSurrogate && usesExternalObjective(objTy)) {
            GP::evalObjectiveBatchGPU(trainingPoints,
                                      numGPPoints,
                                      dimension,
                                      expenseFactor,
                                      objTy,
                                      trainingY);

            if (fboConfig.surrogate == FBOSurrogate::HsicGp) {
                gpModel =
                    GP::trainGP_HSIC_from_precomputedY(
                        trainingPoints,
                        trainingY,
                        numGPPoints,
                        dimension,
                        pcaRankCap,
                        0.95,
                        sigma_f,
                        l,
                        sigma_n,
                        "R2HSic",
                        "median",
                        gpKernelKind,
                        gpRQAlpha);
            } else {
                gpModel =
                    GP::trainGP_from_precomputedY(trainingPoints,
                                                  trainingY,
                                                  numGPPoints,
                                                  dimension,
                                                  pcaRankCap,
                                                  sigma_f,
                                                  l,
                                                  sigma_n,
                                                  expenseFactor,
                                                  objTy,
                                                  gpKernelKind,
                                                  gpRQAlpha);
            }

            PROFILE_SCOPE("GP::makeDeviceCopy");
            d_gpModel = GP::makeDeviceCopy(gpModel,
                                           sigma_f,
                                           l,
                                           sigma_n,
                                           gpKernelKind,
                                           gpRQAlpha);
            d_mu_X = d_gpModel.mu_X;
            d_sd_X = d_gpModel.sd_X;
            d_B = d_gpModel.B;
        } else if (!ftWillResume && useGPSurrogate) {
            if (fboConfig.surrogate == FBOSurrogate::HsicGp) {
                gpModel =
                    GP::trainGP_HSIC(trainingPoints,
                                     numGPPoints,
                                     dimension,
                                     pcaRankCap,
                                     0.95,
                                     sigma_f,
                                     l,
                                     sigma_n,
                                     expenseFactor,
                                     objTy,
                                     "R2HSic",
                                     "median",
                                     gpKernelKind,
                                     gpRQAlpha);
            } else {
                gpModel =
                    GP::trainGP(trainingPoints,
                                numGPPoints,
                                dimension,
                                pcaRankCap,
                                sigma_f,
                                l,
                                sigma_n,
                                expenseFactor,
                                objTy,
                                gpKernelKind,
                                gpRQAlpha);
            }

            PROFILE_SCOPE("GP::makeDeviceCopy");
            d_gpModel = GP::makeDeviceCopy(gpModel,
                                           sigma_f,
                                           l,
                                           sigma_n,
                                           gpKernelKind,
                                           gpRQAlpha);
            d_mu_X = d_gpModel.mu_X;
            d_sd_X = d_gpModel.sd_X;
            d_B = d_gpModel.B;
            trainingY = gpModel.y_phys;
        } else if (!ftWillResume && numGPPoints > 0) {
            GP::evalObjectiveBatchGPU(trainingPoints,
                                      numGPPoints,
                                      dimension,
                                      expenseFactor,
                                      objTy,
                                      trainingY);
        }

        if (!ftWillResume && useSimpleSurrogate) {
            PROFILE_SCOPE("FBO::makeSimpleSurrogateDevice");
            simpleSurrogateModel =
                FBO::makeSimpleSurrogateDevice(
                    trainingPoints,
                    trainingY,
                    dimension,
                    fboConfig.surrogate,
                    fboConfig.surrogateLengthScale,
                    fboConfig.idwPower);

            if (world_rank == 0) {
                std::cout << "[surrogate] simple model N="
                          << simpleSurrogateModel.N
                          << " length_scale="
                          << simpleSurrogateModel.lengthScale
                          << " idw_power="
                          << simpleSurrogateModel.idwPower
                          << "\n";
            }
        }

        Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
            initialCenter("initialCenter", dimension);

        Kokkos::parallel_for(
            "ComputeInitialCenter",
            Kokkos::RangePolicy<exec_space>(0, dimension),
            KOKKOS_LAMBDA(const int j) {
                initialCenter(j) = 0.5 * (objectiveLowerBound +
                                          objectiveUpperBound);
            });

        Kokkos::fence();

        const double initialRadius =
            0.5 * (objectiveUpperBound - objectiveLowerBound);

        const double fixedHypersphereFactor =
            fboRadiusFactor(FBODecomposition::Hypersphere);
        std::vector<double> alphaDepthHost(
            1,
            fixedHypersphereFactor);

        const double initialCenterObjective =
            computeObjectiveValue(initialCenter, expenseFactor, objTy);

        // --------------------------------------------------------------------
        // Initial incumbent.
        //
        // Main improvement:
        // make it truly global before using it for EI on all ranks.
        // --------------------------------------------------------------------
        PROFILE_SCOPE("init_best_objective_from_gp");

        int argmin = -1;
        double fmin = std::numeric_limits<double>::infinity();

        const int initialPointCount = ftWillResume
            ? 0
            : (useGPSurrogate
            ? gpModel.N
            : static_cast<int>(trainingY.size()));

        const std::vector<double>& initialY = useGPSurrogate
            ? gpModel.y_phys
            : trainingY;

        for (int i = 0; i < initialPointCount; ++i) {
            const double yi = initialY[i];

            if (yi < fmin) {
                fmin = yi;
                argmin = i;
            }
        }

        Kokkos::View<double*, Kokkos::HostSpace>
            bestPointHostLocal("bestPointHostLocal", dimension);

        if (argmin >= 0) {
            for (int d = 0; d < dimension; ++d) {
                bestPointHostLocal(d) =
                    trainingPoints[static_cast<size_t>(argmin) * dimension + d];
            }
        } else {
            for (int d = 0; d < dimension; ++d) {
                bestPointHostLocal(d) = 0.0;
            }
        }

        struct {
            double value;
            int rank;
        } localInitMin{fmin, world_rank}, globalInitMin;

        MPI_Allreduce(&localInitMin,
                      &globalInitMin,
                      1,
                      MPI_DOUBLE_INT,
                      MPI_MINLOC,
                      MPI_COMM_WORLD);

        Kokkos::View<double*, Kokkos::HostSpace>
            bestPointHost("bestPointHost", dimension);

        if (world_rank == globalInitMin.rank) {
            for (int d = 0; d < dimension; ++d) {
                bestPointHost(d) = bestPointHostLocal(d);
            }
        }

        MPI_Bcast(bestPointHost.data(),
                  dimension,
                  MPI_DOUBLE,
                  globalInitMin.rank,
                  MPI_COMM_WORLD);

        double bestObjectiveValue = globalInitMin.value;

        Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
            bestSolution("bestSolution", dimension);

        Kokkos::deep_copy(bestSolution, bestPointHost);

        HypersphereID initialID;
        Hypersphere initialHypersphere(dimension, 0, initialID);

        std::vector<Hypersphere> currentHyperspheres =
            decomposeFractalCell(initialHypersphere,
                                 dimension,
                                 fboConfig);

        const int maxDepth =
            parseOptionalIntArg(argc, argv, "--max-depth", 5);

        const double exploitFraction = fboConfig.exploitFraction;

        const double depthGamma =
            parseOptionalDoubleArg(argc, argv, "--depth-gamma", 1.0);

        BudgetPlan budgetPlan;
        if (useFBD) {
            budgetPlan.total = budget;
            budgetPlan.doe = nInitGlobal;
            budgetPlan.exploration = 0;
            budgetPlan.exploitation = std::numeric_limits<int>::max();
            budgetPlan.exploitFraction = exploitFraction;
            budgetPlan.depthGamma = depthGamma;
            budgetPlan.depthBudgets.assign(
                static_cast<size_t>(maxDepth),
                std::numeric_limits<int>::max());

            if (world_rank == 0) {
                std::cout << "\n[FBDPlan]\n";
                std::cout << "  legacy positional argument = " << budget
                          << " (ignored by FBD)\n";
                std::cout << "  initial points             = " << nInitGlobal
                          << "\n";
                std::cout << "  max depth                  = " << maxDepth
                          << "\n";
                std::cout << "  points per child fractal   = "
                          << fbdConfig.pointsPerCell << "\n";
                std::cout << "  tree-search frontier cap   = "
                          << (fbdConfig.beamWidth > 0 ? fbdConfig.beamWidth
                                                      : beta)
                          << "\n";
                std::cout << "  exploration policy         = exhaustive over "
                          << "all generated children\n";
            }
        } else {
            budgetPlan =
                (fboConfig.budgetMode == FBOBudgetMode::FixedFractions)
                    ? makeBudgetPlanFixedFractions(budget,
                                                   nInitGlobal,
                                                   nInitWasExplicit,
                                                   maxDepth,
                                                   exploit_budget,
                                                   fboConfig.doeFraction,
                                                   fboConfig.exploreFraction,
                                                   exploitFraction,
                                                   depthGamma,
                                                   world_rank)
                    : makeBudgetPlan(budget,
                                     nInitGlobal,
                                     maxDepth,
                                     exploit_budget,
                                     exploitFraction,
                                     depthGamma,
                                     world_rank);

            // Phase fractions are nominal allocations. Initialization also
            // evaluates the center (and may replicate DOE evaluations), so
            // exploration must fit the actual global remainder. Preserve the
            // original depth allocation whenever it already fits.
            const long long initialized = globalObjectiveEvaluations();
            const int available = static_cast<int>(std::max(0LL,
                static_cast<long long>(budget) - initialized));
            if (budgetPlan.exploration > available) {
                budgetPlan.exploration = available;
                budgetPlan.depthBudgets = splitBudgetByWeights(
                    available, makeDepthWeights(maxDepth, depthGamma));
            }
            if (world_rank == 0) {
                std::cout << "[Budget] actual initialization evaluations = "
                          << initialized << ", exploration cap = "
                          << budgetPlan.exploration << "\n";
            }
        }

        int activeMaxDepth = maxDepth;
        while (!useFBD &&
               activeMaxDepth > 0 &&
               budgetPlan.depthBudgets[activeMaxDepth - 1] <= 0) {
            --activeMaxDepth;
        }

        if (activeMaxDepth <= 0) {
            if (world_rank == 0) {
                std::cerr << "[Error] no active exploration depths after "
                          << "budget distribution.\n";
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        if (!useFBD && world_rank == 0 && activeMaxDepth != maxDepth) {
            std::cout << "[BudgetPlan] active max depth reduced to "
                      << activeMaxDepth
                      << " because later depths have zero budget.\n";
        }

        int firstDepth = 1;

        if (!ftResumeBaseEffective.empty()) {
            try {
                int checkpointWorldSize = world_size;
                std::vector<FTCheckpointState> elasticStates;

                if (ftElasticResume) {
                    if (!ftCheckpointSetUsable(ftResumeBaseEffective,
                                               &checkpointWorldSize,
                                               nullptr)) {
                        throw std::runtime_error(
                            "elastic checkpoint set failed validation");
                    }
                    elasticStates.reserve(
                        static_cast<size_t>(checkpointWorldSize));
                    for (int r = 0; r < checkpointWorldSize; ++r) {
                        elasticStates.push_back(
                            loadFTCheckpoint(ftResumeBaseEffective, r));
                    }
                }

                FTCheckpointState ckpt = ftElasticResume
                    ? elasticStates.front()
                    : loadFTCheckpoint(ftResumeBaseEffective, world_rank);

                auto validateCheckpointMetadata =
                    [&](const FTCheckpointState& state, int expectedRank) {
                    if (state.worldRank != expectedRank ||
                        state.dimension != dimension ||
                        state.budget != budget ||
                        state.seed != seed ||
                        state.nInit != nInitGlobal ||
                        state.optimizerFamily !=
                            static_cast<int>(optimizerFamily) ||
                        state.surrogate !=
                            static_cast<int>(fboConfig.surrogate) ||
                        state.doeDistribution !=
                            static_cast<int>(
                                fboConfig.doeDistribution)) {
                        throw std::runtime_error(
                            "checkpoint metadata does not match current run");
                    }
                    if (!ftElasticResume &&
                        state.worldSize != world_size) {
                        throw std::runtime_error(
                            "checkpoint world size does not match current run");
                    }
                    if (ftElasticResume &&
                        state.worldSize != checkpointWorldSize) {
                        throw std::runtime_error(
                            "checkpoint set has inconsistent world size");
                    }
                    if (static_cast<int>(state.bestPoint.size()) !=
                        dimension) {
                        throw std::runtime_error(
                            "checkpoint best point has wrong dimension");
                    }
                    const int n =
                        static_cast<int>(state.trainingY.size());
                    if (n > 0 &&
                        state.trainingPoints.size() !=
                            static_cast<size_t>(n) *
                                static_cast<size_t>(dimension)) {
                        throw std::runtime_error(
                            "checkpoint training archive shape is inconsistent");
                    }
                    if (state.completedDepth != ckpt.completedDepth) {
                        throw std::runtime_error(
                            "checkpoint set has inconsistent completed depth");
                    }
                };

                if (ftElasticResume) {
                    validateCheckpointMetadata(ckpt, 0);
                    for (int r = 1; r < checkpointWorldSize; ++r) {
                        validateCheckpointMetadata(elasticStates[r], r);
                    }

                    trainingPoints.clear();
                    trainingY.clear();

                    if (fboConfig.doeDistribution ==
                        FBODoeDistribution::Distributed) {
                        std::uint64_t archiveCounter = 0;
                        for (const auto& state : elasticStates) {
                            const int stateN =
                                static_cast<int>(state.trainingY.size());
                            for (int i = 0; i < stateN; ++i) {
                                if (static_cast<int>(
                                        archiveCounter %
                                        static_cast<std::uint64_t>(
                                            world_size)) == world_rank) {
                                    const size_t src =
                                        static_cast<size_t>(i) *
                                        static_cast<size_t>(dimension);
                                    trainingPoints.insert(
                                        trainingPoints.end(),
                                        state.trainingPoints.begin() + src,
                                        state.trainingPoints.begin() + src +
                                            dimension);
                                    trainingY.push_back(state.trainingY[i]);
                                }
                                ++archiveCounter;
                            }
                        }
                    } else {
                        const int initCount = std::min(
                            nInitGlobal,
                            static_cast<int>(ckpt.trainingY.size()));
                        trainingPoints.reserve(
                            static_cast<size_t>(initCount) *
                            static_cast<size_t>(dimension));
                        trainingY.reserve(static_cast<size_t>(initCount));
                        for (int i = 0; i < initCount; ++i) {
                            const size_t src =
                                static_cast<size_t>(i) *
                                static_cast<size_t>(dimension);
                            trainingPoints.insert(
                                trainingPoints.end(),
                                ckpt.trainingPoints.begin() + src,
                                ckpt.trainingPoints.begin() + src +
                                    dimension);
                            trainingY.push_back(ckpt.trainingY[i]);
                        }

                        std::uint64_t postInitCounter = 0;
                        for (const auto& state : elasticStates) {
                            const int stateN =
                                static_cast<int>(state.trainingY.size());
                            const int firstPost =
                                std::min(nInitGlobal, stateN);
                            for (int i = firstPost; i < stateN; ++i) {
                                if (static_cast<int>(
                                        postInitCounter %
                                        static_cast<std::uint64_t>(
                                            world_size)) == world_rank) {
                                    const size_t src =
                                        static_cast<size_t>(i) *
                                        static_cast<size_t>(dimension);
                                    trainingPoints.insert(
                                        trainingPoints.end(),
                                        state.trainingPoints.begin() + src,
                                        state.trainingPoints.begin() + src +
                                            dimension);
                                    trainingY.push_back(state.trainingY[i]);
                                }
                                ++postInitCounter;
                            }
                        }
                    }

                    auto bestIt = std::min_element(
                        elasticStates.begin(),
                        elasticStates.end(),
                        [](const FTCheckpointState& a,
                           const FTCheckpointState& b) {
                            return a.bestObjective < b.bestObjective;
                        });
                    ckpt.bestObjective = bestIt->bestObjective;
                    ckpt.bestPoint = bestIt->bestPoint;
                } else {
                    validateCheckpointMetadata(ckpt, world_rank);
                    trainingPoints = std::move(ckpt.trainingPoints);
                    trainingY = std::move(ckpt.trainingY);
                }

                const int resumeArchiveN =
                    static_cast<int>(trainingY.size());

                if (useGPSurrogate) {
                    if (resumeArchiveN < 2) {
                        throw std::runtime_error(
                            "checkpoint has too few points for GP resume");
                    }

                    if (fboConfig.surrogate == FBOSurrogate::HsicGp) {
                        gpModel = GP::trainGP_HSIC_from_precomputedY(
                            trainingPoints,
                            trainingY,
                            resumeArchiveN,
                            dimension,
                            pcaRankCap,
                            0.95,
                            sigma_f,
                            l,
                            sigma_n,
                            "R2HSic",
                            "median",
                            gpKernelKind,
                            gpRQAlpha);
                    } else {
                        gpModel = GP::trainGP_from_precomputedY(
                            trainingPoints,
                            trainingY,
                            resumeArchiveN,
                            dimension,
                            pcaRankCap,
                            sigma_f,
                            l,
                            sigma_n,
                            expenseFactor,
                            objTy,
                            gpKernelKind,
                            gpRQAlpha);
                    }

                    d_gpModel = GP::makeDeviceCopy(gpModel,
                                                   sigma_f,
                                                   l,
                                                   sigma_n,
                                                   gpKernelKind,
                                                   gpRQAlpha);
                    d_mu_X = d_gpModel.mu_X;
                    d_sd_X = d_gpModel.sd_X;
                    d_B = d_gpModel.B;
                    trainingY = gpModel.y_phys;
                } else if (useSimpleSurrogate) {
                    if (resumeArchiveN < 1) {
                        throw std::runtime_error(
                            "checkpoint has too few points for simple surrogate resume");
                    }
                    simpleSurrogateModel =
                        FBO::makeSimpleSurrogateDevice(
                            trainingPoints,
                            trainingY,
                            dimension,
                            fboConfig.surrogate,
                            fboConfig.surrogateLengthScale,
                            fboConfig.idwPower);
                }

                currentHyperspheres = std::move(ckpt.frontier);
                bestObjectiveValue = ckpt.bestObjective;
                for (int d = 0; d < dimension; ++d) {
                    bestPointHost(d) = ckpt.bestPoint[d];
                }
                Kokkos::deep_copy(bestSolution, bestPointHost);

                firstDepth = ckpt.completedDepth + 1;

                if (world_rank == 0) {
                    std::cout << "[FT] resumed checkpoint after depth "
                              << ckpt.completedDepth
                              << " with local archive N="
                              << trainingY.size()
                              << " and frontier="
                              << currentHyperspheres.size() << "\n";
                    if (ftElasticResume) {
                        std::cout << "[FT] elastic resume repartitioned "
                                  << "checkpoint from world_size "
                                  << checkpointWorldSize << " to "
                                  << world_size << "\n";
                    }
                }
            } catch (const std::exception& e) {
                if (world_rank == 0) {
                    std::cerr << "[FT] resume failed: " << e.what() << "\n";
                }
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
        }

        if (firstDepth > activeMaxDepth) {
            if (world_rank == 0) {
                std::cerr << "[FT] checkpoint is already past the active "
                          << "exploration depths; rerun without --ft-resume "
                          << "or resume from an earlier checkpoint.\n";
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }

        int maxIterations = 1;

        double stepSize  = 1.75;
        double phi       = 0.5;
        double omega_min = 1e-20;

        bool ftRequestedStop = false;
        long long fbdLocalExplorationEvaluations = 0;

        for (int depth = firstDepth; depth <= activeMaxDepth; ++depth) {
            double depthLocalBestVal =
                std::numeric_limits<double>::infinity();

            HypersphereID depthLocalBestID;
            bool depthLocalBestValid = false;

            PROFILE_SCOPE("depth_loop");

            {
                std::ostringstream oss;
                oss << "depth " << depth << " (max across ranks)";
                const std::string lbl = oss.str();

                ScopedSyncTimer _vts(lbl.c_str(), MPI_COMM_WORLD);

                Kokkos::View<double*, mem_space>
                    alphaDepthDev("alphaDepthDev",
                                  alphaDepthHost.size());
                {
                    auto hAlpha =
                        Kokkos::create_mirror_view(alphaDepthDev);
                    for (size_t i = 0; i < alphaDepthHost.size(); ++i) {
                        hAlpha(static_cast<int>(i)) = alphaDepthHost[i];
                    }
                    Kokkos::deep_copy(alphaDepthDev, hAlpha);
                }

                // ------------------------------------------------------------
                // Process virtual children of current frontier.
                // ------------------------------------------------------------
                const int64_t children_per_parent =
                    static_cast<int64_t>(
                        childCountForConfig(dimension, fboConfig));

                const int64_t numHyperspheres =
                    static_cast<int64_t>(currentHyperspheres.size()) *
                    children_per_parent;

                const int64_t perNode =
                    numHyperspheres / static_cast<int64_t>(num_nodes);

                const int64_t remNode =
                    numHyperspheres % static_cast<int64_t>(num_nodes);

                const int64_t startIdx =
                    static_cast<int64_t>(node_id) * perNode +
                    std::min<int64_t>(node_id, remNode);

                const int64_t endIdx =
                    startIdx + perNode + (node_id < remNode ? 1 : 0);

                const int64_t localNumHyperspheres64 = endIdx - startIdx;

                const int64_t perRank =
                    localNumHyperspheres64 / static_cast<int64_t>(node_size);

                const int64_t remRank =
                    localNumHyperspheres64 % static_cast<int64_t>(node_size);

                const int64_t localStartIdx64 =
                    startIdx +
                    static_cast<int64_t>(node_rank) * perRank +
                    std::min<int64_t>(node_rank, remRank);

                const int64_t localEndIdx64 =
                    localStartIdx64 + perRank + (node_rank < remRank ? 1 : 0);

                const size_t myChildBegin = static_cast<size_t>(localStartIdx64);
                const size_t myChildEnd   = static_cast<size_t>(localEndIdx64);

                size_t totalHyperspheres =
                    static_cast<size_t>(localEndIdx64 - localStartIdx64);

                (void)myChildEnd;

                auto die = [&](const char* why) {
                    std::cerr << "[rank " << world_rank << "] "
                              << why << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, 1);
                };

                if (numHyperspheres < 0) {
                    die("numHyperspheres overflowed (<0)");
                }

                if (localEndIdx64 < localStartIdx64) {
                    die("child range inverted");
                }

                size_t available_memory = get_available_memory();
                const int latentDimForMemory = useGPSurrogate
                    ? std::max(1, gpModel.d)
                    : (useSimpleSurrogate ? std::max(1, dimension) : 1);

                size_t per_hypersphere_memory =
                    static_cast<size_t>(dimension) * sizeof(double) +
                    sizeof(double) +
                    2ull * static_cast<size_t>(dimension) * sizeof(double) +
                    static_cast<size_t>(latentDimForMemory) * sizeof(double) +
                    3ull * sizeof(double) +
                    sizeof(bool);

                size_t max_num_hyperspheres =
                    static_cast<size_t>(available_memory * 0.9) /
                    std::max<size_t>(per_hypersphere_memory, 1);

                if (max_num_hyperspheres == 0) {
                    max_num_hyperspheres = 1;
                }

                if (world_rank == 0) {
                    std::cout << "[diag] parents=" << currentHyperspheres.size()
                              << " children_per_parent=" << children_per_parent
                              << " numHyperspheres(global)="
                              << static_cast<unsigned long long>(numHyperspheres)
                              << " world_size=" << world_size
                              << " numLocalHyperspheres="
                              << static_cast<unsigned long long>(totalHyperspheres)
                              << " beta=" << beta
                              << " rho=" << rho << '\n';
                }

                if (useFBD) {
                    const long long localChildren =
                        static_cast<long long>(totalHyperspheres);
                    const long long probes =
                        static_cast<long long>(
                            std::max(1, fbdConfig.pointsPerCell));
                    fbdLocalExplorationEvaluations += localChildren * probes;
                }

                int budget_per_rank = std::numeric_limits<int>::max();
                if (!useFBD) {
                    const int per_depth_budget =
                        budgetPlan.depthBudgets[depth - 1];
                    const int base_budget  = per_depth_budget / world_size;
                    const int extra_budget = per_depth_budget % world_size;
                    budget_per_rank =
                        base_budget + (world_rank < extra_budget ? 1 : 0);
                }

                int remaining = budget_per_rank;

                Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> d_Xstar;
                Kokkos::View<double*, mem_space> d_mu;
                Kokkos::View<double*, mem_space> d_var;
                Kokkos::View<double*, mem_space> d_acq;
                Kokkos::View<double*, mem_space> d_fbd_scores;
                Kokkos::View<double*, mem_space> d_fbd_best_values;
                Kokkos::View<int*, mem_space> d_fbd_best_probes;
                Kokkos::View<double*, mem_space> d_fx;
                Kokkos::View<int*, mem_space> d_topIdx;
                Kokkos::View<char*, mem_space> d_taken;

                std::vector<double> points_to_add_accum;
                std::vector<double> y_to_add_accum;

                bool did_update_this_depth = false;
                bool progress_thread_active = false;

                struct Pair {
                    double ei;
                    int64_t gidx;
                };

                const size_t free_b = get_available_memory();
                const size_t bytes_per_entry = sizeof(double) + sizeof(int64_t);

                size_t M_cap =
                    static_cast<size_t>(0.20 * static_cast<double>(free_b)) /
                    std::max<size_t>(bytes_per_entry, 1);

                const int frontierCap =
                    useFBD && fbdConfig.beamWidth > 0 ? fbdConfig.beamWidth
                                                       : beta;

                if (M_cap < static_cast<size_t>(frontierCap)) {
                    M_cap = static_cast<size_t>(frontierCap);
                }

                size_t M =
                    std::min(M_cap, static_cast<size_t>(frontierCap) * 16u);

                auto cmp = [](const Pair& a, const Pair& b) {
                    return a.ei > b.ei;
                };

                std::priority_queue<Pair, std::vector<Pair>, decltype(cmp)>
                    topM(cmp);

                std::vector<double> fbdAllWeights;
                std::vector<int64_t> fbdAllIndices;
                if (useFBD) {
                    const size_t reserveN =
                        std::min(totalHyperspheres,
                                 static_cast<size_t>(1000000));
                    fbdAllWeights.reserve(reserveN);
                    fbdAllIndices.reserve(reserveN);
                }

                double totalEI_stream = 0.0;
                double bestEI_stream  = 0.0;

                // ------------------------------------------------------------
                // Precompute PCA sparse-move projection helpers.
                // T(d,j) = B(d,j) / sd(d)
                // base_r = ((initialCenter - mu) / sd)^T B
                // ------------------------------------------------------------
                const int r_latent = useGPSurrogate ? d_gpModel.d_latent : 1;

                Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> T;
                Kokkos::View<double*, mem_space> base_r;

                if (useGPSurrogate) {
                    T = Kokkos::View<double**, Kokkos::LayoutLeft, mem_space>("T", dimension, r_latent);
                    using MD2 =
                        Kokkos::MDRangePolicy<Kokkos::Rank<2>, exec_space>;

                    Kokkos::parallel_for(
                        "prep_T",
                        MD2({0, 0}, {dimension, r_latent}),
                        KOKKOS_LAMBDA(int d, int j) {
                            T(d, j) = d_B(d, j) / d_sd_X(d);
                        });

                    base_r = Kokkos::View<double*, mem_space>("base_r", r_latent);

                    Kokkos::View<double*, mem_space> v("v", dimension);

                    Kokkos::parallel_for(
                        "prep_v",
                        Kokkos::RangePolicy<exec_space>(0, dimension),
                        KOKKOS_LAMBDA(int d) {
                            v(d) = (initialCenter(d) - d_mu_X(d)) / d_sd_X(d);
                        });

                    Kokkos::parallel_for(
                        "prep_base_r",
                        Kokkos::RangePolicy<exec_space>(0, r_latent),
                        KOKKOS_LAMBDA(int j) {
                            double s = 0.0;

                            for (int d = 0; d < dimension; ++d) {
                                s += v(d) * d_B(d, j);
                            }

                            base_r(j) = s;
                        });

                    Kokkos::fence();
                }

                Kokkos::View<double*,
                             Kokkos::LayoutLeft,
                             mem_space>
                    fbdScoreBestSolution("fbdScoreBestSolution", dimension);
                const double fbdScoreBestObjective = bestObjectiveValue;
                if (useFBD) {
                    Kokkos::deep_copy(fbdScoreBestSolution, bestSolution);
                }

                // ------------------------------------------------------------
                // Chunk loop.
                // ------------------------------------------------------------
                for (size_t chunk_start = 0;
                     chunk_start < totalHyperspheres &&
                     (useFBD || remaining > 0);
                     chunk_start += max_num_hyperspheres) {
                    PROFILE_SCOPE("chunk_total");

                    const size_t chunk_size =
                        std::min(max_num_hyperspheres,
                                 totalHyperspheres - chunk_start);

                    PROFILE_SCOPE("copy_chunk_hyperspheres");

                    const size_t childStart = myChildBegin + chunk_start;
                    const size_t childCount = chunk_size;

                    Kokkos::View<int*, mem_space> depths_chunk;
                    Kokkos::View<size_t*, mem_space> idOffsets_chunk;
                    Kokkos::View<int*, mem_space> idDims_chunk;
                    Kokkos::View<int*, mem_space> idSigns_chunk;

                    prepareChildChunkFromParents<mem_space>(
                        currentHyperspheres,
                        childStart,
                        childCount,
                        dimension,
                        fboConfig,
                        depths_chunk,
                        idOffsets_chunk,
                        idDims_chunk,
                        idSigns_chunk);

                    const int numCandidates = static_cast<int>(childCount);
                    const int B =
                        useFBD ? numCandidates : std::min(remaining, numCandidates);

                    if (B <= 0) break;

                    // --------------------------------------------------------
                    // PCA projection using sparse moves.
                    // --------------------------------------------------------
                    if (useGPSurrogate) {
                        PROFILE_SCOPE("PCA:projection (sparse-move)");
                        Kokkos::fence();

                        Kokkos::realloc(d_Xstar,
                                        numCandidates,
                                        d_gpModel.d_latent);

                        using team_policy = Kokkos::TeamPolicy<exec_space>;
                        using member_type = team_policy::member_type;

                        Kokkos::parallel_for(
                            "proj_sparse_moves",
                            team_policy(numCandidates, Kokkos::AUTO),
                            KOKKOS_LAMBDA(const member_type& team) {
                                const int i = team.league_rank();

                                Kokkos::parallel_for(
                                    Kokkos::TeamThreadRange(team,
                                                            d_gpModel.d_latent),
                                    [&](int j) {
                                        d_Xstar(i, j) = base_r(j);
                                    });

                                team.team_barrier();

                                const int depth_i = depths_chunk(i);
                                const size_t st   = idOffsets_chunk(i);

                                for (int dd = 0; dd < depth_i; ++dd) {
                                    const int dim = idDims_chunk(st + dd);
                                    const double off =
                                        fboPathMoveOffset(initialRadius,
                                                          decompositionKind,
                                                          idDims_chunk,
                                                          idSigns_chunk,
                                                          st,
                                                          dd,
                                                          alphaDepthDev);

                                    Kokkos::parallel_for(
                                        Kokkos::TeamThreadRange(
                                            team,
                                            d_gpModel.d_latent),
                                        [&](int j) {
                                            d_Xstar(i, j) +=
                                                off *
                                                T(dim, j);
                                        });

                                    team.team_barrier();
                                }
                            });

                        Kokkos::fence();
                    }

                    // --------------------------------------------------------
                    // Surrogate prediction.
                    // --------------------------------------------------------
                    if (useSurrogate) {
                        Kokkos::fence();

                        Kokkos::realloc(d_mu, numCandidates);
                        Kokkos::realloc(d_var, numCandidates);

                        if (useGPSurrogate) {
                            PROFILE_SCOPE("GP:predict (device)");
                            GP::batchPredictGP(d_gpModel,
                                               d_Xstar,
                                               d_mu,
                                               d_var);
                        } else {
                            PROFILE_SCOPE("simple_surrogate:predict");

                            Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> candidateCenters;
                            Kokkos::View<double*, mem_space> candidateRadii;

                            reconstructCentersAndRadii(
                                static_cast<size_t>(numCandidates),
                                dimension,
                                initialRadius,
                                depths_chunk,
                                idOffsets_chunk,
                                idDims_chunk,
                                idSigns_chunk,
                                initialCenter,
                                alphaDepthDev,
                                candidateCenters,
                                candidateRadii,
                                decompositionKind);

                            FBO::predictSimpleSurrogate(simpleSurrogateModel,
                                                        candidateCenters,
                                                        d_mu,
                                                        d_var);
                        }

                        Kokkos::fence();
                    }

                    if (useFBD) {
                        PROFILE_SCOPE("FBD:score_cells");
                        Kokkos::fence();

                        Kokkos::View<double*, mem_space>
                            alphaDepthDev("alphaDepthDev_fbd", 1);
                        auto hA = Kokkos::create_mirror_view(alphaDepthDev);
                        hA(0) = 1.0 / (1.0 + std::sqrt(2.0));
                        Kokkos::deep_copy(alphaDepthDev, hA);

                        Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> fbdCenters;
                        Kokkos::View<double*, mem_space> fbdRadii;

                        reconstructCentersAndRadii(
                            static_cast<size_t>(numCandidates),
                            dimension,
                            initialRadius,
                            depths_chunk,
                            idOffsets_chunk,
                            idDims_chunk,
                            idSigns_chunk,
                            initialCenter,
                            alphaDepthDev,
                            fbdCenters,
                            fbdRadii,
                            decompositionKind);

                        computeFBDScores(numCandidates,
                                         dimension,
                                         fbdCenters,
                                         fbdRadii,
                                         fbdScoreBestSolution,
                                         fbdScoreBestObjective,
                                         objectiveLowerBound,
                                         objectiveUpperBound,
                                         fbdConfig,
                                         seed,
                                         depth,
                                         expenseFactor,
                                         objTy,
                                         d_fbd_scores,
                                         d_fbd_best_values,
                                         d_fbd_best_probes);

                        MinAcqIndex bestFbdProbe;
                        Kokkos::parallel_reduce(
                            "fbd_min_probe_value",
                            Kokkos::RangePolicy<exec_space>(0, numCandidates),
                            KOKKOS_LAMBDA(int i, MinAcqIndex& upd) {
                                const double v = d_fbd_best_values(i);
                                if (v < upd.value) {
                                    upd.value = v;
                                    upd.index = i;
                                }
                            },
                            Kokkos::Min<MinAcqIndex>(bestFbdProbe));

                        if (bestFbdProbe.index >= 0) {
                            auto h_bestProbes =
                                Kokkos::create_mirror_view_and_copy(
                                    Kokkos::HostSpace(),
                                    d_fbd_best_probes);

                            const int srcIdx = bestFbdProbe.index;
                            const int probe = h_bestProbes(srcIdx);
                            const size_t gChild =
                                childStart + static_cast<size_t>(srcIdx);

                            Hypersphere hs =
                                makeChildHypersphereFromGlobal(gChild,
                                                               currentHyperspheres,
                                                               dimension,
                                                               fboConfig);

                            if (bestFbdProbe.value < depthLocalBestVal) {
                                depthLocalBestVal = bestFbdProbe.value;
                                depthLocalBestID = hs.id;
                                depthLocalBestValid = true;
                            }

                            if (bestFbdProbe.value < bestObjectiveValue) {
                                auto h_fbdRadii =
                                    Kokkos::create_mirror_view_and_copy(
                                        Kokkos::HostSpace(),
                                        fbdRadii);

                                auto x_best =
                                    reconstructCenter(hs,
                                                      initialCenter,
                                                      initialRadius,
                                                      alphaDepthHost,
                                                      decompositionKind);

                                const FBDExploration exploration =
                                    fbdConfig.exploration;
                                const int seedForProbe = seed;
                                const int depthForProbe = depth;
                                const double bestRadius = h_fbdRadii(srcIdx);
                                const double lb = objectiveLowerBound;
                                const double ub = objectiveUpperBound;

                                Kokkos::parallel_for(
                                    "apply_fbd_best_probe_offset",
                                    Kokkos::RangePolicy<exec_space>(0,
                                                                    dimension),
                                    KOKKOS_LAMBDA(int d) {
                                        double x = x_best(d) +
                                            fbdProbeOffset(exploration,
                                                           probe,
                                                           srcIdx,
                                                           d,
                                                           dimension,
                                                           bestRadius,
                                                           seedForProbe,
                                                           depthForProbe);
                                        if (x < lb) x = lb;
                                        if (x > ub) x = ub;
                                        x_best(d) = x;
                                    });
                                Kokkos::fence();

                                Kokkos::deep_copy(bestSolution, x_best);
                                bestObjectiveValue = bestFbdProbe.value;
                            }
                        }

                        Kokkos::fence();
                    }

                    // --------------------------------------------------------
                    // EI stream accumulation.
                    // --------------------------------------------------------
                    {
                        PROFILE_SCOPE("ei_chunk_compute");
                        Kokkos::fence();

                        Kokkos::View<double*, mem_space>
                            d_ei_chunk("d_ei_chunk", numCandidates);

                        if (useFBD) {
                            const std::vector<double> fbdWeights =
                                fbdWeightsFromScores(d_fbd_scores);

                            for (int i = 0; i < numCandidates; ++i) {
                                const double ei =
                                    fbdWeights[static_cast<size_t>(i)];

                                totalEI_stream += ei;

                                if (ei > bestEI_stream) {
                                    bestEI_stream = ei;
                                }

                                const int64_t gChild =
                                    static_cast<int64_t>(myChildBegin) +
                                    static_cast<int64_t>(chunk_start) +
                                    static_cast<int64_t>(i);

                                fbdAllWeights.push_back(ei);
                                fbdAllIndices.push_back(gChild);
                            }
                        } else if (useSurrogate) {
                            const double mu_y  =
                                useGPSurrogate ? gpModel.mu_y : 0.0;
                            const double sd_y  =
                                useGPSurrogate ? gpModel.sd_y : 1.0;
                            const double fstar = bestObjectiveValue;
                            const bool gpScale = useGPSurrogate;

                            Kokkos::parallel_for(
                                "ei_chunk",
                                Kokkos::RangePolicy<exec_space>(0, numCandidates),
                                KOKKOS_LAMBDA(int i) {
                                    const double mu_p = gpScale
                                        ? d_mu(i) * sd_y + mu_y
                                        : d_mu(i);

                                    const double v =
                                        d_var(i) > 1e-12 ? d_var(i) : 1e-12;
                                    const double sigma_p =
                                        sqrt(v) * sd_y;

                                    double ei = 0.0;

                                    if (sigma_p > 1e-12) {
                                        const double z =
                                            (fstar - mu_p) / sigma_p;

                                        const double cdf =
                                            0.5 *
                                            (1.0 + erf(z / sqrt(2.0)));

                                        const double pdf =
                                            0.3989422804014327 *
                                            exp(-0.5 * z * z);

                                        ei = (fstar - mu_p) * cdf +
                                             sigma_p * pdf;

                                        if (ei < 0.0) ei = 0.0;
                                    }

                                    d_ei_chunk(i) = ei;
                                });
                        } else {
                            const uint64_t seed64 =
                                static_cast<uint64_t>(seed + 1) *
                                0x9E3779B97F4A7C15ull;
                            const int depthForHash = depth;
                            const size_t childBeginForHash =
                                myChildBegin + chunk_start;

                            Kokkos::parallel_for(
                                "nosurrogate_ei_chunk",
                                Kokkos::RangePolicy<exec_space>(0, numCandidates),
                                KOKKOS_LAMBDA(int i) {
                                    uint64_t x =
                                        static_cast<uint64_t>(childBeginForHash +
                                                             static_cast<size_t>(i));
                                    x ^= seed64 +
                                         static_cast<uint64_t>(depthForHash) *
                                             0xBF58476D1CE4E5B9ull;
                                    x ^= x >> 30;
                                    x *= 0xBF58476D1CE4E5B9ull;
                                    x ^= x >> 27;
                                    x *= 0x94D049BB133111EBull;
                                    x ^= x >> 31;
                                    const double u =
                                        static_cast<double>(x & 0x1FFFFFFFFFFFFFull) /
                                        static_cast<double>(0x20000000000000ull);
                                    d_ei_chunk(i) = 1e-12 + u;
                                });
                        }

                        if (!useFBD) {
                            auto h_ei_chunk =
                                Kokkos::create_mirror_view_and_copy(
                                    Kokkos::HostSpace(),
                                    d_ei_chunk);

                            for (int i = 0; i < numCandidates; ++i) {
                                const double ei = h_ei_chunk(i);

                                totalEI_stream += ei;

                                if (ei > bestEI_stream) {
                                    bestEI_stream = ei;
                                }

                                const int64_t gChild =
                                    static_cast<int64_t>(myChildBegin) +
                                    static_cast<int64_t>(chunk_start) +
                                    static_cast<int64_t>(i);

                                if (topM.size() < M) {
                                    topM.push({ei, gChild});
                                } else if (ei > topM.top().ei) {
                                    topM.pop();
                                    topM.push({ei, gChild});
                                }
                            }
                        }

                        Kokkos::fence();
                    }

                    if (useFBD) {
                        continue;
                    }

                    // --------------------------------------------------------
                    // LCB acquisition and top-B selection.
                    // --------------------------------------------------------
                    const double kappa = 2.0;

                    Kokkos::realloc(d_acq, numCandidates);

                    {
                        PROFILE_SCOPE("acq_kernel");
                        Kokkos::fence();

                        if (useFBD) {
                            Kokkos::deep_copy(d_acq, d_fbd_scores);
                        } else if (useSurrogate) {
                            Kokkos::parallel_for(
                                "acq",
                                Kokkos::RangePolicy<exec_space>(0, numCandidates),
                                KOKKOS_LAMBDA(int i) {
                                    d_acq(i) =
                                        d_mu(i) - kappa * sqrt(d_var(i));
                                });
                        } else {
                            const uint64_t seed64 =
                                static_cast<uint64_t>(seed + 17) *
                                0x9E3779B97F4A7C15ull;
                            const int depthForHash = depth;
                            const size_t childBeginForHash =
                                myChildBegin + chunk_start;

                            Kokkos::parallel_for(
                                "nosurrogate_acq",
                                Kokkos::RangePolicy<exec_space>(0, numCandidates),
                                KOKKOS_LAMBDA(int i) {
                                    uint64_t x =
                                        static_cast<uint64_t>(childBeginForHash +
                                                             static_cast<size_t>(i));
                                    x ^= seed64 +
                                         static_cast<uint64_t>(depthForHash) *
                                             0xD6E8FEB86659FD93ull;
                                    x ^= x >> 30;
                                    x *= 0xBF58476D1CE4E5B9ull;
                                    x ^= x >> 27;
                                    x *= 0x94D049BB133111EBull;
                                    x ^= x >> 31;
                                    const double u =
                                        static_cast<double>(x & 0x1FFFFFFFFFFFFFull) /
                                        static_cast<double>(0x20000000000000ull);
                                    d_acq(i) = u;
                                });
                        }

                        Kokkos::fence();
                    }

                    Kokkos::realloc(d_topIdx, B);
                    Kokkos::realloc(d_taken, numCandidates);
                    Kokkos::deep_copy(d_taken, char(0));

                    const int actualB = B;

                    if (actualB == 0) continue;
                    if (usesAdditiveDeviceObjective(objTy)) recordObjectiveEvaluations(actualB, true);

                    {
                        PROFILE_SCOPE("topB_selection_minloc");
                        Kokkos::fence();

                        Kokkos::View<AcqIdx*, mem_space>
                            pairs("pairs", numCandidates);

                        Kokkos::parallel_for(
                            "pack_pairs",
                            Kokkos::RangePolicy<exec_space>(0, numCandidates),
                            KOKKOS_LAMBDA(int i) {
                                pairs(i) = AcqIdx(d_acq(i), i);
                            });

                        Kokkos::fence();

                        Kokkos::sort(pairs);

                        Kokkos::parallel_for(
                            "copy_topB",
                            Kokkos::RangePolicy<exec_space>(0, actualB),
                            KOKKOS_LAMBDA(int j) {
                                d_topIdx(j) = pairs(j).idx;
                            });

                        Kokkos::fence();
                    }

                    // --------------------------------------------------------
                    // True objective for selected candidates.
                    //
                    // Important fix:
                    // f(candidate) = f(initialCenter)
                    //              + sum_t [newTerm(t) - oldTerm(t)]
                    //
                    // This is correct for all additive objectives.
                    // --------------------------------------------------------
                    {
                        PROFILE_SCOPE("eval_objective_gpu (device)");
                        Kokkos::fence();

                        Kokkos::realloc(d_fx, actualB);

                        if (usesExternalObjective(objTy)) {
                            auto h_topIdx =
                                Kokkos::create_mirror_view_and_copy(
                                    Kokkos::HostSpace(),
                                    d_topIdx);

                            std::vector<double> X_eval(
                                static_cast<size_t>(actualB) *
                                static_cast<size_t>(dimension));

                            for (int j = 0; j < actualB; ++j) {
                                const int srcIdx = h_topIdx(j);
                                const size_t gChild =
                                    childStart + static_cast<size_t>(srcIdx);

                                Hypersphere hs =
                                    makeChildHypersphereFromGlobal(gChild,
                                                                   currentHyperspheres,
                                                                   dimension,
                                                                   fboConfig);

                                auto x_dev =
                                    reconstructCenter(hs,
                                                      initialCenter,
                                                      initialRadius,
                                                      alphaDepthHost,
                                                      decompositionKind);

                                auto x_host =
                                    Kokkos::create_mirror_view_and_copy(
                                        Kokkos::HostSpace(),
                                        x_dev);

                                for (int d = 0; d < dimension; ++d) {
                                    X_eval[static_cast<size_t>(j) *
                                           static_cast<size_t>(dimension) +
                                           static_cast<size_t>(d)] =
                                        x_host(d);
                                }
                            }

                            std::vector<double> fx_host(actualB);
                            external_eval_batch_host(X_eval.data(),
                                                     dimension,
                                                     actualB,
                                                     fx_host.data());

                            Kokkos::View<double*, Kokkos::HostSpace>
                                h_fx("h_fx_external_objective", actualB);
                            for (int j = 0; j < actualB; ++j) {
                                h_fx(j) = fx_host[static_cast<size_t>(j)];
                            }

                            Kokkos::deep_copy(d_fx, h_fx);
                            Kokkos::fence();
                        } else if (!usesAdditiveDeviceObjective(objTy)) {
                            auto h_topIdx =
                                Kokkos::create_mirror_view_and_copy(
                                    Kokkos::HostSpace(),
                                    d_topIdx);

                            Kokkos::View<double*, Kokkos::HostSpace>
                                h_fx("h_fx_full_objective", actualB);

                            for (int j = 0; j < actualB; ++j) {
                                const int srcIdx = h_topIdx(j);
                                const size_t gChild =
                                    childStart + static_cast<size_t>(srcIdx);

                                Hypersphere hs =
                                    makeChildHypersphereFromGlobal(gChild,
                                                                   currentHyperspheres,
                                                                   dimension,
                                                                   fboConfig);

                                auto x_dev =
                                    reconstructCenter(hs,
                                                      initialCenter,
                                                      initialRadius,
                                                      alphaDepthHost,
                                                      decompositionKind);

                                h_fx(j) =
                                    computeObjectiveValue(x_dev,
                                                          expenseFactor,
                                                          objTy);
                            }

                            Kokkos::deep_copy(d_fx, h_fx);
                            Kokkos::fence();
                        } else {
                            using team_policy = Kokkos::TeamPolicy<exec_space>;
                            using member_type = team_policy::member_type;

                            const double baseFx = initialCenterObjective;

                            Kokkos::parallel_for(
                                "evalFx_from_moves_topB",
                                team_policy(actualB, Kokkos::AUTO),
                                KOKKOS_LAMBDA(const member_type& team) {
                                    const int j = team.league_rank();
                                    const int i = d_topIdx(j);

                                    const int depth_i = depths_chunk(i);
                                    const size_t st   = idOffsets_chunk(i);

                                    double deltaFx = 0.0;

                                    Kokkos::parallel_reduce(
                                        Kokkos::TeamThreadRange(team, dimension),
                                        [&](int dim, double& acc) {
                                            double delta = 0.0;

                                            for (int dd = 0; dd < depth_i; ++dd) {
                                                if (idDims_chunk(st + dd) != dim) {
                                                    continue;
                                                }

                                                delta += fboPathMoveOffset(
                                                    initialRadius,
                                                    decompositionKind,
                                                    idDims_chunk,
                                                    idSigns_chunk,
                                                    st,
                                                    dd,
                                                    alphaDepthDev);
                                            }

                                            if (delta != 0.0) {
                                                const double x0 =
                                                    initialCenter(dim);
                                                const double x1 = x0 + delta;

                                                const double oldTerm =
                                                    objectiveTerm(objTy,
                                                                  x0,
                                                                  dim,
                                                                  dimension,
                                                                  expenseFactor);

                                                const double newTerm =
                                                    objectiveTerm(objTy,
                                                                  x1,
                                                                  dim,
                                                                  dimension,
                                                                  expenseFactor);

                                                acc += newTerm - oldTerm;
                                            }
                                        },
                                        deltaFx);

                                    Kokkos::single(Kokkos::PerTeam(team), [&]() {
                                        d_fx(j) = baseFx + deltaFx;
                                    });
                                });

                            Kokkos::fence();
                        }
                    }

                    MinAcqIndex bestFx;

                    Kokkos::parallel_reduce(
                        "minFx",
                        Kokkos::RangePolicy<exec_space>(0, actualB),
                        KOKKOS_LAMBDA(int i, MinAcqIndex& upd) {
                            const double v = d_fx(i);

                            if (v < upd.value) {
                                upd.value = v;
                                upd.index = i;
                            }
                        },
                        Kokkos::Min<MinAcqIndex>(bestFx));

                    if (bestFx.index >= 0 &&
                        bestFx.value < bestObjectiveValue) {
                        auto h_topIdx =
                            Kokkos::create_mirror_view_and_copy(
                                Kokkos::HostSpace(),
                                d_topIdx);

                        const int srcIdx = h_topIdx(bestFx.index);

                        const size_t gChild =
                            childStart + static_cast<size_t>(srcIdx);

                        Hypersphere hs =
                            makeChildHypersphereFromGlobal(gChild,
                                                           currentHyperspheres,
                                                           dimension,
                                                           fboConfig);

                        auto x_best =
                            reconstructCenter(hs,
                                              initialCenter,
                                              initialRadius,
                                              alphaDepthHost,
                                              decompositionKind);

                        Kokkos::deep_copy(bestSolution, x_best);

                        bestObjectiveValue = bestFx.value;
                    }

                    if (bestFx.index >= 0) {
                        auto h_topIdx =
                            Kokkos::create_mirror_view_and_copy(
                                Kokkos::HostSpace(),
                                d_topIdx);

                        const int srcIdx = h_topIdx(bestFx.index);

                        const size_t gChild =
                            childStart + static_cast<size_t>(srcIdx);

                        Hypersphere hs =
                            makeChildHypersphereFromGlobal(gChild,
                                                           currentHyperspheres,
                                                           dimension,
                                                           fboConfig);

                        if (bestFx.value < depthLocalBestVal) {
                            depthLocalBestVal   = bestFx.value;
                            depthLocalBestID    = std::move(hs.id);
                            depthLocalBestValid = true;
                        }
                    }

                    remaining -= actualB;

                    // --------------------------------------------------------
                    // Accumulate new training data for delayed surrogate update.
                    // --------------------------------------------------------
                    {
                        PROFILE_SCOPE("host_copy_and_accumulate");

                        auto h_topIdx =
                            Kokkos::create_mirror_view_and_copy(
                                Kokkos::HostSpace(),
                                d_topIdx);

                        std::vector<double> newTrainingPoints;
                        newTrainingPoints.reserve(
                            static_cast<size_t>(actualB) *
                            static_cast<size_t>(dimension));

                        for (int j = 0; j < actualB; ++j) {
                            const int srcIdx = h_topIdx(j);

                            const size_t gChild =
                                childStart + static_cast<size_t>(srcIdx);

                            Hypersphere hs =
                                makeChildHypersphereFromGlobal(gChild,
                                                               currentHyperspheres,
                                                               dimension,
                                                               fboConfig);

                            auto x_dev =
                                reconstructCenter(hs,
                                                  initialCenter,
                                                  initialRadius,
                                                  alphaDepthHost,
                                                  decompositionKind);

                            auto x_host =
                                Kokkos::create_mirror_view_and_copy(
                                    Kokkos::HostSpace(),
                                    x_dev);

                            for (int d = 0; d < dimension; ++d) {
                                newTrainingPoints.push_back(x_host(d));
                            }
                        }

                        auto h_fx =
                            Kokkos::create_mirror_view_and_copy(
                                Kokkos::HostSpace(),
                                d_fx);

                        for (int j = 0; j < actualB; ++j) {
                            y_to_add_accum.push_back(h_fx(j));
                        }

                        if (!useGPSurrogate) {
                            trainingY.insert(trainingY.end(),
                                             y_to_add_accum.end() - actualB,
                                             y_to_add_accum.end());
                        }

                        trainingPoints.insert(trainingPoints.end(),
                                              newTrainingPoints.begin(),
                                              newTrainingPoints.end());

                        points_to_add_accum.insert(points_to_add_accum.end(),
                                                   newTrainingPoints.begin(),
                                                   newTrainingPoints.end());
                    }

                    if (remaining == 0) break;
                }

                // ------------------------------------------------------------
                // End-of-depth surrogate update if not overlapped later.
                // ------------------------------------------------------------
                if ((useGPSurrogate || useSimpleSurrogate) &&
                    !did_update_this_depth &&
                    !points_to_add_accum.empty()) {
                    const int n_add =
                        static_cast<int>(points_to_add_accum.size()) /
                        dimension;

                    if (world_rank == 0) {
                        std::cout << "[diag] end-of-depth update, n_add="
                                  << n_add << "\n";
                    }

                    PROFILE_SCOPE("update model (host) [batched end-of-depth]");

                    if (useGPSurrogate) {
                        gpModel =
                            GP::updateGP_incremental_knownY(gpModel,
                                                            points_to_add_accum,
                                                            y_to_add_accum,
                                                            n_add,
                                                            pcaRankCap,
                                                            sigma_f,
                                                            l,
                                                            sigma_n,
                                                            gpKernelKind,
                                                            gpRQAlpha);

                        d_gpModel = GP::makeDeviceCopy(gpModel,
                                                       sigma_f,
                                                       l,
                                                       sigma_n,
                                                       gpKernelKind,
                                                       gpRQAlpha);

                        d_mu_X = d_gpModel.mu_X;
                        d_sd_X = d_gpModel.sd_X;
                        d_B = d_gpModel.B;
                    } else {
                        simpleSurrogateModel =
                            FBO::makeSimpleSurrogateDevice(
                                trainingPoints,
                                trainingY,
                                dimension,
                                fboConfig.surrogate,
                                fboConfig.surrogateLengthScale,
                                fboConfig.idwPower);
                    }

                    did_update_this_depth = true;

                    points_to_add_accum.clear();
                    y_to_add_accum.clear();
                }

                if (!useFBD &&
                    fboConfig.adaptiveHypersphereRadius &&
                    decompositionKind == FBODecomposition::Hypersphere) {
                    double totalEIGlobal = 0.0;
                    double bestEIGlobal = 0.0;
                    MPI_Allreduce(&totalEI_stream,
                                  &totalEIGlobal,
                                  1,
                                  MPI_DOUBLE,
                                  MPI_SUM,
                                  MPI_COMM_WORLD);
                    MPI_Allreduce(&bestEI_stream,
                                  &bestEIGlobal,
                                  1,
                                  MPI_DOUBLE,
                                  MPI_MAX,
                                  MPI_COMM_WORLD);

                    constexpr double eps = 1e-30;
                    constexpr double lambda = 1.5;
                    const double q =
                        bestEIGlobal / (totalEIGlobal + eps);
                    const double q0 =
                        1.0 / static_cast<double>(std::max(1, beta));
                    const double alphaMin =
                        0.25 * fixedHypersphereFactor;
                    const double alphaMax =
                        1.25 * fixedHypersphereFactor;
                    const double alphaNext = std::clamp(
                        fixedHypersphereFactor *
                            (1.0 - lambda * (q - q0)),
                        alphaMin,
                        alphaMax);

                    const int nextMoveIndex = depth + 1;
                    if (static_cast<int>(alphaDepthHost.size()) <=
                        nextMoveIndex) {
                        alphaDepthHost.resize(
                            static_cast<size_t>(nextMoveIndex) + 1u,
                            fixedHypersphereFactor);
                    }
                    alphaDepthHost[static_cast<size_t>(nextMoveIndex)] =
                        alphaNext;

                    if (world_rank == 0) {
                        std::cout << "[radius-adaptation] depth=" << depth
                                  << " next_move_index=" << nextMoveIndex
                                  << " total_ei=" << totalEIGlobal
                                  << " best_ei=" << bestEIGlobal
                                  << " q=" << q
                                  << " q0=" << q0
                                  << " alpha_next=" << alphaNext
                                  << "\n";
                    }
                }

                // ------------------------------------------------------------
                // rho-coverage from streamed EI topM.
                // ------------------------------------------------------------
                std::vector<double> eiCompact;
                std::vector<int64_t> idxCompact;

                if (useFBD) {
                    eiCompact = std::move(fbdAllWeights);
                    idxCompact = std::move(fbdAllIndices);
                } else {
                    eiCompact.reserve(topM.size());
                    idxCompact.reserve(topM.size());

                    std::vector<Pair> tmp;
                    tmp.reserve(topM.size());

                    while (!topM.empty()) {
                        tmp.push_back(topM.top());
                        topM.pop();
                    }

                    std::reverse(tmp.begin(), tmp.end());

                    for (auto& p : tmp) {
                        eiCompact.push_back(p.ei);
                        idxCompact.push_back(p.gidx);
                    }
                }

                std::vector<int> keepIdx;

                {
                    PROFILE_SCOPE("rho_coverage_local");
                    if (useFBD) {
                        const int fbdBeta =
                            fbdConfig.beamWidth > 0 ? fbdConfig.beamWidth : beta;
                        keepIdx = selectFBDFrontier(
                            eiCompact,
                            rho,
                            fbdBeta,
                            fbdConfig,
                            static_cast<uint64_t>(seed + depth * 131 +
                                                  world_rank));
                    } else {
                        keepIdx = selectCoverageGPU(eiCompact, rho, beta);
                    }
                }

                const int localBeta = static_cast<int>(keepIdx.size());

                if (world_rank == 0) {
                    const double totalEI = totalEI_stream;
                    const double bestEI  = bestEI_stream;

                    std::cout << "[DBG d=" << depth << "]  "
                              << "totalEI = " << totalEI
                              << "   bestEI = " << bestEI
                              << "   best/total = "
                              << (totalEI > 0.0 ? bestEI / totalEI : 0.0)
                              << "   rho = " << rho
                              << std::endl;

                    std::cout << "[Info] depth " << depth
                              << ": local Beta = " << localBeta
                              << " (rho = " << rho << ")\n";
                }

                std::vector<double> localTopScores(localBeta);
                std::vector<HypersphereID> localTopIDs(localBeta);

                for (int i = 0; i < localBeta; ++i) {
                    const int compactIdx = keepIdx[i];

                    localTopScores[i] = eiCompact[compactIdx];

                    const size_t gChild =
                        static_cast<size_t>(idxCompact[compactIdx]);

                    Hypersphere child =
                        makeChildHypersphereFromGlobal(gChild,
                                                       currentHyperspheres,
                                                       dimension,
                                                       fboConfig);

                    localTopIDs[i] = std::move(child.id);
                }

                // ------------------------------------------------------------
                // Intra-node exchange.
                // ------------------------------------------------------------
                const int localDataSize = localBeta;

                std::vector<int> allDataSizes(node_size);
                std::vector<int> allIDSizes(node_size);

                MPI_Request size_reqs[2] = {
                    MPI_REQUEST_NULL,
                    MPI_REQUEST_NULL
                };

                MPI_Iallgather(&localDataSize,
                               1,
                               MPI_INT,
                               allDataSizes.data(),
                               1,
                               MPI_INT,
                               node_comm,
                               &size_reqs[0]);

                std::vector<int> localIDsSerialized;
                localIDsSerialized.reserve(
                    static_cast<size_t>(localBeta) *
                    2u *
                    static_cast<size_t>(dimension));

                {
                    PROFILE_SCOPE("serialize_IDs_local");

                    for (const auto& id : localTopIDs) {
                        auto s = serializeHypersphereID(id);

                        localIDsSerialized.insert(localIDsSerialized.end(),
                                                  s.begin(),
                                                  s.end());
                    }
                }

                const int localIDSize =
                    static_cast<int>(localIDsSerialized.size());

                MPI_Iallgather(&localIDSize,
                               1,
                               MPI_INT,
                               allIDSizes.data(),
                               1,
                               MPI_INT,
                               node_comm,
                               &size_reqs[1]);

                mpi_progress_some(size_reqs, 2);

                MPI_Waitall(2, size_reqs, MPI_STATUSES_IGNORE);

                const int totalDataSize =
                    std::accumulate(allDataSizes.begin(),
                                    allDataSizes.end(),
                                    0);

                const int totalIDSize =
                    std::accumulate(allIDSizes.begin(),
                                    allIDSizes.end(),
                                    0);

                std::vector<int> displs(node_size, 0);
                std::vector<int> displsIDs(node_size, 0);

                for (int i = 1; i < node_size; ++i) {
                    displs[i] = displs[i - 1] + allDataSizes[i - 1];
                    displsIDs[i] = displsIDs[i - 1] + allIDSizes[i - 1];
                }

                std::vector<double> allScores(totalDataSize);
                std::vector<int> allIDsSerialized(totalIDSize);

                MPI_Request pay_reqs[2] = {
                    MPI_REQUEST_NULL,
                    MPI_REQUEST_NULL
                };

                MPI_Iallgatherv(localTopScores.data(),
                                localDataSize,
                                MPI_DOUBLE,
                                allScores.data(),
                                allDataSizes.data(),
                                displs.data(),
                                MPI_DOUBLE,
                                node_comm,
                                &pay_reqs[0]);

                MPI_Iallgatherv(localIDsSerialized.data(),
                                localIDSize,
                                MPI_INT,
                                allIDsSerialized.data(),
                                allIDSizes.data(),
                                displsIDs.data(),
                                MPI_INT,
                                node_comm,
                                &pay_reqs[1]);

                // ------------------------------------------------------------
                // Optional MPI-progress thread during overlapped surrogate update.
                //
                // Unlike the old detached version, this one is joined safely.
                // ------------------------------------------------------------
                std::atomic<bool> progress_stop(false);
                std::thread progress_thread;

                auto start_progress_thread = [&]() {
                    if (!have_thread_multiple) return;

                    progress_thread_active = true;

                    progress_thread = std::thread([&]() {
                        while (!progress_stop.load(std::memory_order_relaxed)) {
                            mpi_progress_some(pay_reqs, 2);
                            std::this_thread::yield();
                        }
                    });
                };

                double t_upd0 = MPI_Wtime();

                if ((useGPSurrogate || useSimpleSurrogate) &&
                    !did_update_this_depth &&
                    !points_to_add_accum.empty()) {
                    if (world_rank == 0) {
                        std::cout << "[diag] overlapped update, n_add="
                                  << (points_to_add_accum.size() / dimension)
                                  << "\n";
                    }

                    PROFILE_SCOPE(
                        "update model (host) [overlap with intra-node gathers]");

                    start_progress_thread();

                    const int n_add =
                        static_cast<int>(points_to_add_accum.size()) /
                        dimension;

                    if (useGPSurrogate) {
                        gpModel =
                            GP::updateGP_incremental_knownY(gpModel,
                                                            points_to_add_accum,
                                                            y_to_add_accum,
                                                            n_add,
                                                            pcaRankCap,
                                                            sigma_f,
                                                            l,
                                                            sigma_n,
                                                            gpKernelKind,
                                                            gpRQAlpha);

                        d_gpModel = GP::makeDeviceCopy(gpModel,
                                                       sigma_f,
                                                       l,
                                                       sigma_n,
                                                       gpKernelKind,
                                                       gpRQAlpha);

                        d_mu_X = d_gpModel.mu_X;
                        d_sd_X = d_gpModel.sd_X;
                        d_B = d_gpModel.B;
                    } else {
                        simpleSurrogateModel =
                            FBO::makeSimpleSurrogateDevice(
                                trainingPoints,
                                trainingY,
                                dimension,
                                fboConfig.surrogate,
                                fboConfig.surrogateLengthScale,
                                fboConfig.idwPower);
                    }

                    did_update_this_depth = true;

                    points_to_add_accum.clear();
                    y_to_add_accum.clear();
                }

                if (progress_thread.joinable()) {
                    progress_stop.store(true, std::memory_order_relaxed);
                    progress_thread.join();
                }

                double t_upd1 = MPI_Wtime();

                {
                    PROFILE_SCOPE("intra_payload_waitall");
                    MPI_Waitall(2, pay_reqs, MPI_STATUSES_IGNORE);
                }

                double t_wait = MPI_Wtime();

                if (world_rank == 0) {
                    const double gp_update_s  = t_upd1 - t_upd0;
                    const double wait_after_s = t_wait - t_upd1;

                    std::cout << std::fixed << std::setprecision(3)
                              << "[overlap-intra] depth " << depth
                              << "  GP_update=" << gp_update_s << "s"
                              << "  wait_after_update=" << wait_after_s
                              << "s"
                              << "  progress_thread="
                              << (progress_thread_active ? "on" : "off")
                              << std::endl;
                }

                std::vector<HypersphereID> allIDs(totalDataSize);

                {
                    PROFILE_SCOPE("deserialize_IDs_intra");

                    size_t offset = 0;

                    for (int i = 0; i < totalDataSize; ++i) {
                        allIDs[i] =
                            deserializeHypersphereID(allIDsSerialized,
                                                     offset);
                    }
                }

                std::vector<double> wNode(totalDataSize);

                for (int i = 0; i < totalDataSize; ++i) {
                    wNode[i] = allScores[i];
                }

                std::vector<int> nodeSelectedIndices;

                {
                    PROFILE_SCOPE("rho_coverage_node");
                    if (useFBD) {
                        const int fbdBeta =
                            fbdConfig.beamWidth > 0 ? fbdConfig.beamWidth : beta;
                        nodeSelectedIndices = selectFBDFrontier(
                            wNode,
                            rho,
                            fbdBeta,
                            fbdConfig,
                            static_cast<uint64_t>(seed + depth * 521 +
                                                  node_id));
                    } else {
                        nodeSelectedIndices = selectCoverage(wNode, rho);
                    }
                }

                int nodeBeta =
                    static_cast<int>(nodeSelectedIndices.size());

                const int nodeBetaCap =
                    useFBD && fbdConfig.beamWidth > 0 ? fbdConfig.beamWidth
                                                       : beta;
                if (nodeBeta > nodeBetaCap) {
                    nodeSelectedIndices.resize(nodeBetaCap);
                    nodeBeta = nodeBetaCap;
                }

                if (world_rank == 0) {
                    std::cout << "[Info] depth " << depth
                              << ": Node Beta = " << nodeBeta
                              << " (rho = " << rho << ")\n";
                }

                std::vector<double> nodeTopScores(nodeBeta);
                std::vector<HypersphereID> nodeTopIDs(nodeBeta);

                for (int i = 0; i < nodeBeta; ++i) {
                    const int idx = nodeSelectedIndices[i];

                    nodeTopScores[i] = allScores[idx];
                    nodeTopIDs[i] = allIDs[idx];
                }

                // ------------------------------------------------------------
                // Inter-node leader exchange.
                // ------------------------------------------------------------
                if (node_rank == 0) {
                    std::vector<int> nodeLocalIDsSerialized;

                    for (const auto& id : nodeTopIDs) {
                        auto s = serializeHypersphereID(id);

                        nodeLocalIDsSerialized.insert(
                            nodeLocalIDsSerialized.end(),
                            s.begin(),
                            s.end());
                    }

                    const int nodeLocalIDSize =
                        static_cast<int>(nodeLocalIDsSerialized.size());

                    std::vector<int> nodeAllIDSizes(num_nodes);

                    MPI_Allgather(&nodeLocalIDSize,
                                  1,
                                  MPI_INT,
                                  nodeAllIDSizes.data(),
                                  1,
                                  MPI_INT,
                                  node_leader_comm);

                    const int nodeLocalDataSize = nodeBeta;

                    std::vector<int> nodeAllDataSizes(num_nodes);

                    MPI_Allgather(&nodeLocalDataSize,
                                  1,
                                  MPI_INT,
                                  nodeAllDataSizes.data(),
                                  1,
                                  MPI_INT,
                                  node_leader_comm);

                    const int nodeTotalDataSize =
                        std::accumulate(nodeAllDataSizes.begin(),
                                        nodeAllDataSizes.end(),
                                        0);

                    std::vector<int> nodeDispls(num_nodes, 0);

                    for (int i = 1; i < num_nodes; ++i) {
                        nodeDispls[i] =
                            nodeDispls[i - 1] +
                            nodeAllDataSizes[i - 1];
                    }

                    const int nodeTotalIDSize =
                        std::accumulate(nodeAllIDSizes.begin(),
                                        nodeAllIDSizes.end(),
                                        0);

                    std::vector<int> nodeDisplsIDs(num_nodes, 0);

                    for (int i = 1; i < num_nodes; ++i) {
                        nodeDisplsIDs[i] =
                            nodeDisplsIDs[i - 1] +
                            nodeAllIDSizes[i - 1];
                    }

                    std::vector<double> nodeAllScores(
                        static_cast<size_t>(
                            std::max(0, nodeTotalDataSize)));

                    if (nodeTotalDataSize > 0) {
                        MPI_Allgatherv(nodeTopScores.data(),
                                       nodeLocalDataSize,
                                       MPI_DOUBLE,
                                       nodeAllScores.data(),
                                       nodeAllDataSizes.data(),
                                       nodeDispls.data(),
                                       MPI_DOUBLE,
                                       node_leader_comm);
                    }

                    std::vector<int> nodeAllIDsSerialized(nodeTotalIDSize);

                    MPI_Allgatherv(nodeLocalIDsSerialized.data(),
                                   nodeLocalIDSize,
                                   MPI_INT,
                                   nodeAllIDsSerialized.data(),
                                   nodeAllIDSizes.data(),
                                   nodeDisplsIDs.data(),
                                   MPI_INT,
                                   node_leader_comm);

                    std::vector<HypersphereID> nodeAllIDs;

                    size_t offsetNode = 0;

                    while (offsetNode < nodeAllIDsSerialized.size()) {
                        nodeAllIDs.emplace_back(
                            deserializeHypersphereID(nodeAllIDsSerialized,
                                                     offsetNode));
                    }

                    PROFILE_SCOPE("rho_coverage_global");

                    std::vector<int> globalSelectedIndices;
                    if (useFBD) {
                        const int fbdBeta =
                            fbdConfig.beamWidth > 0 ? fbdConfig.beamWidth : beta;
                        globalSelectedIndices = selectFBDFrontier(
                            nodeAllScores,
                            rho,
                            fbdBeta,
                            fbdConfig,
                            static_cast<uint64_t>(seed + depth * 997 +
                                                  node_id));
                    } else {
                        globalSelectedIndices = selectCoverage(nodeAllScores,
                                                              rho);
                    }

                    int globalBeta =
                        static_cast<int>(globalSelectedIndices.size());

                    const int globalBetaCap =
                        useFBD && fbdConfig.beamWidth > 0 ? fbdConfig.beamWidth
                                                           : beta;
                    if (globalBeta > globalBetaCap) {
                        globalSelectedIndices.resize(globalBetaCap);
                        globalBeta = globalBetaCap;
                    }

                    if (world_rank == 0) {
                        std::cout << "[Info] depth " << depth
                                  << ": Global Beta = " << globalBeta
                                  << " (rho = " << rho << ")\n";
                    }

                    std::vector<Hypersphere> selectedHyperspheres;
                    selectedHyperspheres.reserve(globalBeta);

                    for (int idx : globalSelectedIndices) {
                        HypersphereID selID = nodeAllIDs[idx];

                        selectedHyperspheres.emplace_back(
                            dimension,
                            static_cast<int>(selID.size()),
                            selID);
                    }

                    if (selectedHyperspheres.empty()) {
                        std::cerr << "[Error] selectedHyperspheres is empty "
                                     "after global selection."
                                  << std::endl;

                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }

                    bool expandedForUtilization = false;

                    if (globalBeta < world_size) {
                        std::vector<Hypersphere> expanded;

                        expanded.reserve(
                            selectedHyperspheres.size() *
                            static_cast<size_t>(
                                childCountForConfig(dimension,
                                                    fboConfig)));

                        for (const auto& hs : selectedHyperspheres) {
                            auto kids = decomposeFractalCell(hs,
                                                             dimension,
                                                             fboConfig);

                            for (const auto& ch : kids) {
                                expanded.push_back(ch);

                                if (static_cast<int>(expanded.size()) >=
                                    world_size) {
                                    break;
                                }
                            }

                            if (static_cast<int>(expanded.size()) >=
                                world_size) {
                                break;
                            }
                        }

                        if (!expanded.empty()) {
                            selectedHyperspheres.swap(expanded);
                            expandedForUtilization = true;
                            globalBeta =
                                static_cast<int>(
                                    selectedHyperspheres.size());

                            if (world_rank == 0) {
                                std::cout << "[Info] depth " << depth
                                          << ": expanded "
                                             "(dispatch children) to fill "
                                          << globalBeta
                                          << " tasks for "
                                          << world_size
                                          << " procs\n";
                            }
                        }
                    }

                    int numSelected =
                        static_cast<int>(selectedHyperspheres.size());

                    int expandFlag =
                        expandedForUtilization ? 1 : 0;

                    MPI_Bcast(&numSelected,
                              1,
                              MPI_INT,
                              0,
                              node_comm);

                    MPI_Bcast(&expandFlag,
                              1,
                              MPI_INT,
                              0,
                              node_comm);

                    std::vector<int> selectedIDsSerialized;

                    selectedIDsSerialized.reserve(
                        static_cast<size_t>(numSelected) *
                        2u *
                        static_cast<size_t>(dimension));

                    for (const auto& hs : selectedHyperspheres) {
                        auto s = serializeHypersphereID(hs.id);

                        selectedIDsSerialized.insert(
                            selectedIDsSerialized.end(),
                            s.begin(),
                            s.end());
                    }

                    int selectedTotalIDSize =
                        static_cast<int>(selectedIDsSerialized.size());

                    MPI_Bcast(&selectedTotalIDSize,
                              1,
                              MPI_INT,
                              0,
                              node_comm);

                    MPI_Bcast(selectedIDsSerialized.data(),
                              selectedTotalIDSize,
                              MPI_INT,
                              0,
                              node_comm);

                    std::vector<Hypersphere> selectedHyperspheresAll;
                    selectedHyperspheresAll.reserve(numSelected);

                    size_t off = 0;

                    for (int i = 0; i < numSelected; ++i) {
                        HypersphereID id =
                            deserializeHypersphereID(selectedIDsSerialized,
                                                     off);

                        const int dpth = static_cast<int>(id.size());

                        selectedHyperspheresAll.emplace_back(dimension,
                                                             dpth,
                                                             id);
                    }

                    currentHyperspheres =
                        std::move(selectedHyperspheresAll);
                } else {
                    int numSelected = 0;
                    int expandFlag  = 0;

                    MPI_Bcast(&numSelected,
                              1,
                              MPI_INT,
                              0,
                              node_comm);

                    MPI_Bcast(&expandFlag,
                              1,
                              MPI_INT,
                              0,
                              node_comm);

                    int selectedTotalIDSize = 0;

                    MPI_Bcast(&selectedTotalIDSize,
                              1,
                              MPI_INT,
                              0,
                              node_comm);

                    std::vector<int> selectedIDsSerialized(
                        selectedTotalIDSize);

                    MPI_Bcast(selectedIDsSerialized.data(),
                              selectedTotalIDSize,
                              MPI_INT,
                              0,
                              node_comm);

                    std::vector<Hypersphere> selectedHyperspheres;
                    selectedHyperspheres.reserve(numSelected);

                    size_t off = 0;

                    for (int i = 0; i < numSelected; ++i) {
                        HypersphereID id =
                            deserializeHypersphereID(selectedIDsSerialized,
                                                     off);

                        const int dpth = static_cast<int>(id.size());

                        selectedHyperspheres.emplace_back(dimension,
                                                          dpth,
                                                          id);
                    }

                    currentHyperspheres =
                        std::move(selectedHyperspheres);
                }

                // ------------------------------------------------------------
                // FBD probes need their actual coordinates: a cell ID only
                // reconstructs the centre, which may not be the winning probe.
                // ------------------------------------------------------------
                if (useFBD) {
                    struct {
                        double val;
                        int rank;
                    } mine{bestObjectiveValue, world_rank}, winner;
                    MPI_Allreduce(&mine, &winner, 1, MPI_DOUBLE_INT,
                                  MPI_MINLOC, MPI_COMM_WORLD);
                    auto winningPoint = Kokkos::create_mirror_view_and_copy(
                        Kokkos::HostSpace(), bestSolution);
                    MPI_Bcast(winningPoint.data(), dimension, MPI_DOUBLE,
                              winner.rank, MPI_COMM_WORLD);
                    bestObjectiveValue = winner.val;
                    Kokkos::deep_copy(bestSolution, winningPoint);
                } else {
                    std::vector<int> localIDSer;

                    if (depthLocalBestValid) {
                        localIDSer =
                            serializeHypersphereID(depthLocalBestID);
                    }

                    const int localIDSize =
                        static_cast<int>(localIDSer.size());

                    const double localVal =
                        depthLocalBestValid
                            ? depthLocalBestVal
                            : std::numeric_limits<double>::infinity();

                    std::vector<int> allIDSizes(node_size);
                    std::vector<double> allVals(node_size);

                    MPI_Allgather(&localIDSize,
                                  1,
                                  MPI_INT,
                                  allIDSizes.data(),
                                  1,
                                  MPI_INT,
                                  node_comm);

                    MPI_Allgather(&localVal,
                                  1,
                                  MPI_DOUBLE,
                                  allVals.data(),
                                  1,
                                  MPI_DOUBLE,
                                  node_comm);

                    std::vector<int> idDispls(node_size, 0);

                    for (int i = 1; i < node_size; ++i) {
                        idDispls[i] =
                            idDispls[i - 1] + allIDSizes[i - 1];
                    }

                    const int totID =
                        node_size > 0
                            ? idDispls.back() + allIDSizes.back()
                            : 0;

                    std::vector<int> allIDSer(totID);

                    MPI_Allgatherv(localIDSer.data(),
                                   localIDSize,
                                   MPI_INT,
                                   allIDSer.data(),
                                   allIDSizes.data(),
                                   idDispls.data(),
                                   MPI_INT,
                                   node_comm);

                    int bestRankInNode = -1;
                    double nodeBestVal =
                        std::numeric_limits<double>::infinity();

                    for (int r = 0; r < node_size; ++r) {
                        if (allVals[r] < nodeBestVal) {
                            nodeBestVal = allVals[r];
                            bestRankInNode = r;
                        }
                    }

                    HypersphereID nodeBestID;

                    if (bestRankInNode >= 0 &&
                        nodeBestVal !=
                            std::numeric_limits<double>::infinity()) {
                        size_t off =
                            static_cast<size_t>(idDispls[bestRankInNode]);

                        nodeBestID =
                            deserializeHypersphereID(allIDSer, off);
                    }

                    double clusterBestVal = nodeBestVal;
                    int clusterBestNode = node_id;

                    if (node_rank == 0) {
                        struct {
                            double val;
                            int rank;
                        } mine{nodeBestVal, node_id}, winner;

                        MPI_Allreduce(&mine,
                                      &winner,
                                      1,
                                      MPI_DOUBLE_INT,
                                      MPI_MINLOC,
                                      node_leader_comm);

                        clusterBestVal  = winner.val;
                        clusterBestNode = winner.rank;

                        const int is_src =
                            (node_id == clusterBestNode &&
                             std::isfinite(clusterBestVal))
                                ? 1
                                : 0;

                        std::vector<int> winnerIDSer;

                        if (is_src) {
                            winnerIDSer =
                                serializeHypersphereID(nodeBestID);
                        }

                        int winSize =
                            is_src
                                ? static_cast<int>(winnerIDSer.size())
                                : 0;

                        MPI_Bcast(&winSize,
                                  1,
                                  MPI_INT,
                                  clusterBestNode,
                                  node_leader_comm);

                        if (!is_src) {
                            winnerIDSer.resize(winSize);
                        }

                        if (winSize > 0) {
                            MPI_Bcast(winnerIDSer.data(),
                                      winSize,
                                      MPI_INT,
                                      clusterBestNode,
                                      node_leader_comm);
                        }

                        MPI_Bcast(&clusterBestVal,
                                  1,
                                  MPI_DOUBLE,
                                  0,
                                  node_comm);

                        MPI_Bcast(&winSize,
                                  1,
                                  MPI_INT,
                                  0,
                                  node_comm);

                        if (static_cast<int>(winnerIDSer.size()) !=
                            winSize) {
                            winnerIDSer.resize(winSize);
                        }

                        if (winSize > 0) {
                            MPI_Bcast(winnerIDSer.data(),
                                      winSize,
                                      MPI_INT,
                                      0,
                                      node_comm);
                        }

                        if (winSize > 0) {
                            size_t oo = 0;

                            depthLocalBestID =
                                deserializeHypersphereID(winnerIDSer,
                                                         oo);

                            depthLocalBestVal = clusterBestVal;
                            depthLocalBestValid = true;
                        } else {
                            depthLocalBestValid = false;
                        }
                    } else {
                        MPI_Bcast(&clusterBestVal,
                                  1,
                                  MPI_DOUBLE,
                                  0,
                                  node_comm);

                        int winSize = 0;

                        MPI_Bcast(&winSize,
                                  1,
                                  MPI_INT,
                                  0,
                                  node_comm);

                        std::vector<int> winnerIDSer(winSize);

                        if (winSize > 0) {
                            MPI_Bcast(winnerIDSer.data(),
                                      winSize,
                                      MPI_INT,
                                      0,
                                      node_comm);
                        }

                        if (winSize > 0) {
                            size_t oo = 0;

                            depthLocalBestID =
                                deserializeHypersphereID(winnerIDSer,
                                                         oo);

                            depthLocalBestVal = clusterBestVal;
                            depthLocalBestValid = true;
                        } else {
                            depthLocalBestValid = false;
                        }
                    }

                    if (depthLocalBestValid &&
                        depthLocalBestVal < bestObjectiveValue) {
                        Hypersphere incHS(
                            dimension,
                            static_cast<int>(depthLocalBestID.size()),
                            depthLocalBestID);

                        auto incCenter =
                            reconstructCenter(incHS,
                                              initialCenter,
                                              initialRadius,
                                              alphaDepthHost,
                                              decompositionKind);

                        bestObjectiveValue = depthLocalBestVal;

                        Kokkos::deep_copy(bestSolution, incCenter);
                    }
                }

                if (world_rank == 0) {
                    std::cout << "\n[DEPTH " << depth
                              << "]  incumbent f* = "
                              << bestObjectiveValue << "\n";
                }

                if (!ftCheckpointBase.empty() && depth < activeMaxDepth) {
                    const std::string checkpointWriteBase =
                        ftRollingCheckpoints
                            ? ftDepthBase(ftCheckpointBase, depth)
                            : ftCheckpointBase;
                    const std::string checkpointReplicaWriteBase =
                        (!ftCheckpointReplicaBase.empty() &&
                         ftRollingCheckpoints)
                            ? ftDepthBase(ftCheckpointReplicaBase, depth)
                            : ftCheckpointReplicaBase;
                    unsigned long long localCheckpointBytes = 0;
                    unsigned long long localCheckpointHash = 0;

                    try {
                        auto bestHost =
                            Kokkos::create_mirror_view_and_copy(
                                Kokkos::HostSpace(),
                                bestSolution);

                        FTCheckpointState state;
                        state.worldSize = world_size;
                        state.worldRank = world_rank;
                        state.completedDepth = depth;
                        state.dimension = dimension;
                        state.budget = budget;
                        state.seed = seed;
                        state.nInit = nInitGlobal;
                        state.optimizerFamily =
                            static_cast<int>(optimizerFamily);
                        state.surrogate =
                            static_cast<int>(fboConfig.surrogate);
                        state.doeDistribution =
                            static_cast<int>(
                                fboConfig.doeDistribution);
                        state.bestObjective = bestObjectiveValue;
                        state.bestPoint.resize(dimension);
                        for (int d = 0; d < dimension; ++d) {
                            state.bestPoint[d] = bestHost(d);
                        }
                        state.frontier = currentHyperspheres;
                        state.trainingPoints = trainingPoints;
                        state.trainingY = useGPSurrogate
                            ? gpModel.y_phys
                            : trainingY;

                        saveFTCheckpoint(checkpointWriteBase, state);
                        if (!checkpointReplicaWriteBase.empty()) {
                            saveFTCheckpoint(checkpointReplicaWriteBase,
                                             state);
                        }

                        const std::string localPath =
                            ftRankPath(checkpointWriteBase, world_rank);
                        localCheckpointBytes =
                            static_cast<unsigned long long>(
                                ftFileSizeBytes(localPath));
                        localCheckpointHash =
                            static_cast<unsigned long long>(
                                ftFileFnv1a64(localPath));
                    } catch (const std::exception& e) {
                        std::cerr << "[rank " << world_rank
                                  << "] checkpoint failed: "
                                  << e.what() << "\n";
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }

                    std::vector<unsigned long long> checkpointBytesAll;
                    std::vector<unsigned long long> checkpointHashAll;
                    if (world_rank == 0) {
                        checkpointBytesAll.resize(world_size);
                        checkpointHashAll.resize(world_size);
                    }

                    MPI_Gather(&localCheckpointBytes,
                               1,
                               MPI_UNSIGNED_LONG_LONG,
                               world_rank == 0 ? checkpointBytesAll.data()
                                               : nullptr,
                               1,
                               MPI_UNSIGNED_LONG_LONG,
                               0,
                               MPI_COMM_WORLD);
                    MPI_Gather(&localCheckpointHash,
                               1,
                               MPI_UNSIGNED_LONG_LONG,
                               world_rank == 0 ? checkpointHashAll.data()
                                               : nullptr,
                               1,
                               MPI_UNSIGNED_LONG_LONG,
                               0,
                               MPI_COMM_WORLD);

                    if (world_rank == 0) {
                        try {
                            std::ostringstream manifest;
                            manifest << "version=1\n"
                                     << "world_size=" << world_size << "\n"
                                     << "completed_depth=" << depth << "\n"
                                     << "dimension=" << dimension << "\n"
                                     << "budget=" << budget << "\n"
                                     << "seed=" << seed << "\n"
                                     << "optimizer_family="
                                     << toString(optimizerFamily) << "\n"
                                     << "surrogate="
                                     << toString(fboConfig.surrogate) << "\n"
                                     << "logical_base=" << ftCheckpointBase
                                     << "\n"
                                     << "generation_base="
                                     << checkpointWriteBase << "\n"
                                     << "keep_checkpoints="
                                     << ftKeepCheckpoints << "\n";
                            for (int r = 0; r < world_size; ++r) {
                                manifest << "rank" << r << "_bytes="
                                         << checkpointBytesAll[r] << "\n"
                                         << "rank" << r << "_fnv1a64="
                                         << checkpointHashAll[r] << "\n";
                            }
                            if (!checkpointReplicaWriteBase.empty()) {
                                manifest << "replica_base="
                                         << checkpointReplicaWriteBase << "\n";
                            }
                            manifest << "complete=1\n";
                            ftPublishTextFileAtomically(
                                checkpointWriteBase + ".manifest",
                                manifest.str());
                            if (ftRollingCheckpoints) {
                                ftPublishTextFileAtomically(
                                    ftCheckpointBase + ".latest",
                                    checkpointWriteBase + "\n");
                            }

                            if (!checkpointReplicaWriteBase.empty()) {
                                std::ostringstream replicaManifest;
                                replicaManifest
                                    << "version=1\n"
                                    << "world_size=" << world_size << "\n"
                                    << "completed_depth=" << depth << "\n"
                                    << "dimension=" << dimension << "\n"
                                    << "budget=" << budget << "\n"
                                    << "seed=" << seed << "\n"
                                    << "optimizer_family="
                                    << toString(optimizerFamily) << "\n"
                                    << "surrogate="
                                    << toString(fboConfig.surrogate) << "\n"
                                    << "logical_base="
                                    << ftCheckpointReplicaBase << "\n"
                                    << "generation_base="
                                    << checkpointReplicaWriteBase << "\n"
                                    << "keep_checkpoints="
                                    << ftKeepCheckpoints << "\n"
                                    << "primary_base=" << checkpointWriteBase
                                    << "\n";
                                for (int r = 0; r < world_size; ++r) {
                                    replicaManifest
                                        << "rank" << r << "_bytes="
                                        << checkpointBytesAll[r] << "\n"
                                        << "rank" << r << "_fnv1a64="
                                        << checkpointHashAll[r] << "\n";
                                }
                                replicaManifest << "complete=1\n";
                                ftPublishTextFileAtomically(
                                    checkpointReplicaWriteBase + ".manifest",
                                    replicaManifest.str());
                                if (ftRollingCheckpoints) {
                                    ftPublishTextFileAtomically(
                                        ftCheckpointReplicaBase + ".latest",
                                        checkpointReplicaWriteBase + "\n");
                                }
                            }
                            if (ftRollingCheckpoints) {
                                ftPruneOldGenerations(ftCheckpointBase,
                                                      ftKeepCheckpoints);
                                if (!ftCheckpointReplicaBase.empty()) {
                                    ftPruneOldGenerations(
                                        ftCheckpointReplicaBase,
                                        ftKeepCheckpoints);
                                }
                            }
                        } catch (const std::exception& e) {
                            std::cerr << "[FT] manifest publish failed: "
                                      << e.what() << "\n";
                            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                        }
                        std::cout << "[FT] checkpointed depth " << depth
                                  << " to base " << checkpointWriteBase
                                  << "\n";
                        if (!checkpointReplicaWriteBase.empty()) {
                            std::cout << "[FT] replicated checkpoint depth "
                                      << depth << " to base "
                                      << checkpointReplicaWriteBase << "\n";
                        }
                    }

                    MPI_Barrier(MPI_COMM_WORLD);

                    if (ftStopAfterDepth == depth) {
                        if (world_rank == 0) {
                            std::cout << "[FT] intentional stop after "
                                      << "checkpointed depth " << depth
                                      << "\n";
                        }
                        ftRequestedStop = true;
                        break;
                    }

                    if (ftSignalRequested()) {
                        if (world_rank == 0) {
                            std::cout << "[FT] signal "
                                      << ftRequestedSignalNumber()
                                      << " observed; stopping after "
                                      << "checkpointed depth " << depth
                                      << "\n";
                        }
                        ftRequestedStop = true;
                        break;
                    }
                }

                if (ftRequestedStop) {
                    break;
                }

                // ------------------------------------------------------------
                // Exploitation at max depth.
                // ------------------------------------------------------------
                if (depth == activeMaxDepth) {
                    const bool useRandomRestartExploit =
                        fboConfig.exploitation == FBOExploitation::RandomRestartLocalSearch;
                    const int startsPerCell = useRandomRestartExploit
                        ? std::max(1, fboConfig.exploitStarts) : 1;
                    int exploitationIterations = maxIterations;
                    if (fboConfig.exploitation == FBOExploitation::BestCell ||
                        fboConfig.exploitation == FBOExploitation::RandomLocalPerturbation) {
                        exploitationIterations = 0;
                    } else if (fboConfig.exploitLocalIters >= 0) {
                        exploitationIterations = fboConfig.exploitLocalIters;
                    }
                    int exploitationCellLimit = budgetPlan.exploitation;
                    if (!useFBD) {
                        const long long used = globalObjectiveEvaluations();
                        const long long available = std::max(0LL,
                            static_cast<long long>(budget) - used);
                        const long long allowance = std::min<long long>(
                            budgetPlan.exploitation, available);
                        // Reserve the worst-case calls for every selected
                        // start, including coordinate probes in local search.
                        const bool searches = usesAdditiveDeviceObjective(objTy);
                        const long long perStart = 1 + (searches
                            ? 2LL * dimension * std::max(0, exploitationIterations) : 0);
                        exploitationCellLimit = static_cast<int>(
                            allowance / perStart / startsPerCell);
                        if (world_rank == 0) {
                            std::cout << "[Budget] before exploitation used = " << used
                                      << ", remaining = " << available
                                      << ", exploitation cell cap = "
                                      << exploitationCellLimit << "\n";
                        }
                    }
                    if (fboConfig.exploitation == FBOExploitation::None ||
                        (!useFBD && exploitationCellLimit <= 0)) {
                        if (world_rank == 0) {
                            auto finalPoint =
                                Kokkos::create_mirror_view_and_copy(
                                    Kokkos::HostSpace(),
                                    bestSolution);

                            double minVal = finalPoint.extent(0) > 0
                                ? finalPoint(0)
                                : 0.0;
                            double maxVal = minVal;
                            double sum = 0.0;

                            for (int k = 0; k < dimension; ++k) {
                                const double xk = finalPoint(k);
                                sum += xk;
                                if (xk < minVal) minVal = xk;
                                if (xk > maxVal) maxVal = xk;
                            }

                            std::cout << "[exploit] skipped by "
                                      << (useFBD ? "FBD" : "FBO")
                                      << " policy\n";
                            std::cout
                                << std::defaultfloat
                                << std::setprecision(
                                       std::numeric_limits<double>::max_digits10)
                                << "Minimum objective value: "
                                << bestObjectiveValue << "\n";
                            std::cout << "Solution vector statistics:\n";
                            std::cout << "Dimension: " << dimension << "\n";
                            std::cout << "Min value: " << minVal << "\n";
                            std::cout << "Max value: " << maxVal << "\n";
                            std::cout << "Mean value: "
                                      << (dimension > 0
                                              ? sum / static_cast<double>(dimension)
                                              : 0.0)
                                      << "\n";
                            const double dist2 =
                                squaredDistanceToObjectiveReference(finalPoint,
                                                                    objTy);
                            if (std::isfinite(dist2)) {
                                std::cout << "Euclidean distance to optimum: "
                                          << std::sqrt(dist2) << "\n";
                            } else {
                                std::cout << "Euclidean distance to optimum: "
                                          << "unavailable\n";
                            }
                        }
                        break;
                    }

                    PROFILE_SCOPE("exploitation_total");
                    VERIFY_SCOPE_MAX_COMM("exploitation_total(max across ranks)",
                                          MPI_COMM_WORLD);

                    Kokkos::fence();

                    int numSelectedGlobal =
                        static_cast<int>(currentHyperspheres.size());

                    if (!useFBD) {
                        numSelectedGlobal =
                            std::min(numSelectedGlobal,
                                     exploitationCellLimit);
                    }

                    if (world_rank == 0) {
                        std::cout << "[exploit] global tasks considered = "
                                  << numSelectedGlobal << " / "
                                  << currentHyperspheres.size() << "\n";
                    }

                    const int numSelected = numSelectedGlobal;

                    const int numPerNode =
                        numSelected / num_nodes;

                    const int remainder =
                        numSelected % num_nodes;

                    const int startIdx =
                        node_id * numPerNode +
                        std::min(node_id, remainder);

                    const int endIdx =
                        startIdx +
                        numPerNode +
                        (node_id < remainder ? 1 : 0);

                    const int localNumHyperspheres =
                        endIdx - startIdx;

                    const int numPerRank =
                        localNumHyperspheres / node_size;

                    const int remainder_local =
                        localNumHyperspheres % node_size;

                    const int localStartIdx =
                        startIdx +
                        node_rank * numPerRank +
                        std::min(node_rank, remainder_local);

                    const int localEndIdx =
                        localStartIdx +
                        numPerRank +
                        (node_rank < remainder_local ? 1 : 0);

                    std::vector<Hypersphere> myHyperspheres;

                    for (int i = localStartIdx; i < localEndIdx; ++i) {
                        myHyperspheres.push_back(currentHyperspheres[i]);
                    }

                    std::cout << "[rank " << world_rank
                              << "] exploit H_local = "
                              << static_cast<int>(myHyperspheres.size())
                              << " (world_size=" << world_size
                              << ", exploit_budget="
                              << (useFBD ? numSelectedGlobal
                                         : budgetPlan.exploitation)
                              << ")\n";

                    double localBestObjective =
                        std::numeric_limits<double>::max();

                    int localBestIndex = -1;

                    Kokkos::View<double*,
                                 Kokkos::LayoutLeft,
                                 mem_space>
                        localBestPoint("localBestPoint", dimension);

                    Kokkos::View<int*, mem_space> e_depths;
                    Kokkos::View<size_t*, mem_space> e_offs;
                    Kokkos::View<int*, mem_space> e_dims;
                    Kokkos::View<int*, mem_space> e_signs;

                    prepareHypersphereData(myHyperspheres,
                                           e_depths,
                                           e_offs,
                                           e_dims,
                                           e_signs);

                    Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> e_centers;
                    Kokkos::View<double*, mem_space> e_radii;

                    {
                        PROFILE_SCOPE("exploit:reconstruct_centers");
                        Kokkos::fence();

                        Kokkos::View<double*, mem_space>
                            alphaDepthDev("alphaDepthDev_exploit",
                                          alphaDepthHost.size());

                        {
                            auto hA =
                                Kokkos::create_mirror_view(alphaDepthDev);

                            for (size_t i = 0;
                                 i < alphaDepthHost.size();
                                 ++i) {
                                hA(static_cast<int>(i)) =
                                    alphaDepthHost[i];
                            }

                            Kokkos::deep_copy(alphaDepthDev, hA);
                        }

                        reconstructCentersAndRadii(
                            static_cast<size_t>(myHyperspheres.size()),
                            dimension,
                            initialRadius,
                            e_depths,
                            e_offs,
                            e_dims,
                            e_signs,
                            initialCenter,
                            alphaDepthDev,
                            e_centers,
                            e_radii,
                            decompositionKind);

                        Kokkos::fence();
                    }

                    std::vector<double> exploitationSigma(
                        myHyperspheres.size(),
                        1.0);
                    double exploitationSigmaMedian = 1.0;
                    if (fboConfig.adaptiveExploitationScale &&
                        useGPSurrogate &&
                        !myHyperspheres.empty()) {
                        Kokkos::View<double**, Kokkos::LayoutLeft, mem_space>
                            projectedCenters(
                                "exploit_projected_centers",
                                myHyperspheres.size(),
                                d_gpModel.d_latent);
                        Kokkos::View<const double**, Kokkos::LayoutLeft, mem_space>
                            constExploitCenters = e_centers;
                        GP::projectOnDevice(d_gpModel,
                                            constExploitCenters,
                                            projectedCenters);
                        Kokkos::View<double*, mem_space>
                            exploitMu("exploit_mu",
                                      myHyperspheres.size());
                        Kokkos::View<double*, mem_space>
                            exploitVar("exploit_var",
                                       myHyperspheres.size());
                        GP::batchPredictGP(d_gpModel,
                                           projectedCenters,
                                           exploitMu,
                                           exploitVar);
                        auto hVar =
                            Kokkos::create_mirror_view_and_copy(
                                Kokkos::HostSpace(),
                                exploitVar);
                        for (size_t i = 0;
                             i < exploitationSigma.size();
                             ++i) {
                            exploitationSigma[i] =
                                std::sqrt(std::max(
                                    1e-12,
                                    hVar(static_cast<int>(i))));
                        }
                        exploitationSigmaMedian =
                            std::max(1e-12,
                                     medianHost(exploitationSigma));
                    }

                    const size_t free_bytes = get_available_memory();
                    const size_t per_h =
                        3ull *
                        static_cast<size_t>(dimension) *
                        sizeof(double) *
                        static_cast<size_t>(startsPerCell);

                    const size_t budget_b =
                        static_cast<size_t>(
                            0.70 * static_cast<double>(free_bytes));

                    const int maxBatch =
                        std::max(
                            1,
                            static_cast<int>(
                                std::min(
                                    static_cast<size_t>(
                                        myHyperspheres.size()),
                                    per_h ? (budget_b / per_h)
                                          : static_cast<size_t>(1))));

                    if (world_rank == 0) {
                        std::cout << "[exploit] policy = "
                                  << toString(fboConfig.exploitation)
                                  << ", local iterations = "
                                  << exploitationIterations
                                  << ", starts/cell = "
                                  << startsPerCell
                                  << ", radius scale = "
                                  << fboConfig.exploitRadiusScale << "\n";
                    }

                    for (int s = 0;
                         s < static_cast<int>(myHyperspheres.size());
                         s += maxBatch) {
                        const int h =
                            std::min(maxBatch,
                                     static_cast<int>(
                                         myHyperspheres.size()) -
                                         s);

                        auto centers_sub =
                            Kokkos::subview(e_centers,
                                            std::make_pair(s, s + h),
                                            Kokkos::ALL());

                        auto radii_sub =
                            Kokkos::subview(e_radii,
                                            std::make_pair(s, s + h));

                        const int evalCount = h * startsPerCell;
                        double stepSizeBatch = stepSize;
                        if (fboConfig.adaptiveExploitationScale) {
                            double sigmaMean = 0.0;
                            for (int i = 0; i < h; ++i) {
                                sigmaMean += exploitationSigma[
                                    static_cast<size_t>(s + i)];
                            }
                            sigmaMean /= static_cast<double>(
                                std::max(1, h));
                            const double uncertaintyRatio =
                                sigmaMean /
                                (exploitationSigmaMedian + 1e-12);
                            stepSizeBatch = std::clamp(
                                stepSize *
                                    (1.0 + 0.75 * uncertaintyRatio),
                                omega_min,
                                0.75 * initialRadius);
                            if (world_rank == 0) {
                                std::cout
                                    << "[exploit-guidance] batch_start="
                                    << s
                                    << " uncertainty_ratio="
                                    << uncertaintyRatio
                                    << " step=" << stepSizeBatch
                                    << " iterations="
                                    << exploitationIterations
                                    << "\n";
                            }
                        }

                        Kokkos::View<double*, mem_space>
                            bestVals("bestVals", evalCount);

                        Kokkos::View<double**, Kokkos::LayoutLeft, mem_space>
                            finalPos("finalPos", evalCount, dimension);

                        {
                            PROFILE_SCOPE("exploit:ILS_batch");
                            Kokkos::fence();

                            if (fboConfig.exploitation ==
                                    FBOExploitation::RandomLocalPerturbation ||
                                useRandomRestartExploit) {
                                Kokkos::View<double**, Kokkos::LayoutLeft, mem_space>
                                    randomStarts("random_exploit_starts",
                                                 evalCount,
                                                 dimension);

                                using MD2 =
                                    Kokkos::MDRangePolicy<Kokkos::Rank<2>,
                                                          exec_space>;

                                const uint64_t exploitSeed =
                                    static_cast<uint64_t>(seed + 97) *
                                    0x9E3779B97F4A7C15ull;
                                const int batchOffset = s;
                                const double radiusScale =
                                    fboConfig.exploitRadiusScale;

                                Kokkos::parallel_for(
                                    "exploit_random_perturb",
                                    MD2({0, 0}, {evalCount, dimension}),
                                    KOKKOS_LAMBDA(int i, int d) {
                                        const int cell =
                                            i / startsPerCell;
                                        const int start =
                                            i - cell * startsPerCell;
                                        if (start == 0 &&
                                            useRandomRestartExploit) {
                                            randomStarts(i, d) =
                                                centers_sub(cell, d);
                                            return;
                                        }

                                        uint64_t x =
                                            static_cast<uint64_t>(
                                                batchOffset +
                                                cell * startsPerCell +
                                                start + 1);
                                        x ^= exploitSeed +
                                             static_cast<uint64_t>(d + 1) *
                                                 0xBF58476D1CE4E5B9ull;
                                        x ^= x >> 30;
                                        x *= 0xBF58476D1CE4E5B9ull;
                                        x ^= x >> 27;
                                        x *= 0x94D049BB133111EBull;
                                        x ^= x >> 31;
                                        const double u =
                                            static_cast<double>(
                                                x & 0x1FFFFFFFFFFFFFull) /
                                            static_cast<double>(
                                                0x20000000000000ull);
                                        double xd =
                                            centers_sub(cell, d) +
                                            (2.0 * u - 1.0) *
                                                radiusScale *
                                                radii_sub(cell);
                                        if (xd < lowerBounds(d)) {
                                            xd = lowerBounds(d);
                                        }
                                        if (xd > upperBounds(d)) {
                                            xd = upperBounds(d);
                                        }
                                        randomStarts(i, d) = xd;
                                    });

                                intensiveLocalSearchBatch(randomStarts,
                                                          exploitationIterations,
                                                          stepSizeBatch,
                                                          phi,
                                                          omega_min,
                                                          objTy,
                                                          expenseFactor,
                                                          bestVals,
                                                          finalPos);
                            } else {
                                intensiveLocalSearchBatch(centers_sub,
                                                          exploitationIterations,
                                                          stepSizeBatch,
                                                          phi,
                                                          omega_min,
                                                          objTy,
                                                          expenseFactor,
                                                          bestVals,
                                                          finalPos);
                            }

                            Kokkos::fence();
                        }

                        MinAcqIndex best{DBL_MAX, -1};

                        {
                            PROFILE_SCOPE("exploit:argmin_batch");

                            Kokkos::parallel_reduce(
                                "argmin_exploit_batch",
                                Kokkos::RangePolicy<exec_space>(0, evalCount),
                                KOKKOS_LAMBDA(int i,
                                              MinAcqIndex& upd) {
                                    const double v = bestVals(i);

                                    if (v < upd.value) {
                                        upd.value = v;
                                        upd.index = i;
                                    }
                                },
                                Kokkos::Min<MinAcqIndex>(best));
                        }

                        if (best.index >= 0 &&
                            best.value < localBestObjective) {
                            localBestObjective = best.value;
                            localBestIndex =
                                s + best.index / startsPerCell;

                            auto bestRow =
                                Kokkos::subview(finalPos,
                                                best.index,
                                                Kokkos::ALL());

                            Kokkos::deep_copy(localBestPoint, bestRow);
                        }
                    }

                    if (localBestIndex < 0) {
                        Kokkos::deep_copy(localBestPoint, 0.0);
                    }

                    struct {
                        double value;
                        int rank;
                    } nodeLocalMin{localBestObjective, node_rank},
                        nodeBestMin;

                    {
                        PROFILE_SCOPE("exploit:intra_node_minloc");

                        MPI_Allreduce(&nodeLocalMin,
                                      &nodeBestMin,
                                      1,
                                      MPI_DOUBLE_INT,
                                      MPI_MINLOC,
                                      node_comm);
                    }

                    const double nodeBestObjective = nodeBestMin.value;

                    Kokkos::View<double*, Kokkos::HostSpace>
                        nodeBestPointHost("nodeBestPointHost", dimension);

                    if (node_rank == nodeBestMin.rank) {
                        auto localPointHost =
                            Kokkos::create_mirror_view_and_copy(
                                Kokkos::HostSpace(),
                                localBestPoint);

                        for (int d = 0; d < dimension; ++d) {
                            nodeBestPointHost(d) =
                                localPointHost(d);
                        }
                    }

                    {
                        PROFILE_SCOPE("exploit:bcast_node_best");

                        MPI_Bcast(nodeBestPointHost.data(),
                                  dimension,
                                  MPI_DOUBLE,
                                  nodeBestMin.rank,
                                  node_comm);
                    }

                    Kokkos::View<double*, mem_space>
                        nodeBestPoint("nodeBestPoint", dimension);

                    Kokkos::deep_copy(nodeBestPoint,
                                      nodeBestPointHost);

                    struct {
                        double value;
                        int rank;
                    } globalMin;

                    if (node_rank == 0) {
                        PROFILE_SCOPE("exploit:inter_node_minloc");

                        struct {
                            double value;
                            int rank;
                        } nodeLeaderMin{nodeBestObjective, node_id};

                        MPI_Allreduce(&nodeLeaderMin,
                                      &globalMin,
                                      1,
                                      MPI_DOUBLE_INT,
                                      MPI_MINLOC,
                                      node_leader_comm);
                    }

                    MPI_Bcast(&globalMin,
                              1,
                              MPI_DOUBLE_INT,
                              0,
                              node_comm);

                    Kokkos::View<double*, Kokkos::HostSpace>
                        globalBestPointHost("globalBestPointHost",
                                            dimension);

                    if (node_rank == 0) {
                        if (node_id == globalMin.rank) {
                            PROFILE_SCOPE("exploit:bcast_leader_best_out");

                            auto tmpHost =
                                Kokkos::create_mirror_view_and_copy(
                                    Kokkos::HostSpace(),
                                    nodeBestPoint);

                            MPI_Bcast(tmpHost.data(),
                                      dimension,
                                      MPI_DOUBLE,
                                      globalMin.rank,
                                      node_leader_comm);

                            Kokkos::deep_copy(globalBestPointHost,
                                              tmpHost);
                        } else {
                            PROFILE_SCOPE("exploit:bcast_leader_best_in");

                            Kokkos::View<double*, Kokkos::HostSpace>
                                tmpHost("tmpHost", dimension);

                            MPI_Bcast(tmpHost.data(),
                                      dimension,
                                      MPI_DOUBLE,
                                      globalMin.rank,
                                      node_leader_comm);

                            Kokkos::deep_copy(globalBestPointHost,
                                              tmpHost);
                        }
                    }

                    {
                        PROFILE_SCOPE("exploit:bcast_global_best_to_node");

                        MPI_Bcast(globalBestPointHost.data(),
                                  dimension,
                                  MPI_DOUBLE,
                                  0,
                                  node_comm);
                    }

                    auto incHost2 =
                        Kokkos::create_mirror_view_and_copy(
                            Kokkos::HostSpace(),
                            bestSolution);

                    Kokkos::View<double*, Kokkos::HostSpace>
                        finalBestPoint("finalBestPoint", dimension);

                    for (int d = 0; d < dimension; ++d) {
                        finalBestPoint(d) = incHost2(d);
                    }

                    double finalBestVal = bestObjectiveValue;

                    if (globalMin.value < finalBestVal) {
                        finalBestVal = globalMin.value;

                        for (int d = 0; d < dimension; ++d) {
                            finalBestPoint(d) =
                                globalBestPointHost(d);
                        }
                    }

                    struct {
                        double val;
                        int rank;
                    } cand{finalBestVal, world_rank}, winner2;

                    MPI_Allreduce(&cand,
                                  &winner2,
                                  1,
                                  MPI_DOUBLE_INT,
                                  MPI_MINLOC,
                                  MPI_COMM_WORLD);

                    Kokkos::View<double*, Kokkos::HostSpace>
                        finalPointToShare("finalPointToShare",
                                          dimension);

                    if (world_rank == winner2.rank) {
                        for (int d = 0; d < dimension; ++d) {
                            finalPointToShare(d) =
                                finalBestPoint(d);
                        }
                    }

                    {
                        PROFILE_SCOPE("exploit:bcast_final_point_world");

                        MPI_Bcast(finalPointToShare.data(),
                                  dimension,
                                  MPI_DOUBLE,
                                  winner2.rank,
                                  MPI_COMM_WORLD);
                    }

                    bestObjectiveValue = winner2.val;
                    Kokkos::deep_copy(bestSolution,
                                      finalPointToShare);

                    if (world_rank == 0) {
                        std::cout
                            << std::defaultfloat
                            << std::setprecision(
                                   std::numeric_limits<double>::max_digits10)
                            << "Minimum objective value: "
                            << bestObjectiveValue
                            << std::endl;

                        const double dist2 =
                            squaredDistanceToObjectiveReference(
                                finalPointToShare,
                                objTy);

                        const double euclidDist = std::isfinite(dist2)
                            ? std::sqrt(dist2)
                            : std::numeric_limits<double>::quiet_NaN();

                        double minVal = finalPointToShare(0);
                        double maxVal = finalPointToShare(0);
                        double sum = 0.0;

                        for (size_t k = 0;
                             k < finalPointToShare.extent(0);
                             ++k) {
                            const double v = finalPointToShare(k);

                            sum += v;

                            if (v < minVal) minVal = v;
                            if (v > maxVal) maxVal = v;
                        }

                        const double meanVal =
                            sum / finalPointToShare.extent(0);

                        std::cout << "Entire solution found: [";

                        for (size_t k = 0;
                             k < finalPointToShare.extent(0);
                             ++k) {
                            std::cout << finalPointToShare(k);

                            if (k + 1 < finalPointToShare.extent(0)) {
                                std::cout << ", ";
                            }
                        }

                        std::cout << "]\n";

                        std::cout << "Solution vector statistics:\n";
                        std::cout << "Dimension: "
                                  << finalPointToShare.extent(0)
                                  << "\n";
                        std::cout << "Min value: " << minVal << "\n";
                        std::cout << "Max value: " << maxVal << "\n";
                        std::cout << "Mean value: " << meanVal << "\n";
                        std::cout
                            << std::defaultfloat
                            << std::setprecision(
                                   std::numeric_limits<double>::max_digits10)
                            << "Minimum objective value: "
                            << bestObjectiveValue << "\n";
                        if (std::isfinite(euclidDist)) {
                            std::cout << "Euclidean distance to optimum: "
                                      << euclidDist << "\n";
                        } else {
                            std::cout << "Euclidean distance to optimum: "
                                      << "unavailable\n";
                        }
                    }

                    break;
                } else {
                    if (world_rank == 0) {
                        std::cout << "Completed depth: "
                                  << depth
                                  << std::endl;
                    }
                }
            }
        }

        long long fbdGlobalExplorationEvaluations = 0;
        if (useFBD) {
            MPI_Reduce(&fbdLocalExplorationEvaluations,
                       &fbdGlobalExplorationEvaluations,
                       1,
                       MPI_LONG_LONG,
                       MPI_SUM,
                       0,
                       MPI_COMM_WORLD);
        }

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> duration = end - start;

        long long localEvaluationCounts[2] = {
            objectiveEvaluationAudit().device.load(), objectiveEvaluationAudit().host.load()};
        long long globalEvaluationCounts[2] = {0,0};
        MPI_Reduce(localEvaluationCounts, globalEvaluationCounts, 2, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        std::vector<long long> rankEvaluationCounts(static_cast<size_t>(2 * world_size));
        MPI_Gather(localEvaluationCounts, 2, MPI_LONG_LONG, rankEvaluationCounts.data(), 2, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
        const auto resultPath=parseOptionalStringArg(argc,argv,"--result-json","");
        if(world_rank==0 && !resultPath.empty()) {
            auto point=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),bestSolution);
            std::ofstream result(resultPath);
            if(!result || !std::isfinite(bestObjectiveValue)) {
                std::cerr << "Cannot write a finite optimization result to " << resultPath << '\n';
                MPI_Abort(MPI_COMM_WORLD,1);
            }
            result << std::setprecision(17) << "{\"schema_version\":1,\"family\":\""
                   << toString(optimizerFamily) << "\",\"fun\":" << bestObjectiveValue
                   << ",\"elapsed_seconds\":" << duration.count()
                   << ",\"device_evaluations\":" << globalEvaluationCounts[0]
                   << ",\"host_evaluations\":" << globalEvaluationCounts[1]
                   << ",\"evaluations\":" << globalEvaluationCounts[0] + globalEvaluationCounts[1]
#if defined(KOKKOS_ENABLE_CUDA)
                   << ",\"objective_backend\":\"CUDA\""
#elif defined(KOKKOS_ENABLE_HIP)
                   << ",\"objective_backend\":\"HIP\""
#else
                   << ",\"objective_backend\":\"CPU\""
#endif
                   << ",\"rank_evaluations\":[";
            for(int r=0;r<world_size;++r) { if(r) result << ','; result << rankEvaluationCounts[2*r] + rankEvaluationCounts[2*r+1]; }
            result << "],\"x\":[";
            for(int j=0;j<dimension;++j) { if(j) result << ','; result << point(j); }
            result << "]}\n";
            result.close();
            if(!result) MPI_Abort(MPI_COMM_WORLD,1);
        }

        if (world_rank == 0) {
            if (useFBD) {
                const long long finalEvaluations =
                    static_cast<long long>(nInitGlobal) +
                    fbdGlobalExplorationEvaluations;
                std::cout << "Final evaluations: "
                          << finalEvaluations << "\n";
            }

            std::cout << "Total Execution time: "
                      << duration.count()
                      << " seconds"
                      << std::endl;

            timing::Registry::instance().print(std::cout);
        }
    }

    Kokkos::finalize();

    if (node_rank == 0) {
        MPI_Comm_free(&node_leader_comm);
    }

    MPI_Comm_free(&node_comm);

    MPI_Finalize();

    return 0;
}
