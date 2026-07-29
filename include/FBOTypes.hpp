#ifndef FBO_TYPES_HPP
#define FBO_TYPES_HPP

#include <Kokkos_Core.hpp>

#include <cstddef>

enum class FBODecomposition {
    Hypersphere,
    Hypercube,
    Bisection,
    Trisection,
    Voronoi,
    Simplex
};

enum class FBOSplitPolicy {
    Cyclic,
    LongestSide,
    Random
};

enum class FBODoe {
    Sobol,
    DoeFile,
    Random,
    LatinHypercube,
    Halton,
    Hammersley
};

enum class FBODoeDistribution {
    Replicated,
    Distributed
};

enum class FBOSurrogate {
    PcaGp,
    HsicGp,
    InverseDistance,
    RbfKernel,
    None
};

enum class FBOGPKernel {
    SquaredExponential,
    Exponential,
    Matern32,
    Matern52,
    RationalQuadratic
};

enum class FBOExploitation {
    SurrogateLocalSearch,
    RandomLocalPerturbation,
    RandomRestartLocalSearch,
    BestCell,
    None
};

enum class FBOBudgetMode {
    Astra,
    FixedFractions
};

KOKKOS_INLINE_FUNCTION
int fboClampPositive(const int value, const int fallback)
{
    return value > 0 ? value : fallback;
}

KOKKOS_INLINE_FUNCTION
int fboActiveHypercubeDims(const int dimension, const int requested)
{
    if (dimension <= 0) return 0;
    const int active = fboClampPositive(requested, dimension);
    return active < dimension ? active : dimension;
}

KOKKOS_INLINE_FUNCTION
int fboPow2Int(const int exponent)
{
    if (exponent <= 0) return 1;
    if (exponent >= 30) return 1 << 30;
    return 1 << exponent;
}

KOKKOS_INLINE_FUNCTION
int fboChildrenPerParent(const int dimension,
                         const FBODecomposition kind,
                         const int hypercubeActiveDims = 12,
                         const int voronoiSites = 0,
                         const int simplexChildren = 0)
{
    if (dimension <= 0) return 0;

    switch (kind) {
        case FBODecomposition::Hypercube:
            return fboPow2Int(
                fboActiveHypercubeDims(dimension, hypercubeActiveDims));
        case FBODecomposition::Bisection:
            return 2;
        case FBODecomposition::Trisection:
            return 3;
        case FBODecomposition::Voronoi:
            return fboClampPositive(voronoiSites, 2 * dimension + 1);
        case FBODecomposition::Simplex:
            return fboClampPositive(simplexChildren, dimension + 1);
        case FBODecomposition::Hypersphere:
        default:
            return 2 * dimension;
    }
}

KOKKOS_INLINE_FUNCTION
bool fboUsesIndependentIntervalWidths(const FBODecomposition kind)
{
    switch (kind) {
        case FBODecomposition::Hypercube:
        case FBODecomposition::Bisection:
        case FBODecomposition::Trisection:
            return true;
        default:
            return false;
    }
}

KOKKOS_INLINE_FUNCTION
double fboRadiusFactor(const FBODecomposition kind)
{
    switch (kind) {
        case FBODecomposition::Hypercube:
        case FBODecomposition::Bisection:
            return 0.5;
        case FBODecomposition::Trisection:
            return 1.0 / 3.0;
        case FBODecomposition::Simplex:
            return 0.5;
        case FBODecomposition::Voronoi:
        case FBODecomposition::Hypersphere:
        default:
            return 1.0 / (1.0 + 1.4142135623730951);
    }
}

KOKKOS_INLINE_FUNCTION
double fboChildOffset(const double radius,
                      const FBODecomposition kind,
                      const int splitCode)
{
    if (splitCode == 0) return 0.0;

    if (fboUsesIndependentIntervalWidths(kind)) {
        const double childHalfWidth = radius * fboRadiusFactor(kind);
        if (kind == FBODecomposition::Trisection) {
            return static_cast<double>(splitCode) * 2.0 * childHalfWidth;
        }
        return static_cast<double>(splitCode > 0 ? 1 : -1) * childHalfWidth;
    }

    const double nextRadius = radius * fboRadiusFactor(kind);
    return static_cast<double>(splitCode > 0 ? 1 : -1) * (radius - nextRadius);
}

template <class DimsView, class SignsView>
KOKKOS_INLINE_FUNCTION
double fboPathMoveOffset(const double initialRadius,
                         const FBODecomposition kind,
                         const DimsView& idDims,
                         const SignsView& idSigns,
                         const std::size_t start,
                         const int moveIndex)
{
    const int dim = idDims(start + static_cast<std::size_t>(moveIndex));
    const int code = idSigns(start + static_cast<std::size_t>(moveIndex));

    if (code == 0) return 0.0;

    if (fboUsesIndependentIntervalWidths(kind)) {
        double halfWidth = initialRadius;
        const double factor = fboRadiusFactor(kind);

        for (int prev = 0; prev < moveIndex; ++prev) {
            const int prevDim = idDims(start + static_cast<std::size_t>(prev));
            if (prevDim == dim) {
                halfWidth *= factor;
            }
        }

        return fboChildOffset(halfWidth, kind, code);
    }

    double radius = initialRadius;
    const double factor = fboRadiusFactor(kind);
    for (int prev = 0; prev < moveIndex; ++prev) {
        radius *= factor;
    }
    return fboChildOffset(radius, kind, code);
}

template <class DimsView, class SignsView, class AlphaView>
KOKKOS_INLINE_FUNCTION
double fboPathMoveOffset(const double initialRadius,
                         const FBODecomposition kind,
                         const DimsView& idDims,
                         const SignsView& idSigns,
                         const std::size_t start,
                         const int moveIndex,
                         const AlphaView& alphaDepth)
{
    if (fboUsesIndependentIntervalWidths(kind)) {
        return fboPathMoveOffset(initialRadius,
                                 kind,
                                 idDims,
                                 idSigns,
                                 start,
                                 moveIndex);
    }

    const int code =
        idSigns(start + static_cast<std::size_t>(moveIndex));
    if (code == 0) return 0.0;

    const double alpha0 = fboRadiusFactor(kind);
    double radius = initialRadius;

    for (int prev = 0; prev < moveIndex; ++prev) {
        const double factor =
            (kind == FBODecomposition::Hypersphere &&
             prev < static_cast<int>(alphaDepth.extent(0)))
                ? alphaDepth(prev)
                : alpha0;
        radius *= factor;
    }

    const double factor =
        (kind == FBODecomposition::Hypersphere &&
         moveIndex < static_cast<int>(alphaDepth.extent(0)))
            ? alphaDepth(moveIndex)
            : alpha0;
    const double nextRadius = radius * factor;
    return static_cast<double>(code > 0 ? 1 : -1) *
           (radius - nextRadius);
}

#endif // FBO_TYPES_HPP
