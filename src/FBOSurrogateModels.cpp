#include "FBOSurrogateModels.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace FBO {

bool isGPSurrogate(FBOSurrogate kind)
{
    return kind == FBOSurrogate::PcaGp || kind == FBOSurrogate::HsicGp;
}

bool isSimpleSurrogate(FBOSurrogate kind)
{
    return kind == FBOSurrogate::InverseDistance ||
           kind == FBOSurrogate::RbfKernel;
}

double estimateLengthScale(const std::vector<double>& X, int N, int D)
{
    if (N < 2 || D <= 0 || X.size() < static_cast<size_t>(N) * D) {
        return 1.0;
    }

    const int maxPairs = 4096;
    int pairs = 0;
    double sumDist2 = 0.0;

    for (int i = 0; i < N && pairs < maxPairs; ++i) {
        for (int j = i + 1; j < N && pairs < maxPairs; ++j) {
            double dist2 = 0.0;
            for (int d = 0; d < D; ++d) {
                const double diff =
                    X[static_cast<size_t>(i) * D + d] -
                    X[static_cast<size_t>(j) * D + d];
                dist2 += diff * diff;
            }
            sumDist2 += dist2;
            ++pairs;
        }
    }

    if (pairs <= 0 || sumDist2 <= 0.0) {
        return 1.0;
    }

    const double rmsDistance = std::sqrt(sumDist2 / pairs);
    return std::max(1e-9, rmsDistance / std::sqrt(2.0));
}

SimpleSurrogateDevice makeSimpleSurrogateDevice(
    const std::vector<double>& X,
    const std::vector<double>& y,
    int D,
    FBOSurrogate kind,
    double requestedLengthScale,
    double idwPower)
{
    if (!isSimpleSurrogate(kind)) {
        throw std::runtime_error(
            "makeSimpleSurrogateDevice: unsupported surrogate kind");
    }
    if (D <= 0) {
        throw std::runtime_error(
            "makeSimpleSurrogateDevice: dimension must be positive");
    }

    const int N = static_cast<int>(y.size());
    if (X.size() != static_cast<size_t>(N) * static_cast<size_t>(D)) {
        throw std::runtime_error(
            "makeSimpleSurrogateDevice: X/y shapes are inconsistent");
    }

    SimpleSurrogateDevice model;
    model.N = N;
    model.D = D;
    model.kind = kind;
    model.lengthScale = requestedLengthScale > 0.0
        ? requestedLengthScale
        : estimateLengthScale(X, N, D);
    model.idwPower = idwPower > 0.0 ? idwPower : 2.0;

    double sum = 0.0;
    for (double v : y) sum += v;
    model.yMean = N > 0 ? sum / static_cast<double>(N) : 0.0;

    double ss = 0.0;
    for (double v : y) {
        const double c = v - model.yMean;
        ss += c * c;
    }
    model.yVar = N > 1 ? ss / static_cast<double>(N - 1) : 1.0;
    if (!(model.yVar > 1e-18)) model.yVar = 1.0;

    model.X = Kokkos::View<double**, layout_t, mem_space>(
        "fbo_simple_surrogate_X",
        N,
        D);
    model.y = Kokkos::View<double*, layout_t, mem_space>(
        "fbo_simple_surrogate_y",
        N);

    auto hX = Kokkos::create_mirror_view(model.X);
    auto hy = Kokkos::create_mirror_view(model.y);

    for (int i = 0; i < N; ++i) {
        hy(i) = y[static_cast<size_t>(i)];
        for (int d = 0; d < D; ++d) {
            hX(i, d) = X[static_cast<size_t>(i) * D + d];
        }
    }

    Kokkos::deep_copy(model.X, hX);
    Kokkos::deep_copy(model.y, hy);

    return model;
}

void predictSimpleSurrogate(const SimpleSurrogateDevice& model,
                            const Kokkos::View<double**, Kokkos::LayoutLeft, mem_space>& Xstar,
                            Kokkos::View<double*, mem_space> mu,
                            Kokkos::View<double*, mem_space> var)
{
    const int M = static_cast<int>(Xstar.extent(0));
    const int N = model.N;
    const int D = model.D;
    const FBOSurrogate kind = model.kind;
    const double yMean = model.yMean;
    const double yVar = model.yVar;
    const double lengthScale = model.lengthScale > 1e-12
        ? model.lengthScale
        : 1.0;
    const double invTwoEll2 = 0.5 / (lengthScale * lengthScale);
    const double idwPower = model.idwPower > 0.0 ? model.idwPower : 2.0;
    const auto Xtrain = model.X;
    const auto ytrain = model.y;

    Kokkos::parallel_for(
        "FBO_simple_surrogate_predict",
        Kokkos::RangePolicy<exec_space>(0, M),
        KOKKOS_LAMBDA(int i) {
            if (N <= 0 || D <= 0) {
                mu(i) = yMean;
                var(i) = yVar;
                return;
            }

            double sumW = 0.0;
            double sumWY = 0.0;
            double sumWY2 = 0.0;
            bool exact = false;
            double exactY = yMean;

            for (int n = 0; n < N; ++n) {
                double dist2 = 0.0;
                for (int d = 0; d < D; ++d) {
                    const double diff = Xstar(i, d) - Xtrain(n, d);
                    dist2 += diff * diff;
                }

                const double yn = ytrain(n);
                if (dist2 <= 1e-24) {
                    exact = true;
                    exactY = yn;
                    break;
                }

                double w = 0.0;
                if (kind == FBOSurrogate::InverseDistance) {
                    w = 1.0 / pow(sqrt(dist2) + 1e-12, idwPower);
                } else {
                    w = exp(-dist2 * invTwoEll2);
                }

                sumW += w;
                sumWY += w * yn;
                sumWY2 += w * yn * yn;
            }

            if (exact) {
                mu(i) = exactY;
                var(i) = 1e-12;
                return;
            }

            if (!(sumW > 1e-300)) {
                mu(i) = yMean;
                var(i) = yVar;
                return;
            }

            const double mean = sumWY / sumW;
            double vv = sumWY2 / sumW - mean * mean;
            if (!(vv > 1e-12)) vv = 1e-12;

            mu(i) = mean;
            var(i) = vv + yVar / (1.0 + sumW);
        });
}

} // namespace FBO
