#ifndef FBO_SURROGATE_MODELS_HPP
#define FBO_SURROGATE_MODELS_HPP

#include "FBOTypes.hpp"

#include <Kokkos_Core.hpp>

#include <vector>

namespace FBO {

using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space = exec_space::memory_space;
using layout_t = Kokkos::LayoutLeft;

struct SimpleSurrogateDevice {
    Kokkos::View<double**, layout_t, mem_space> X;
    Kokkos::View<double*, layout_t, mem_space> y;

    int N = 0;
    int D = 0;
    FBOSurrogate kind = FBOSurrogate::InverseDistance;

    double yMean = 0.0;
    double yVar = 1.0;
    double lengthScale = 1.0;
    double idwPower = 2.0;
};

bool isGPSurrogate(FBOSurrogate kind);
bool isSimpleSurrogate(FBOSurrogate kind);

double estimateLengthScale(const std::vector<double>& X,
                           int N,
                           int D);

SimpleSurrogateDevice makeSimpleSurrogateDevice(
    const std::vector<double>& X,
    const std::vector<double>& y,
    int D,
    FBOSurrogate kind,
    double requestedLengthScale,
    double idwPower);

void predictSimpleSurrogate(const SimpleSurrogateDevice& model,
                            const Kokkos::View<double**, mem_space>& Xstar,
                            Kokkos::View<double*, mem_space> mu,
                            Kokkos::View<double*, mem_space> var);

} // namespace FBO

#endif // FBO_SURROGATE_MODELS_HPP
