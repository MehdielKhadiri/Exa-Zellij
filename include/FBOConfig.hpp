#ifndef FBO_CONFIG_HPP
#define FBO_CONFIG_HPP

#include "FBOTypes.hpp"

#include <iosfwd>
#include <string>

struct FBOConfig {
    FBODecomposition decomposition = FBODecomposition::Hypersphere;
    FBOSplitPolicy splitPolicy = FBOSplitPolicy::Cyclic;
    FBODoe doe = FBODoe::Sobol;
    FBODoeDistribution doeDistribution = FBODoeDistribution::Replicated;
    FBOSurrogate surrogate = FBOSurrogate::PcaGp;
    FBOGPKernel gpKernel = FBOGPKernel::SquaredExponential;
    FBOExploitation exploitation = FBOExploitation::SurrogateLocalSearch;
    FBOBudgetMode budgetMode = FBOBudgetMode::Astra;
    bool adaptiveHypersphereRadius = false;
    bool adaptiveExploitationScale = false;

    double doeFraction = -1.0;
    double exploreFraction = -1.0;
    double exploitFraction = 0.10;
    double surrogateLengthScale = -1.0;
    double idwPower = 2.0;
    double gpSigmaF = 1.0;
    double gpLengthScale = 1.0;
    double gpNoise = 1e-3;
    double gpRQAlpha = 1.0;
    int exploitStarts = 4;
    int exploitLocalIters = -1;
    double exploitRadiusScale = 0.5;

    // Hypercube defaults to a block decomposition; <=0 requests all dimensions.
    int hypercubeActiveDims = 12;
    int voronoiSites = 0;
    int simplexChildren = 0;
};

FBOConfig parseFBOConfig(int argc, char** argv, int worldRank);

std::string toString(FBODecomposition v);
std::string toString(FBOSplitPolicy v);
std::string toString(FBODoe v);
std::string toString(FBODoeDistribution v);
std::string toString(FBOSurrogate v);
std::string toString(FBOGPKernel v);
std::string toString(FBOExploitation v);
std::string toString(FBOBudgetMode v);

void printFBOConfig(const FBOConfig& cfg, std::ostream& os);

#endif // FBO_CONFIG_HPP
