#include "FBOConfig.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

std::string lowerCopy(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

bool hasFlagValue(int argc, char** argv, const std::string& flag)
{
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) return true;
    }
    return false;
}

std::string getStringFlag(int argc,
                          char** argv,
                          const std::string& flag,
                          const std::string& fallback)
{
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) return argv[i + 1];
    }
    return fallback;
}

double getDoubleFlag(int argc,
                     char** argv,
                     const std::string& flag,
                     double fallback)
{
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) return std::stod(argv[i + 1]);
    }
    return fallback;
}

int getIntFlag(int argc,
               char** argv,
               const std::string& flag,
               int fallback)
{
    for (int i = 1; i < argc - 1; ++i) {
        if (std::string(argv[i]) == flag) return std::stoi(argv[i + 1]);
    }
    return fallback;
}

FBODecomposition parseDecomposition(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "hypersphere" || s == "astra" || s == "astra-hypersphere") {
        return FBODecomposition::Hypersphere;
    }
    if (s == "hypercube" || s == "cube" || s == "fractal-hypercube") {
        return FBODecomposition::Hypercube;
    }
    if (s == "bisection" || s == "bisect" || s == "axis-bisection" ||
        s == "axis-aligned" || s == "axis") {
        return FBODecomposition::Bisection;
    }
    if (s == "trisection" || s == "interval-trisection" || s == "tri") {
        return FBODecomposition::Trisection;
    }
    if (s == "voronoi" || s == "voronoi-style" || s == "star-voronoi") {
        return FBODecomposition::Voronoi;
    }
    if (s == "simplex" || s == "simplicial" || s == "simplex-lite") {
        return FBODecomposition::Simplex;
    }
    throw std::invalid_argument("unknown --fbo-decomposition value: " + s);
}

FBOSplitPolicy parseSplitPolicy(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "cyclic" || s == "round-robin") return FBOSplitPolicy::Cyclic;
    if (s == "longest-side" || s == "longest" || s == "balanced") {
        return FBOSplitPolicy::LongestSide;
    }
    if (s == "random" || s == "hashed-random" || s == "hash") {
        return FBOSplitPolicy::Random;
    }
    throw std::invalid_argument("unknown --fbo-split-policy value: " + s);
}

FBODoe parseDoe(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "sobol" || s == "sobol-file") return FBODoe::Sobol;
    if (s == "file" || s == "doe-file" || s == "csv") return FBODoe::DoeFile;
    if (s == "random" || s == "uniform-random") return FBODoe::Random;
    if (s == "lhs" || s == "latin" || s == "latin-hypercube") {
        return FBODoe::LatinHypercube;
    }
    if (s == "halton" || s == "halton-sequence") return FBODoe::Halton;
    if (s == "hammersley" || s == "hammersley-set") {
        return FBODoe::Hammersley;
    }
    throw std::invalid_argument("unknown --fbo-doe value: " + s);
}

FBODoeDistribution parseDoeDistribution(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "replicated" || s == "replicate" || s == "broadcast" ||
        s == "shared") {
        return FBODoeDistribution::Replicated;
    }
    if (s == "distributed" || s == "sharded" || s == "shard" ||
        s == "local") {
        return FBODoeDistribution::Distributed;
    }
    throw std::invalid_argument(
        "unknown --fbo-doe-distribution value: " + s);
}

FBOSurrogate parseSurrogate(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "pca-gp" || s == "gp" || s == "astra" || s == "current") {
        return FBOSurrogate::PcaGp;
    }
    if (s == "hsic-gp" || s == "hsic") return FBOSurrogate::HsicGp;
    if (s == "idw" || s == "inverse-distance" ||
        s == "inverse-distance-weighting") {
        return FBOSurrogate::InverseDistance;
    }
    if (s == "rbf" || s == "rbf-kernel" || s == "kernel-smoother" ||
        s == "nadaraya-watson") {
        return FBOSurrogate::RbfKernel;
    }
    if (s == "none" || s == "no" || s == "no-surrogate" || s == "baseline") {
        return FBOSurrogate::None;
    }
    throw std::invalid_argument("unknown --fbo-surrogate value: " + s);
}

FBOGPKernel parseGPKernel(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "rbf" || s == "se" || s == "squared-exponential" ||
        s == "gaussian") {
        return FBOGPKernel::SquaredExponential;
    }
    if (s == "exponential" || s == "matern12" || s == "matern-1/2") {
        return FBOGPKernel::Exponential;
    }
    if (s == "matern32" || s == "matern-3/2" || s == "m32") {
        return FBOGPKernel::Matern32;
    }
    if (s == "matern52" || s == "matern-5/2" || s == "m52") {
        return FBOGPKernel::Matern52;
    }
    if (s == "rq" || s == "rational-quadratic" ||
        s == "rationalquadratic") {
        return FBOGPKernel::RationalQuadratic;
    }
    throw std::invalid_argument("unknown --fbo-gp-kernel value: " + s);
}

FBOExploitation parseExploitation(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "surrogate-local" || s == "surrogate-local-search" ||
        s == "local" || s == "ils" ||
        s == "center-local" || s == "current" || s == "astra") {
        return FBOExploitation::SurrogateLocalSearch;
    }
    if (s == "random-local" || s == "random-perturbation" ||
        s == "local-perturbation" || s == "random-probe") {
        return FBOExploitation::RandomLocalPerturbation;
    }
    if (s == "random-restart" || s == "random-restart-local" ||
        s == "multi-start" || s == "multistart" ||
        s == "stochastic-local") {
        return FBOExploitation::RandomRestartLocalSearch;
    }
    if (s == "best-cell" || s == "center" || s == "cell-center") {
        return FBOExploitation::BestCell;
    }
    if (s == "none" || s == "no" || s == "no-exploitation") {
        return FBOExploitation::None;
    }
    throw std::invalid_argument("unknown --fbo-exploitation value: " + s);
}

FBOBudgetMode parseBudgetMode(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "astra" || s == "current" || s == "default") {
        return FBOBudgetMode::Astra;
    }
    if (s == "fixed" || s == "fixed-fractions" || s == "fractions") {
        return FBOBudgetMode::FixedFractions;
    }
    throw std::invalid_argument("unknown --fbo-budget value: " + s);
}

bool parseAdaptiveHypersphereRadius(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "fixed" || s == "standard" || s == "constant") {
        return false;
    }
    if (s == "adaptive" || s == "acquisition-adaptive" ||
        s == "astra") {
        return true;
    }
    throw std::invalid_argument(
        "unknown --fbo-radius-mode value: " + s +
        " (expected fixed or adaptive)");
}

bool parseAdaptiveExploitationScale(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "fixed" || s == "static" || s == "unguided") {
        return false;
    }
    if (s == "adaptive" || s == "guided" ||
        s == "surrogate-guided") {
        return true;
    }
    throw std::invalid_argument(
        "unknown --fbo-exploitation-guidance value: " + s +
        " (expected fixed or adaptive)");
}

} // namespace

FBOConfig parseFBOConfig(int argc, char** argv, int worldRank)
{
    FBOConfig cfg;

    try {
        cfg.decomposition = parseDecomposition(
            getStringFlag(argc, argv, "--fbo-decomposition", "hypersphere"));
        cfg.splitPolicy = parseSplitPolicy(
            getStringFlag(argc, argv, "--fbo-split-policy", "cyclic"));
        cfg.doe = parseDoe(getStringFlag(argc, argv, "--fbo-doe", "sobol"));
        cfg.doeDistribution = parseDoeDistribution(
            getStringFlag(argc,
                          argv,
                          "--fbo-doe-distribution",
                          "replicated"));
        cfg.surrogate = parseSurrogate(
            getStringFlag(argc, argv, "--fbo-surrogate", "pca-gp"));
        cfg.gpKernel = parseGPKernel(
            getStringFlag(argc, argv, "--fbo-gp-kernel", "rbf"));
        cfg.exploitation = parseExploitation(
            getStringFlag(argc, argv, "--fbo-exploitation", "surrogate-local"));
        cfg.budgetMode = parseBudgetMode(
            getStringFlag(argc, argv, "--fbo-budget", "astra"));
        cfg.adaptiveHypersphereRadius =
            parseAdaptiveHypersphereRadius(
                getStringFlag(argc,
                              argv,
                              "--fbo-radius-mode",
                              "fixed"));
        cfg.adaptiveExploitationScale =
            parseAdaptiveExploitationScale(
                getStringFlag(argc,
                              argv,
                              "--fbo-exploitation-guidance",
                              "fixed"));

        cfg.doeFraction =
            getDoubleFlag(argc, argv, "--doe-frac", cfg.doeFraction);
        cfg.exploreFraction =
            getDoubleFlag(argc, argv, "--explore-frac", cfg.exploreFraction);
        cfg.exploitFraction =
            getDoubleFlag(argc, argv, "--exploit-frac", cfg.exploitFraction);
        cfg.surrogateLengthScale =
            getDoubleFlag(argc,
                          argv,
                          "--fbo-surrogate-length-scale",
                          cfg.surrogateLengthScale);
        cfg.idwPower =
            getDoubleFlag(argc, argv, "--fbo-idw-power", cfg.idwPower);
        cfg.gpSigmaF =
            getDoubleFlag(argc, argv, "--fbo-gp-sigma-f", cfg.gpSigmaF);
        cfg.gpLengthScale =
            getDoubleFlag(argc,
                          argv,
                          "--fbo-gp-length-scale",
                          cfg.gpLengthScale);
        cfg.gpNoise =
            getDoubleFlag(argc, argv, "--fbo-gp-noise", cfg.gpNoise);
        cfg.gpRQAlpha =
            getDoubleFlag(argc, argv, "--fbo-gp-rq-alpha", cfg.gpRQAlpha);
        cfg.exploitStarts =
            getIntFlag(argc,
                       argv,
                       "--fbo-exploit-starts",
                       cfg.exploitStarts);
        cfg.exploitLocalIters =
            getIntFlag(argc,
                       argv,
                       "--fbo-exploit-local-iters",
                       cfg.exploitLocalIters);
        cfg.exploitRadiusScale =
            getDoubleFlag(argc,
                          argv,
                          "--fbo-exploit-radius-scale",
                          cfg.exploitRadiusScale);
        cfg.hypercubeActiveDims =
            getIntFlag(argc,
                       argv,
                       "--fbo-hypercube-active-dims",
                       cfg.hypercubeActiveDims);
        cfg.voronoiSites =
            getIntFlag(argc, argv, "--fbo-voronoi-sites", cfg.voronoiSites);
        cfg.simplexChildren =
            getIntFlag(argc,
                       argv,
                       "--fbo-simplex-children",
                       cfg.simplexChildren);

        if (hasFlagValue(argc, argv, "--doe-frac") ||
            hasFlagValue(argc, argv, "--explore-frac")) {
            cfg.budgetMode = FBOBudgetMode::FixedFractions;
        }

        if (cfg.budgetMode == FBOBudgetMode::FixedFractions) {
            if (cfg.doeFraction < 0.0) cfg.doeFraction = 0.25;
            if (cfg.exploreFraction < 0.0) cfg.exploreFraction = 0.65;
            if (cfg.exploitFraction < 0.0) cfg.exploitFraction = 0.10;
        }
        if (cfg.gpSigmaF <= 0.0) cfg.gpSigmaF = 1.0;
        if (cfg.gpLengthScale <= 0.0) cfg.gpLengthScale = 1.0;
        if (cfg.gpNoise < 0.0) cfg.gpNoise = 1e-3;
        if (cfg.gpRQAlpha <= 0.0) cfg.gpRQAlpha = 1.0;
        if (cfg.exploitStarts <= 0) cfg.exploitStarts = 1;
        if (cfg.exploitRadiusScale < 0.0) cfg.exploitRadiusScale = 0.0;
    } catch (const std::exception& e) {
        if (worldRank == 0) {
            std::cerr << "[FBOConfig] " << e.what() << "\n";
        }
        throw;
    }

    return cfg;
}

std::string toString(FBODecomposition v)
{
    switch (v) {
        case FBODecomposition::Hypersphere:
            return "hypersphere";
        case FBODecomposition::Hypercube:
            return "hypercube";
        case FBODecomposition::Bisection:
            return "bisection";
        case FBODecomposition::Trisection:
            return "trisection";
        case FBODecomposition::Voronoi:
            return "voronoi-style";
        case FBODecomposition::Simplex:
            return "simplex-lite";
    }
    return "unknown";
}

std::string toString(FBOSplitPolicy v)
{
    switch (v) {
        case FBOSplitPolicy::Cyclic:
            return "cyclic";
        case FBOSplitPolicy::LongestSide:
            return "longest-side";
        case FBOSplitPolicy::Random:
            return "random";
    }
    return "unknown";
}

std::string toString(FBODoe v)
{
    switch (v) {
        case FBODoe::Sobol:
            return "sobol";
        case FBODoe::DoeFile:
            return "doe-file";
        case FBODoe::Random:
            return "random";
        case FBODoe::LatinHypercube:
            return "latin-hypercube";
        case FBODoe::Halton:
            return "halton";
        case FBODoe::Hammersley:
            return "hammersley";
    }
    return "unknown";
}

std::string toString(FBODoeDistribution v)
{
    switch (v) {
        case FBODoeDistribution::Replicated:
            return "replicated";
        case FBODoeDistribution::Distributed:
            return "distributed";
    }
    return "unknown";
}

std::string toString(FBOSurrogate v)
{
    switch (v) {
        case FBOSurrogate::PcaGp:
            return "pca-gp";
        case FBOSurrogate::HsicGp:
            return "hsic-gp";
        case FBOSurrogate::InverseDistance:
            return "idw";
        case FBOSurrogate::RbfKernel:
            return "rbf-kernel";
        case FBOSurrogate::None:
            return "none";
    }
    return "unknown";
}

std::string toString(FBOGPKernel v)
{
    switch (v) {
        case FBOGPKernel::SquaredExponential:
            return "rbf";
        case FBOGPKernel::Exponential:
            return "exponential";
        case FBOGPKernel::Matern32:
            return "matern32";
        case FBOGPKernel::Matern52:
            return "matern52";
        case FBOGPKernel::RationalQuadratic:
            return "rational-quadratic";
    }
    return "unknown";
}

std::string toString(FBOExploitation v)
{
    switch (v) {
        case FBOExploitation::SurrogateLocalSearch:
            return "surrogate-local-search";
        case FBOExploitation::RandomLocalPerturbation:
            return "random-local-perturbation";
        case FBOExploitation::RandomRestartLocalSearch:
            return "random-restart-local";
        case FBOExploitation::BestCell:
            return "best-cell";
        case FBOExploitation::None:
            return "none";
    }
    return "unknown";
}

std::string toString(FBOBudgetMode v)
{
    switch (v) {
        case FBOBudgetMode::Astra:
            return "astra";
        case FBOBudgetMode::FixedFractions:
            return "fixed-fractions";
    }
    return "unknown";
}

void printFBOConfig(const FBOConfig& cfg, std::ostream& os)
{
    os << "\n[FBOConfig]\n";
    os << "  decomposition = " << toString(cfg.decomposition) << "\n";
    os << "  split policy  = " << toString(cfg.splitPolicy) << "\n";
    os << "  DOE           = " << toString(cfg.doe) << "\n";
    os << "  DOE execution = " << toString(cfg.doeDistribution) << "\n";
    os << "  surrogate     = " << toString(cfg.surrogate) << "\n";
    os << "  GP kernel     = " << toString(cfg.gpKernel) << "\n";
    os << "  exploitation  = " << toString(cfg.exploitation) << "\n";
    os << "  budget mode   = " << toString(cfg.budgetMode) << "\n";
    os << "  radius mode   = "
       << (cfg.adaptiveHypersphereRadius ? "adaptive" : "fixed")
       << "\n";
    os << "  exploitation guidance = "
       << (cfg.adaptiveExploitationScale ? "adaptive" : "fixed")
       << "\n";
    os << "  doe frac      = " << cfg.doeFraction << "\n";
    os << "  explore frac  = " << cfg.exploreFraction << "\n";
    os << "  exploit frac  = " << cfg.exploitFraction << "\n";
    os << "  surrogate length scale = "
       << cfg.surrogateLengthScale << "\n";
    os << "  IDW power              = " << cfg.idwPower << "\n";
    os << "  GP sigma_f            = " << cfg.gpSigmaF << "\n";
    os << "  GP length scale       = " << cfg.gpLengthScale << "\n";
    os << "  GP noise              = " << cfg.gpNoise << "\n";
    os << "  GP RQ alpha           = " << cfg.gpRQAlpha << "\n";
    os << "  exploit starts        = " << cfg.exploitStarts << "\n";
    os << "  exploit local iters   = " << cfg.exploitLocalIters << "\n";
    os << "  exploit radius scale  = " << cfg.exploitRadiusScale << "\n";
    os << "  hypercube active dims = " << cfg.hypercubeActiveDims << "\n";
    os << "  voronoi sites         = " << cfg.voronoiSites << "\n";
    os << "  simplex children      = " << cfg.simplexChildren << "\n\n";
}
