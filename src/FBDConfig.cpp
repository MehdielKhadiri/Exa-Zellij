#include "FBDConfig.hpp"

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
    if (s == "hypercube" || s == "cube") return FBODecomposition::Hypercube;
    if (s == "bisection" || s == "bisect" || s == "axis-bisection") {
        return FBODecomposition::Bisection;
    }
    if (s == "trisection" || s == "tri") return FBODecomposition::Trisection;
    if (s == "voronoi" || s == "voronoi-style" || s == "star-voronoi") {
        return FBODecomposition::Voronoi;
    }
    if (s == "simplex" || s == "simplex-lite") return FBODecomposition::Simplex;
    throw std::invalid_argument("unknown --fbd-decomposition value: " + s);
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
    throw std::invalid_argument("unknown --fbd-split-policy value: " + s);
}

FBODoe parseInitialDesign(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "random" || s == "uniform-random") return FBODoe::Random;
    if (s == "lhs" || s == "latin" || s == "latin-hypercube") {
        return FBODoe::LatinHypercube;
    }
    if (s == "halton" || s == "low-discrepancy") return FBODoe::Halton;
    if (s == "hammersley") return FBODoe::Hammersley;
    throw std::invalid_argument("unknown --fbd-initial-design value: " + s);
}

FBOExploitation parseExploitation(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "best-cell" || s == "center" || s == "cell-center") {
        return FBOExploitation::BestCell;
    }
    if (s == "random-local" || s == "random-perturbation") {
        return FBOExploitation::RandomLocalPerturbation;
    }
    if (s == "random-restart" || s == "random-restart-local" ||
        s == "multi-start" || s == "multistart") {
        return FBOExploitation::RandomRestartLocalSearch;
    }
    if (s == "none" || s == "no" || s == "no-exploitation") {
        return FBOExploitation::None;
    }
    throw std::invalid_argument("unknown --fbd-exploitation value: " + s);
}

FBDExploration parseExploration(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "center" || s == "centre") return FBDExploration::Center;
    if (s == "random" || s == "random-in-cell") return FBDExploration::Random;
    if (s == "center-random" || s == "centre-random") {
        return FBDExploration::CenterRandom;
    }
    if (s == "center-axial" || s == "centre-axial" || s == "axial" ||
        s == "promising-hypersphere" || s == "promishing-hypersphere") {
        return FBDExploration::CenterAxial;
    }
    if (s == "corners" || s == "corner") return FBDExploration::Corners;
    if (s == "low-discrepancy" || s == "halton" || s == "hammersley") {
        return FBDExploration::LowDiscrepancy;
    }
    throw std::invalid_argument("unknown --fbd-exploration value: " + s);
}

FBDScoring parseScoring(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "best" || s == "best-value" || s == "min") {
        return FBDScoring::BestValue;
    }
    if (s == "mean" || s == "mean-value") return FBDScoring::MeanValue;
    if (s == "improvement") return FBDScoring::Improvement;
    if (s == "improvement-distance" || s == "old-fbd" ||
        s == "legacy") {
        return FBDScoring::ImprovementDistance;
    }
    if (s == "optimistic" || s == "optimism" || s == "lower-bound") {
        return FBDScoring::Optimistic;
    }
    throw std::invalid_argument("unknown --fbd-scoring value: " + s);
}

FBDTreeSearch parseTreeSearch(std::string raw)
{
    const std::string s = lowerCopy(std::move(raw));
    if (s == "beam" || s == "beam-search") return FBDTreeSearch::Beam;
    if (s == "best-first" || s == "best") return FBDTreeSearch::BestFirst;
    if (s == "breadth-first" || s == "bfs") return FBDTreeSearch::BreadthFirst;
    if (s == "depth-first" || s == "dfs") return FBDTreeSearch::DepthFirst;
    if (s == "epsilon-greedy" || s == "eps-greedy") {
        return FBDTreeSearch::EpsilonGreedy;
    }
    if (s == "cyclic-best" || s == "cyclic-best-first") {
        return FBDTreeSearch::CyclicBest;
    }
    throw std::invalid_argument("unknown --fbd-tree-search value: " + s);
}

} // namespace

OptimizerFamily parseOptimizerFamily(int argc, char** argv, int worldRank)
{
    const std::string raw =
        getStringFlag(argc, argv, "--optimizer-family", "fbo");
    const std::string s = lowerCopy(raw);

    if (s == "fbo" || s == "bayesian" || s == "fractal-bayesian") {
        return OptimizerFamily::FBO;
    }
    if (s == "fbd" || s == "decomposition" ||
        s == "fractal-decomposition") {
        return OptimizerFamily::FBD;
    }

    if (worldRank == 0) {
        std::cerr << "[OptimizerFamily] unknown --optimizer-family value: "
                  << raw << "\n";
    }
    throw std::invalid_argument("unknown optimizer family");
}

FBDConfig parseFBDConfig(int argc, char** argv, int worldRank)
{
    FBDConfig cfg;

    try {
        cfg.decomposition = parseDecomposition(
            getStringFlag(argc, argv, "--fbd-decomposition", "hypersphere"));
        cfg.splitPolicy = parseSplitPolicy(
            getStringFlag(argc, argv, "--fbd-split-policy", "cyclic"));
        cfg.exploration = parseExploration(
            getStringFlag(argc, argv, "--fbd-exploration", "center-axial"));
        cfg.scoring = parseScoring(
            getStringFlag(argc, argv, "--fbd-scoring", "improvement-distance"));
        cfg.treeSearch = parseTreeSearch(
            getStringFlag(argc, argv, "--fbd-tree-search", "beam"));
        cfg.exploitation = parseExploitation(
            getStringFlag(argc, argv, "--fbd-exploitation", "best-cell"));
        cfg.initialDesign = parseInitialDesign(
            getStringFlag(argc, argv, "--fbd-initial-design", "random"));

        cfg.pointsPerCell =
            getIntFlag(argc, argv, "--fbd-points-per-cell", cfg.pointsPerCell);
        cfg.initialPoints =
            getIntFlag(argc, argv, "--fbd-initial-points", cfg.initialPoints);
        cfg.beamWidth =
            getIntFlag(argc, argv, "--fbd-beam-width", cfg.beamWidth);
        cfg.epsilon =
            getDoubleFlag(argc, argv, "--fbd-epsilon", cfg.epsilon);
        cfg.hypercubeActiveDims =
            getIntFlag(argc,
                       argv,
                       "--fbd-hypercube-active-dims",
                       cfg.hypercubeActiveDims);
        cfg.voronoiSites =
            getIntFlag(argc, argv, "--fbd-voronoi-sites", cfg.voronoiSites);
        cfg.simplexChildren =
            getIntFlag(argc,
                       argv,
                       "--fbd-simplex-children",
                       cfg.simplexChildren);

        if (cfg.pointsPerCell == 0) cfg.pointsPerCell = -1;
        if (cfg.initialPoints <= 0) cfg.initialPoints = 1;
        if (cfg.epsilon < 0.0) cfg.epsilon = 0.0;
        if (cfg.epsilon > 1.0) cfg.epsilon = 1.0;
    } catch (const std::exception& e) {
        if (worldRank == 0) {
            std::cerr << "[FBDConfig] " << e.what() << "\n";
        }
        throw;
    }

    return cfg;
}

std::string toString(OptimizerFamily v)
{
    switch (v) {
        case OptimizerFamily::FBO:
            return "fbo";
        case OptimizerFamily::FBD:
            return "fbd";
    }
    return "unknown";
}

std::string toString(FBDExploration v)
{
    switch (v) {
        case FBDExploration::Center:
            return "center";
        case FBDExploration::Random:
            return "random";
        case FBDExploration::CenterRandom:
            return "center-random";
        case FBDExploration::CenterAxial:
            return "center-axial";
        case FBDExploration::Corners:
            return "corners";
        case FBDExploration::LowDiscrepancy:
            return "low-discrepancy";
    }
    return "unknown";
}

std::string toString(FBDScoring v)
{
    switch (v) {
        case FBDScoring::BestValue:
            return "best-value";
        case FBDScoring::MeanValue:
            return "mean-value";
        case FBDScoring::Improvement:
            return "improvement";
        case FBDScoring::ImprovementDistance:
            return "improvement-distance";
        case FBDScoring::Optimistic:
            return "optimistic";
    }
    return "unknown";
}

std::string toString(FBDTreeSearch v)
{
    switch (v) {
        case FBDTreeSearch::Beam:
            return "beam";
        case FBDTreeSearch::BestFirst:
            return "best-first";
        case FBDTreeSearch::BreadthFirst:
            return "breadth-first";
        case FBDTreeSearch::DepthFirst:
            return "depth-first";
        case FBDTreeSearch::EpsilonGreedy:
            return "epsilon-greedy";
        case FBDTreeSearch::CyclicBest:
            return "cyclic-best";
    }
    return "unknown";
}

int defaultFBDPointsPerCell(FBDExploration v)
{
    switch (v) {
        case FBDExploration::Center:
            return 1;
        case FBDExploration::CenterRandom:
            return 2;
        case FBDExploration::CenterAxial:
            return 3;
        case FBDExploration::Random:
        case FBDExploration::Corners:
        case FBDExploration::LowDiscrepancy:
        default:
            return 1;
    }
}

void printFBDConfig(const FBDConfig& cfg, std::ostream& os)
{
    os << "\n[FBDConfig]\n";
    os << "  decomposition = " << toString(cfg.decomposition) << "\n";
    os << "  split policy  = " << toString(cfg.splitPolicy) << "\n";
    os << "  exploration   = " << toString(cfg.exploration) << "\n";
    os << "  scoring       = " << toString(cfg.scoring) << "\n";
    os << "  tree search   = " << toString(cfg.treeSearch) << "\n";
    os << "  exploitation  = " << toString(cfg.exploitation) << "\n";
    os << "  initial design = " << toString(cfg.initialDesign) << "\n";
    os << "  points/cell   = " << cfg.pointsPerCell << "\n";
    os << "  initial points = " << cfg.initialPoints << "\n";
    os << "  beam width    = " << cfg.beamWidth << "\n";
    os << "  epsilon       = " << cfg.epsilon << "\n";
    os << "  hypercube active dims = " << cfg.hypercubeActiveDims << "\n";
    os << "  voronoi sites         = " << cfg.voronoiSites << "\n";
    os << "  simplex children      = " << cfg.simplexChildren << "\n\n";
}
