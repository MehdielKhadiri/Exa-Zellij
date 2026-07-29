#ifndef FBD_CONFIG_HPP
#define FBD_CONFIG_HPP

#include "FBOTypes.hpp"

#include <iosfwd>
#include <string>

enum class OptimizerFamily {
    FBO,
    FBD
};

enum class FBDExploration {
    Center,
    Random,
    CenterRandom,
    CenterAxial,
    Corners,
    LowDiscrepancy
};

enum class FBDScoring {
    BestValue,
    MeanValue,
    Improvement,
    ImprovementDistance,
    Optimistic
};

enum class FBDTreeSearch {
    Beam,
    BestFirst,
    BreadthFirst,
    DepthFirst,
    EpsilonGreedy,
    CyclicBest
};

struct FBDConfig {
    FBODecomposition decomposition = FBODecomposition::Hypersphere;
    FBOSplitPolicy splitPolicy = FBOSplitPolicy::Cyclic;
    FBDExploration exploration = FBDExploration::CenterAxial;
    FBDScoring scoring = FBDScoring::ImprovementDistance;
    FBDTreeSearch treeSearch = FBDTreeSearch::Beam;
    FBOExploitation exploitation = FBOExploitation::BestCell;
    FBODoe initialDesign = FBODoe::Random;

    int pointsPerCell = -1; // -1 means exploration-dependent default.
    int initialPoints = 1;
    int beamWidth = -1;
    double epsilon = 0.10;

    int hypercubeActiveDims = 12;
    int voronoiSites = 0;
    int simplexChildren = 0;
};

OptimizerFamily parseOptimizerFamily(int argc, char** argv, int worldRank);
FBDConfig parseFBDConfig(int argc, char** argv, int worldRank);

std::string toString(OptimizerFamily v);
std::string toString(FBDExploration v);
std::string toString(FBDScoring v);
std::string toString(FBDTreeSearch v);
int defaultFBDPointsPerCell(FBDExploration v);

void printFBDConfig(const FBDConfig& cfg, std::ostream& os);

#endif // FBD_CONFIG_HPP
