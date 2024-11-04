#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>
#include <numeric>

// Define the execution and memory spaces
using exec_space = Kokkos::DefaultExecutionSpace;
using mem_space = typename exec_space::memory_space;

// Hypersphere identifier: a vector of pairs (dimension index, sign)
using HypersphereID = std::vector<std::pair<int, int>>;

// Hypersphere structure definition
struct Hypersphere {
    HypersphereID id; // Unique identifier for the hypersphere
    int dimension;    // Dimension of the space
    int depth;        // Depth in the decomposition tree

    Hypersphere(int dim, int d, const HypersphereID& identifier)
        : dimension(dim), depth(d), id(identifier) {}
};

// SearchSpaceBounds structure definition
struct SearchSpaceBounds {
    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> lowerBounds;
    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> upperBounds;
    int dimension;

    // Constructor
    SearchSpaceBounds(int n, const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& lBounds,
                      const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& uBounds)
        : lowerBounds(lBounds), upperBounds(uBounds), dimension(n) {}
};

// Compute a single term of the Shubert function
KOKKOS_INLINE_FUNCTION
double computeShubertTerm(double x, int i) {
    return i * sin((i + 1) * x + i);
}

// Compute the objective function using a Kokkos::View
template<typename ViewType>
KOKKOS_INLINE_FUNCTION
double objectiveFunction(const ViewType& pointView) {
    double result = 1.0;
    const int dimension = pointView.extent(0);

    for (int d = 0; d < dimension; ++d) {
        double x = pointView(d);
        double term = 0.0;
        for (int i = 1; i <= 5; ++i) {
            term += computeShubertTerm(x, i);
        }
        result *= term;
    }

    return result;
}












// Helper function for Sphere
KOKKOS_INLINE_FUNCTION
double computeSphereTerm(double x) {
    return x * x;
}

// Objective function for Sphere
template<typename ViewType>
KOKKOS_INLINE_FUNCTION
double sphereObjectiveFunction(const ViewType& pointView) {
    double result = 0.0;
    const int dimension = pointView.extent(0);

    for(int d = 0; d < dimension; ++d){
        double x = pointView(d);
        result += computeSphereTerm(x);
    }

    return result;
}

// ------------------------------
// 2. Rastrigin Function (Multimodal)
// ------------------------------

// Helper function for Rastrigin
KOKKOS_INLINE_FUNCTION
double computeRastriginTerm(double x, int /*i*/) {
    return x * x - 10.0 * cos(2.0 * M_PI * x);
}

// Objective function for Rastrigin
template<typename ViewType>
KOKKOS_INLINE_FUNCTION
double rastriginObjectiveFunction(const ViewType& pointView) {
    double result = 10.0 * pointView.extent(0); // 10 * dimension
    const int dimension = pointView.extent(0);

    for(int d = 0; d < dimension; ++d){
        double x = pointView(d);
        result += computeRastriginTerm(x, d);
    }

    return result;
}

// ------------------------------
// 3. Schwefel Function (Multimodal with Many Local Minima)
// ------------------------------

// Helper function for Schwefel
KOKKOS_INLINE_FUNCTION
double computeSchwefelTerm(double x, int /*i*/) {
    return -x * sin(sqrt(std::abs(x)));
}

// Objective function for Schwefel
template<typename ViewType>
KOKKOS_INLINE_FUNCTION
double schwefelObjectiveFunction(const ViewType& pointView) {
    double result = 0.0;
    const int dimension = pointView.extent(0);

    for(int d = 0; d < dimension; ++d){
        double x = pointView(d);
        result += computeSchwefelTerm(x, d);
    }

    return result;
}

// ------------------------------
// 4. Ellipsoidal Function (Separable, Scalable)
// ------------------------------

// Helper function for Ellipsoidal
KOKKOS_INLINE_FUNCTION
double computeEllipsoidalTerm(double x, int i) {
    return std::pow(10.0, static_cast<double>(i)) * x * x;
}

// Objective function for Ellipsoidal
template<typename ViewType>
KOKKOS_INLINE_FUNCTION
double ellipsoidalObjectiveFunction(const ViewType& pointView) {
    double result = 0.0;
    const int dimension = pointView.extent(0);

    for(int d = 0; d < dimension; ++d){
        double x = pointView(d);
        result += computeEllipsoidalTerm(x, d + 1); // i starts from 1
    }

    return result;
}


















// Structure to hold a hypersphere and its score
struct HypersphereScore {
    Hypersphere hs;
    double score;

    HypersphereScore() = default;

    HypersphereScore(const Hypersphere& h, double sc)
        : hs(h), score(sc) {}
};





// Serialize HypersphereID into a vector of integers
std::vector<int> serializeHypersphereID(const HypersphereID& id) {
    std::vector<int> serialized;
    serialized.push_back(id.size()); // First, store the size
    for (const auto& pair : id) {
        serialized.push_back(pair.first);  // dimension index
        serialized.push_back(pair.second); // sign
    }
    return serialized;
}






// Deserialize vector of integers into HypersphereID
HypersphereID deserializeHypersphereID(const std::vector<int>& data, size_t& offset) {
    HypersphereID id;
    size_t idSize = data[offset++];
    for (size_t i = 0; i < idSize; ++i) {
        int dim = data[offset++];
        int sign = data[offset++];
        id.emplace_back(dim, sign);
    }
    return id;
}









// Generate random points around the center of a hypersphere
Kokkos::View<double**, Kokkos::LayoutLeft, mem_space>
generateRandomPointsAroundCenter(const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& center, double radius, int dimension) {
    const int numPoints = 3; // Number of points to generate
    Kokkos::View<double**, Kokkos::LayoutLeft, mem_space> points("points", numPoints, dimension);

    // Define the team policy (hierarchical parallelism)
    auto team_policy = Kokkos::TeamPolicy<>(numPoints, Kokkos::AUTO);

    // Generate random values for each point and dimension in parallel
    Kokkos::parallel_for("GenerateRandomPoints", team_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const int i = team.league_rank();  // Each team works on one point

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dimension), [&](const int dim) {
            double val = 0.5; // For simplicity, using a fixed value
            points(i, dim) = val;  // Store the random value in the points view
        });

        team.team_barrier();  // Ensure all threads in the team are synchronized
    });

    // Compute the norm for each point and normalize the points
    Kokkos::parallel_for("NormalizePoints", team_policy, KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        const int i = team.league_rank();  // Each team works on one point
        double norm = 0.0;

        // Parallel reduction to compute the norm across dimensions
        Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, dimension), [&](const int dim, double& localNorm) {
            localNorm += points(i, dim) * points(i, dim);  // Sum of squares
        }, norm);

        norm = sqrt(norm);  // Final norm

        // Normalize the points in parallel across dimensions
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dimension), [&](const int dim) {
            points(i, dim) = center(dim) + (points(i, dim) / norm) * radius;  // Normalize and scale
        });

        team.team_barrier();  // Ensure all threads in the team are synchronized
    });

    return points;
}
























// Score a hypersphere using the Sphere objective by computing the average objective value of sampled points
double scoreHypersphere(const Hypersphere& hs,
                               const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initialCenter,
                               double initialRadius) {
    // Reconstruct center in host memory
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> centerHost("centerHost", hs.dimension);

    // Copy initialCenter to centerHost
    auto initialCenterHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), initialCenter);
    Kokkos::deep_copy(centerHost, initialCenterHost);

    double r = initialRadius;
    for (int d = 0; d < hs.depth; ++d) {
        double sqrt2 = sqrt(2.0);
        double r_prime = r / (1.0 + sqrt2);
        double offset = r - r_prime;
        int dim = hs.id[d].first;
        int sign = hs.id[d].second;

        centerHost(dim) += sign * offset;
        r = r_prime; // Update radius for next level
    }

    // Now, create device copy of centerHost
    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> center("center", hs.dimension);
    Kokkos::deep_copy(center, centerHost);

    // Generate points around the center
    auto points = generateRandomPointsAroundCenter(center, r, hs.dimension);

    double totalScore = 0.0;

    auto team_policy = Kokkos::TeamPolicy<>(points.extent(0), Kokkos::AUTO); // Number of points, AUTO threads per team

    Kokkos::parallel_reduce(
        "ComputeScoreSphere",
        team_policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team, double& teamSum) {
            int i = team.league_rank(); // Each team works on one point

            // Compute the Sphere objective value
            double objValue = sphereObjectiveFunction(Kokkos::subview(points, i, Kokkos::ALL()));

            // Team-wide reduction
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                teamSum += objValue;
            });
        }, totalScore);

    // Compute the arithmetic mean
    return totalScore / static_cast<double>(points.extent(0));
}
















// Compute the objective value of the solution using parallel_reduce
double computeObjectiveValue(const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& solution) {
    double objValue = 0.0;

    // Using a lambda that accesses the entire view
    Kokkos::parallel_reduce("ComputeObjectiveValue", 1,
        KOKKOS_LAMBDA(const int, double& lsum) {
            lsum = sphereObjectiveFunction(Kokkos::subview(solution, Kokkos::ALL()));
        }, objValue);

    return objValue;
}


















Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
intensiveLocalSearch(const Hypersphere& hs,
                            const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initialCenter,
                            double initialRadius,
                            int maxIterations, double stepSize,
                            double phi, double omega_min) {

    // Reconstruct center in host memory
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> centerHost("centerHost", hs.dimension);

    // Copy initialCenter to centerHost
    auto initialCenterHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), initialCenter);
    Kokkos::deep_copy(centerHost, initialCenterHost);

    double r = initialRadius;
    for (int d = 0; d < hs.depth; ++d) {
        double sqrt2 = sqrt(2.0);
        double r_prime = r / (1.0 + sqrt2);
        double offset = r - r_prime;
        int dim = hs.id[d].first;
        int sign = hs.id[d].second;

        centerHost(dim) += sign * offset;
        r = r_prime; // Update radius for next level
    }

    // Now, create device copy of centerHost
    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> center("center", hs.dimension);
    Kokkos::deep_copy(center, centerHost);

    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> bestSolution("bestSolution", hs.dimension);
    Kokkos::deep_copy(bestSolution, center);

    // Policy without scratch memory
    auto policy = Kokkos::TeamPolicy<exec_space>(1, Kokkos::AUTO);

    Kokkos::View<double*, mem_space> localBestPoint("localBestPoint", hs.dimension);
    Kokkos::View<double*, mem_space> newValues("newValues", hs.dimension);
    Kokkos::View<double*, mem_space> term_d("term_d", hs.dimension);  // Store the terms in global memory
    Kokkos::View<bool*, mem_space> improvementMade("improvementMade", 1); // Global memory for improvement flag

    Kokkos::parallel_for("IntensiveLocalSearchSphere", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<exec_space>::member_type& team) {

            // Initialize localBestPoint
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension), [=](int dim) {
                localBestPoint(dim) = bestSolution(dim);
            });

            team.team_barrier();

            // Compute initial objective terms
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension), [=](int dim) {
                double x = localBestPoint(dim);
                double term = computeSphereTerm(x);
                term_d(dim) = term;
            });

            team.team_barrier();

            // Compute initial bestValue
            double bestValue = 0.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, hs.dimension),
                [=](int dim, double& sum) {
                    sum += term_d(dim);
                }, bestValue);

            team.team_barrier();

            double omega = stepSize;  // Initialize omega

            for (int iter = 0; iter < maxIterations; ++iter) {
                if (team.team_rank() == 0) {
                    improvementMade(0) = false;
                }
                team.team_barrier();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension), [=](int dim) {
                    double xs1 = localBestPoint(dim) + omega;
                    double xs2 = localBestPoint(dim) - omega;

                    double neighborValue1 = computeSphereTerm(xs1);
                    double neighborValue2 = computeSphereTerm(xs2);

                    double localBestValue = bestValue;
                    double newTerm = term_d(dim);
                    double new_x = localBestPoint(dim);

                    if (neighborValue1 < localBestValue) {
                        localBestValue = neighborValue1;
                        new_x = xs1;
                        newTerm = neighborValue1;
                        improvementMade(0) = true;
                    }
                    if (neighborValue2 < localBestValue) {
                        localBestValue = neighborValue2;
                        new_x = xs2;
                        newTerm = neighborValue2;
                        improvementMade(0) = true;
                    }

                    newValues(dim) = new_x;
                    term_d(dim) = newTerm;
                });

                team.team_barrier();

                // Update localBestPoint with new values
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension), [=](int dim) {
                    localBestPoint(dim) = newValues(dim);
                });

                team.team_barrier();

                // Recompute bestValue
                double newBestValue = 0.0;
                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, hs.dimension),
                    [=](int dim, double& sum) {
                        sum += term_d(dim);
                    }, newBestValue);

                bestValue = newBestValue;

                team.team_barrier();

                // Adjust omega if no improvement
                if (team.team_rank() == 0) {
                    if (!improvementMade(0)) {
                        omega /= phi;
                        if (omega < omega_min) {
                            iter = maxIterations;
                        }
                    }
                }
                team.team_barrier();
            }

            // Update bestSolution with localBestPoint
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension), [=](int dim) {
                bestSolution(dim) = localBestPoint(dim);
            });
        });

    return bestSolution;
}






















// Decompose a hypersphere into child hyperspheres
std::vector<Hypersphere> decomposeHypersphere(const Hypersphere& parentHs) {
    std::vector<Hypersphere> childHyperspheres;
    int D = parentHs.dimension;
    int newDepth = parentHs.depth + 1;

    for (int k = 0; k < D; ++k) {
        for (int sign = -1; sign <= 1; sign += 2) {
            HypersphereID childID = parentHs.id;
            childID.emplace_back(k, sign);
            Hypersphere childHs(D, newDepth, childID);
            childHyperspheres.push_back(childHs);
        }
    }

    return childHyperspheres;
}













// Function to select the indices of the best 'beta' scores
std::vector<int> selectBestIndices(const std::vector<double>& scores, int beta) {
    std::vector<int> indices(scores.size());
    std::iota(indices.begin(), indices.end(), 0);

    beta = std::min(beta, static_cast<int>(scores.size()));

    std::partial_sort(indices.begin(), indices.begin() + beta, indices.end(),
                      [&scores](int a, int b) { return scores[a] < scores[b]; });

    indices.resize(beta);
    return indices;
}

























int main(int argc, char* argv[]) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Create a communicator for processes on the same node
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, world_rank, MPI_INFO_NULL, &node_comm);

    int node_rank, node_size;
    MPI_Comm_rank(node_comm, &node_rank);
    MPI_Comm_size(node_comm, &node_size);

    // Identify node leaders (processes with rank 0 in node_comm)
    int is_node_leader = (node_rank == 0) ? 1 : 0;

    // Create a communicator for node leaders
    MPI_Comm node_leader_comm;
    if (node_rank == 0) {
        MPI_Comm_split(MPI_COMM_WORLD, 0, world_rank, &node_leader_comm);
    } else {
        MPI_Comm_split(MPI_COMM_WORLD, MPI_UNDEFINED, world_rank, &node_leader_comm);
    }

    int num_nodes = 0;
    int node_id = -1;
    if (node_rank == 0) {
        MPI_Comm_size(node_leader_comm, &num_nodes);
        MPI_Comm_rank(node_leader_comm, &node_id);
    }

    // Broadcast num_nodes and node_id to all processes in node_comm
    MPI_Bcast(&num_nodes, 1, MPI_INT, 0, node_comm);
    MPI_Bcast(&node_id, 1, MPI_INT, 0, node_comm);

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Define parameters for the search
        int dimension = 5000; // Changed to non-const to avoid issues with MPI_Bcast if needed
        const int beta = 5;
        const int maxDepth = 5;
        // Parameters for applying the intensive local search starting from the center
        int maxIterations = 100;   // Number of iterations for the search
        double stepSize = 2.0;     // Initial step size (ω)
        double phi = 1.2;          // Factor to reduce step size
        double omega_min = 0.001;  // Minimum step size before stopping

        // Initialize the search space
        Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
            lowerBounds("lowerBounds", dimension);
        Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
            upperBounds("upperBounds", dimension);
        Kokkos::deep_copy(lowerBounds, -10.0); // Example bounds
        Kokkos::deep_copy(upperBounds, 10.0);
        SearchSpaceBounds searchSpace(dimension, lowerBounds, upperBounds);

        // Seed for random number generation (unique per rank)
        uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() + world_rank;
        Kokkos::Random_XorShift64_Pool<exec_space> randomPool(seed);

        // Compute initial hypersphere center
        Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
            initialCenter("initialCenter", dimension);
        Kokkos::parallel_for("ComputeInitialCenter", Kokkos::RangePolicy<exec_space>(0, dimension),
            KOKKOS_LAMBDA(const int j) {
                initialCenter(j) = -10.0 + (10.0 - (-10.0)) / 2.0; // Center at 0.0
            });

        // Compute radius: r = (U_j - L_j) / 2
        double initialRadius = 10.0; // Since upperBounds - lowerBounds = 20, r = 10

        // Create the initial hypersphere
        HypersphereID initialID;
        Hypersphere initialHypersphere(dimension, 0, initialID);

        // Decompose the initial hypersphere into child hyperspheres
        std::vector<Hypersphere> currentHyperspheres = decomposeHypersphere(initialHypersphere);

        if (world_rank == 0) {
            std::cout << "Rank " << world_rank << " is entering the main loop." << std::endl;
        }

        // Main iterative process
        for (int depth = 1; depth <= maxDepth; ++depth) {
            // Distribute currentHyperspheres among nodes
            int numHyperspheres = currentHyperspheres.size();
            int numPerNode = numHyperspheres / num_nodes;
            int remainder = numHyperspheres % num_nodes;

            // Compute the range of hyperspheres assigned to this node
            int startIdx = node_id * numPerNode + std::min(node_id, remainder);
            int endIdx = startIdx + numPerNode + (node_id < remainder ? 1 : 0);
            int localNumHyperspheres = endIdx - startIdx;

            // Within the node, distribute hyperspheres among local processes
            int numPerRank = localNumHyperspheres / node_size;
            int remainder_local = localNumHyperspheres % node_size;

            int localStartIdx = startIdx + node_rank * numPerRank + std::min(node_rank, remainder_local);
            int localEndIdx = localStartIdx + numPerRank + (node_rank < remainder_local ? 1 : 0);
            int numLocalHyperspheres = localEndIdx - localStartIdx;

            // Initialize a vector to store local scores and IDs
            std::vector<double> localScoresOnly(numLocalHyperspheres);
            std::vector<int> localIDsSerialized;

            // Process assigned hyperspheres
            for (int i = 0; i < numLocalHyperspheres; ++i) {
                int idx = localStartIdx + i;
                double score = scoreHypersphere(currentHyperspheres[idx], initialCenter, initialRadius);
                localScoresOnly[i] = score;

                auto serializedID = serializeHypersphereID(currentHyperspheres[idx].id);
                localIDsSerialized.insert(localIDsSerialized.end(), serializedID.begin(), serializedID.end());
            }

            // Intra-node communication to gather scores and IDs
            // Communicate sizes within the node
            int localIDSize = localIDsSerialized.size();
            std::vector<int> allNumHyperspheres(node_size);
            std::vector<int> allIDSizes(node_size);
            MPI_Allgather(&numLocalHyperspheres, 1, MPI_INT,
                          allNumHyperspheres.data(), 1, MPI_INT, node_comm);
            MPI_Allgather(&localIDSize, 1, MPI_INT,
                          allIDSizes.data(), 1, MPI_INT, node_comm);

            // Compute displacements
            int totalHyperspheres = std::accumulate(allNumHyperspheres.begin(), allNumHyperspheres.end(), 0);
            int totalIDSize = std::accumulate(allIDSizes.begin(), allIDSizes.end(), 0);

            std::vector<int> displsScores(node_size, 0);
            std::vector<int> displsIDs(node_size, 0);
            for (int i = 1; i < node_size; ++i) {
                displsScores[i] = displsScores[i - 1] + allNumHyperspheres[i - 1];
                displsIDs[i] = displsIDs[i - 1] + allIDSizes[i - 1];
            }

            // Gather all scores and identifiers within the node
            std::vector<double> allScores(totalHyperspheres);
            MPI_Allgatherv(localScoresOnly.data(), numLocalHyperspheres, MPI_DOUBLE,
                           allScores.data(), allNumHyperspheres.data(), displsScores.data(), MPI_DOUBLE, node_comm);

            std::vector<int> allIDsSerialized(totalIDSize);
            MPI_Allgatherv(localIDsSerialized.data(), localIDSize, MPI_INT,
                           allIDsSerialized.data(), allIDSizes.data(), displsIDs.data(), MPI_INT, node_comm);

            // Reconstruct all IDs
            std::vector<HypersphereID> allIDs(totalHyperspheres);
            size_t offset = 0;
            for (int i = 0; i < totalHyperspheres; ++i) {
                allIDs[i] = deserializeHypersphereID(allIDsSerialized, offset);
            }

            // Select best hyperspheres within the node
            std::vector<int> selectedIndices = selectBestIndices(allScores, beta);

            std::vector<Hypersphere> selectedHyperspheres;
            for (int idx : selectedIndices) {
                HypersphereID selectedID = allIDs[idx];
                Hypersphere hs(dimension, depth, selectedID);
                selectedHyperspheres.push_back(hs);
            }

            // Node leaders exchange best hyperspheres across nodes
            if (node_rank == 0) {
                // Prepare data for inter-node communication
                int numSelected = selectedHyperspheres.size();
                std::vector<double> nodeLocalScores(numSelected);
                std::vector<int> nodeLocalIDsSerialized;

                for (int i = 0; i < numSelected; ++i) {
                    nodeLocalScores[i] = allScores[selectedIndices[i]];
                    auto serializedID = serializeHypersphereID(selectedHyperspheres[i].id);
                    nodeLocalIDsSerialized.insert(nodeLocalIDsSerialized.end(), serializedID.begin(), serializedID.end());
                }

                // Communicate sizes among node leaders
                int nodeLocalIDSize = nodeLocalIDsSerialized.size();
                std::vector<int> nodeAllNumHyperspheres(num_nodes);
                std::vector<int> nodeAllIDSizes(num_nodes);
                MPI_Allgather(&numSelected, 1, MPI_INT,
                              nodeAllNumHyperspheres.data(), 1, MPI_INT, node_leader_comm);
                MPI_Allgather(&nodeLocalIDSize, 1, MPI_INT,
                              nodeAllIDSizes.data(), 1, MPI_INT, node_leader_comm);

                // Compute displacements
                int nodeTotalHyperspheres = std::accumulate(nodeAllNumHyperspheres.begin(), nodeAllNumHyperspheres.end(), 0);
                int nodeTotalIDSize = std::accumulate(nodeAllIDSizes.begin(), nodeAllIDSizes.end(), 0);

                std::vector<int> nodeDisplsScores(num_nodes, 0);
                std::vector<int> nodeDisplsIDs(num_nodes, 0);
                for (int i = 1; i < num_nodes; ++i) {
                    nodeDisplsScores[i] = nodeDisplsScores[i - 1] + nodeAllNumHyperspheres[i - 1];
                    nodeDisplsIDs[i] = nodeDisplsIDs[i - 1] + nodeAllIDSizes[i - 1];
                }

                // Gather all scores and identifiers among node leaders
                std::vector<double> nodeAllScores(nodeTotalHyperspheres);
                MPI_Allgatherv(nodeLocalScores.data(), numSelected, MPI_DOUBLE,
                               nodeAllScores.data(), nodeAllNumHyperspheres.data(), nodeDisplsScores.data(), MPI_DOUBLE, node_leader_comm);

                std::vector<int> nodeAllIDsSerialized(nodeTotalIDSize);
                MPI_Allgatherv(nodeLocalIDsSerialized.data(), nodeLocalIDSize, MPI_INT,
                               nodeAllIDsSerialized.data(), nodeAllIDSizes.data(), nodeDisplsIDs.data(), MPI_INT, node_leader_comm);

                // Reconstruct all IDs and select best hyperspheres across nodes
                std::vector<HypersphereID> nodeAllIDs(nodeTotalHyperspheres);
                offset = 0;
                for (int i = 0; i < nodeTotalHyperspheres; ++i) {
                    nodeAllIDs[i] = deserializeHypersphereID(nodeAllIDsSerialized, offset);
                }

                selectedIndices = selectBestIndices(nodeAllScores, beta);
                selectedHyperspheres.clear();
                for (int idx : selectedIndices) {
                    HypersphereID selectedID = nodeAllIDs[idx];
                    Hypersphere hs(dimension, depth, selectedID);
                    selectedHyperspheres.push_back(hs);
                }
            }

            // Broadcast the number of selected hyperspheres
            int numSelected;
            if (node_rank == 0) {
                numSelected = selectedHyperspheres.size();
            }
            MPI_Bcast(&numSelected, 1, MPI_INT, 0, node_comm);

            // Broadcast serialized identifiers
            std::vector<int> selectedIDsSerialized;
            if (node_rank == 0) {
                for (const auto& hs : selectedHyperspheres) {
                    auto serializedID = serializeHypersphereID(hs.id);
                    selectedIDsSerialized.insert(selectedIDsSerialized.end(), serializedID.begin(), serializedID.end());
                }
            }
            int selectedTotalIDSize = selectedIDsSerialized.size();
            MPI_Bcast(&selectedTotalIDSize, 1, MPI_INT, 0, node_comm);

            if (node_rank != 0) {
                selectedIDsSerialized.resize(selectedTotalIDSize);
            }
            MPI_Bcast(selectedIDsSerialized.data(), selectedTotalIDSize, MPI_INT, 0, node_comm);

            // Reconstruct selected hyperspheres
            selectedHyperspheres.clear();
            offset = 0;
            for (int i = 0; i < numSelected; ++i) {
                HypersphereID selectedID = deserializeHypersphereID(selectedIDsSerialized, offset);
                Hypersphere hs(dimension, depth, selectedID);
                selectedHyperspheres.push_back(hs);
            }


            // Update currentHyperspheres for the next depth
            currentHyperspheres = std::move(selectedHyperspheres);

            // Check for maximum depth
            if (depth == maxDepth) {
                // Exploitation phase

                // Distribute selected hyperspheres among nodes
                int numSelected = currentHyperspheres.size();
                int numPerNode = numSelected / num_nodes;
                int remainder = numSelected % num_nodes;

                int startIdx = node_id * numPerNode + std::min(node_id, remainder);
                int endIdx = startIdx + numPerNode + (node_id < remainder ? 1 : 0);
                int localNumHyperspheres = endIdx - startIdx;

                // Within the node, distribute hyperspheres among local processes
                int numPerRank = localNumHyperspheres / node_size;
                int remainder_local = localNumHyperspheres % node_size;

                int localStartIdx = startIdx + node_rank * numPerRank + std::min(node_rank, remainder_local);
                int localEndIdx = localStartIdx + numPerRank + (node_rank < remainder_local ? 1 : 0);
                int numLocalHyperspheres = localEndIdx - localStartIdx;

                // Initialize a vector to store local hyperspheres
                std::vector<Hypersphere> myHyperspheres;
                for (int i = localStartIdx; i < localEndIdx; ++i) {
                    myHyperspheres.push_back(currentHyperspheres[i]);
                }

                // Variables to hold the local best objective value and point
                double localBestObjective = std::numeric_limits<double>::max();
                Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
                    localBestPoint("localBestPoint", dimension);
                Kokkos::deep_copy(localBestPoint, 0.0); // Initialize to zeros

                // Process all assigned hyperspheres
                for (auto& hs : myHyperspheres) {
                    // Perform intensive local search
                    auto solution = intensiveLocalSearch(hs, initialCenter, initialRadius, maxIterations, stepSize, phi, omega_min);

                    // Compute the objective value on device
                    double objValue = computeObjectiveValue(solution);

                    // Update local best if necessary
                    if (objValue < localBestObjective) {
                        localBestObjective = objValue;
                        Kokkos::deep_copy(localBestPoint, solution);
                    }
                }

                // Within each node, find the local best solution
                struct {
                    double value;
                    int rank;
                } nodeLocalMin = { localBestObjective, node_rank }, nodeBestMin;

                // Perform reduction within node_comm
                MPI_Allreduce(&nodeLocalMin, &nodeBestMin, 1, MPI_DOUBLE_INT, MPI_MINLOC, node_comm);

                // Node leader collects the node best solution
                double nodeBestObjective = nodeBestMin.value;
                Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> nodeBestPoint("nodeBestPoint", dimension);
                if (node_rank == nodeBestMin.rank) {
                    Kokkos::deep_copy(nodeBestPoint, localBestPoint);
                }

                // Node leaders perform inter-node reduction
                struct {
                    double value;
                    int rank;
                } globalMin;
                if (node_rank == 0) {
                    struct {
                        double value;
                        int rank;
                    } nodeLeaderMin = { nodeBestObjective, node_id };
                    MPI_Allreduce(&nodeLeaderMin, &globalMin, 1, MPI_DOUBLE_INT, MPI_MINLOC, node_leader_comm);
                }

                // Broadcast globalMin to all processes
                MPI_Bcast(&globalMin, 1, MPI_DOUBLE_INT, 0, node_comm);

                // Node leader with the global best solution broadcasts the point to all nodes
                Kokkos::View<double*, Kokkos::HostSpace> globalBestPointHost("globalBestPointHost", dimension);

                if (node_rank == 0) {
                    if (node_id == globalMin.rank) {
                        auto nodeBestPointHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), nodeBestPoint);
                        MPI_Bcast(nodeBestPointHost.data(), dimension, MPI_DOUBLE, globalMin.rank, node_leader_comm);
                        Kokkos::deep_copy(globalBestPointHost, nodeBestPointHost);
                    } else {
                        Kokkos::View<double*, Kokkos::HostSpace> nodeBestPointHost("nodeBestPointHost", dimension);
                        MPI_Bcast(nodeBestPointHost.data(), dimension, MPI_DOUBLE, globalMin.rank, node_leader_comm);
                        Kokkos::deep_copy(globalBestPointHost, nodeBestPointHost);
                    }
                }

                // Broadcast the global best point to local processes
                MPI_Bcast(globalBestPointHost.data(), dimension, MPI_DOUBLE, 0, node_comm);

                // Copy the data back to globalBestPoint on device
                Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> globalBestPoint("globalBestPoint", dimension);
                Kokkos::deep_copy(globalBestPoint, globalBestPointHost);

                // Only the root process outputs the result
                if (world_rank == 0) {
                    // Output the point with the minimum objective function value
                    std::cout << "Minimum objective value: " << globalMin.value << std::endl;

                    // Output summary statistics
                    double minVal = globalBestPointHost(0);
                    double maxVal = globalBestPointHost(0);
                    double sum = 0.0;
                    for (size_t k = 0; k < globalBestPointHost.extent(0); ++k) {
                        double val = globalBestPointHost(k);
                        sum += val;
                        if (val < minVal) minVal = val;
                        if (val > maxVal) maxVal = val;
                    }
                    double meanVal = sum / globalBestPointHost.extent(0);
                    std::cout << "Solution vector statistics:" << std::endl;
                    std::cout << "Dimension: " << globalBestPointHost.extent(0) << std::endl;
                    std::cout << "Min value: " << minVal << std::endl;
                    std::cout << "Max value: " << maxVal << std::endl;
                    std::cout << "Mean value: " << meanVal << std::endl;

                    // Output the entire solution vector
                    std::cout << "Entire solution found: [";
                    for (size_t k = 0; k < globalBestPointHost.extent(0); ++k) {
                        std::cout << globalBestPointHost(k) << (k < globalBestPointHost.extent(0) - 1 ? ", " : "");
                    }
                    std::cout << "]" << std::endl;
                }

                // Exit the loop after exploitation
                break;
            } else {
                // Decompose selected hyperspheres for the next depth

                std::vector<Hypersphere> nextHyperspheres;
                for (auto& hs : currentHyperspheres) {
                    auto childHyperspheres = decomposeHypersphere(hs);
                    nextHyperspheres.insert(nextHyperspheres.end(), childHyperspheres.begin(), childHyperspheres.end());
                }

                // Update currentHyperspheres for the next depth
                currentHyperspheres = std::move(nextHyperspheres);
            }

            if (world_rank == 0) {
                std::cout << "Completed depth: " << depth << std::endl;
            }
        }

        // Calculate total duration
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        // Only rank 0 prints the total execution time
        if (world_rank == 0) {
            std::cout << "Total Execution time: " << duration.count() << " seconds" << std::endl;
        }
    }
    // Finalize Kokkos and MPI
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
