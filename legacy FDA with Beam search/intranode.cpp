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
























// Score a hypersphere by computing the average objective value of sampled points
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

    auto team_policy = Kokkos::TeamPolicy<>(3, Kokkos::AUTO); // 3 points, AUTO threads per team

    Kokkos::parallel_reduce(
        "ComputeScore",
        team_policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team, double& teamSum) {
            int i = team.league_rank(); // Each team works on one point

            double objValue = 1.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, points.extent(1)), [&](const int dim, double& prod) {
                double x = points(i, dim);
                double term = 0.0;
                for (int k = 1; k <= 5; ++k) {
                    term += computeShubertTerm(x, k);
                }
                prod *= term;
            }, Kokkos::Prod<double>(objValue));

            // Team-wide reduction
            Kokkos::single(Kokkos::PerTeam(team), [&]() {
                teamSum += objValue;
            });
        }, totalScore);

    // Compute the arithmetic mean
    return totalScore / 3.0;
}
















// Compute the objective value of the solution using parallel_reduce
double computeObjectiveValue(const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& solution) {
    double objValue = 0.0;

    // Using a lambda that accesses the entire view
    Kokkos::parallel_reduce("ComputeObjectiveValue", 1,
        KOKKOS_LAMBDA(const int, double& lsum) {
            lsum = objectiveFunction(Kokkos::subview(solution, Kokkos::ALL()));
        }, objValue);

    return objValue;
}



















// Intensive local search algorithm with multidimensional parallelism
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

    Kokkos::parallel_for("IntensiveLocalSearch", policy,
        KOKKOS_LAMBDA(const Kokkos::TeamPolicy<exec_space>::member_type& team) {

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension), [=](int dim) {
                localBestPoint(dim) = bestSolution(dim);
            });

            team.team_barrier();

            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension), [=](int dim) {
                double x = localBestPoint(dim);
                double term = 0.0;
                for (int i = 1; i <= 5; ++i) {
                    term += computeShubertTerm(x, i);
                }
                term_d(dim) = term;
            });

            team.team_barrier();

            double bestValue = 1.0;
            Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, hs.dimension),
                [=](int dim, double& prod) {
                    prod *= term_d(dim);
                }, Kokkos::Prod<double>(bestValue));

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

                    double neighborValue1, neighborValue2;

                    double original_x = localBestPoint(dim);
                    double original_term = term_d(dim);

                    double term_xs1 = 0.0;
                    for (int i = 1; i <= 5; ++i) {
                        term_xs1 += computeShubertTerm(xs1, i);
                    }
                    neighborValue1 = (bestValue / original_term) * term_xs1;

                    double term_xs2 = 0.0;
                    for (int i = 1; i <= 5; ++i) {
                        term_xs2 += computeShubertTerm(xs2, i);
                    }
                    neighborValue2 = (bestValue / original_term) * term_xs2;

                    double localBestValue = bestValue;
                    double newTerm = original_term;
                    double new_x = original_x;

                    if (neighborValue1 < localBestValue) {
                        localBestValue = neighborValue1;
                        new_x = xs1;
                        newTerm = term_xs1;
                        improvementMade(0) = true;
                    }
                    if (neighborValue2 < localBestValue) {
                        localBestValue = neighborValue2;
                        new_x = xs2;
                        newTerm = term_xs2;
                        improvementMade(0) = true;
                    }

                    newValues(dim) = new_x;
                    term_d(dim) = newTerm;
                });

                team.team_barrier();

                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension), [=](int dim) {
                    localBestPoint(dim) = newValues(dim);
                });

                team.team_barrier();

                Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, hs.dimension),
                    [=](int dim, double& prod) {
                        prod *= term_d(dim);
                    }, Kokkos::Prod<double>(bestValue));

                team.team_barrier();

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

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Define parameters for the search
        int dimension = 2000; // Changed to non-const to avoid issues with MPI_Bcast if needed
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
        uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() + rank;
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

        std::cout << "Rank " << rank << " is entering the main loop." << std::endl;

        // Main iterative process
        for (int depth = 1; depth <= maxDepth; ++depth) {
            // Distribute currentHyperspheres among ranks for scoring
            int numHyperspheres = currentHyperspheres.size();
            int numPerRank = numHyperspheres / num_procs;
            int remainder = numHyperspheres % num_procs;

            // Calculate the starting and ending indices for the current rank
            int startIdx = rank * numPerRank + std::min(rank, remainder);
            int endIdx = startIdx + numPerRank + (rank < remainder ? 1 : 0);

            // Determine the number of hyperspheres assigned to this rank
            int localNumHyperspheres = endIdx - startIdx;

            // Initialize a vector to store local HypersphereScores
            std::vector<HypersphereScore> localScores;
            localScores.reserve(localNumHyperspheres);

            // Parallel computation of scores using Kokkos
            for (int i = 0; i < localNumHyperspheres; ++i) {
                int idx = startIdx + i;
                double score = scoreHypersphere(currentHyperspheres[idx], initialCenter, initialRadius);
                localScores.emplace_back(currentHyperspheres[idx], score);
            }

            // Prepare data for communication
            std::vector<double> localScoresOnly(localNumHyperspheres);
            std::vector<int> localIDsSerialized;

            // Serialize identifiers
            for (int i = 0; i < localNumHyperspheres; ++i) {
                localScoresOnly[i] = localScores[i].score;
                auto serializedID = serializeHypersphereID(localScores[i].hs.id);
                localIDsSerialized.insert(localIDsSerialized.end(), serializedID.begin(), serializedID.end());
            }

            // Communicate sizes
            int localIDSize = localIDsSerialized.size();
            std::vector<int> allNumHyperspheres(num_procs);
            std::vector<int> allIDSizes(num_procs);
            MPI_Allgather(&localNumHyperspheres, 1, MPI_INT,
                          allNumHyperspheres.data(), 1, MPI_INT, MPI_COMM_WORLD);
            MPI_Allgather(&localIDSize, 1, MPI_INT,
                          allIDSizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

            // Compute displacements
            int totalHyperspheres = std::accumulate(allNumHyperspheres.begin(), allNumHyperspheres.end(), 0);
            int totalIDSize = std::accumulate(allIDSizes.begin(), allIDSizes.end(), 0);

            std::vector<int> displsScores(num_procs, 0);
            std::vector<int> displsIDs(num_procs, 0);
            for (int i = 1; i < num_procs; ++i) {
                displsScores[i] = displsScores[i - 1] + allNumHyperspheres[i - 1];
                displsIDs[i] = displsIDs[i - 1] + allIDSizes[i - 1];
            }

            // Gather all scores
            std::vector<double> allScores(totalHyperspheres);
            MPI_Allgatherv(localScoresOnly.data(), localNumHyperspheres, MPI_DOUBLE,
                           allScores.data(), allNumHyperspheres.data(), displsScores.data(), MPI_DOUBLE, MPI_COMM_WORLD);

            // Gather all serialized IDs
            std::vector<int> allIDsSerialized(totalIDSize);
            MPI_Allgatherv(localIDsSerialized.data(), localIDSize, MPI_INT,
                           allIDsSerialized.data(), allIDSizes.data(), displsIDs.data(), MPI_INT, MPI_COMM_WORLD);

            // Reconstruct all IDs
            std::vector<HypersphereID> allIDs(totalHyperspheres);
            size_t offset = 0;
            for (int i = 0; i < totalHyperspheres; ++i) {
                allIDs[i] = deserializeHypersphereID(allIDsSerialized, offset);
            }

            // Select best hyperspheres
            std::vector<int> selectedIndices = selectBestIndices(allScores, beta);

            // Each process reconstructs the selected hyperspheres
            std::vector<Hypersphere> selectedHyperspheres;
            for (int idx : selectedIndices) {
                HypersphereID selectedID = allIDs[idx];
                Hypersphere hs(dimension, depth, selectedID);
                selectedHyperspheres.push_back(hs);
            }

            // Update currentHyperspheres for the next depth
            currentHyperspheres = std::move(selectedHyperspheres);

            // Check for maximum depth
            if (depth == maxDepth) {
                // Exploit selected hyperspheres

                int numSelected = currentHyperspheres.size();
                int numPerRank = numSelected / num_procs;
                int remainder = numSelected % num_procs;

                int startIdx = rank * numPerRank + std::min(rank, remainder);
                int endIdx = startIdx + numPerRank + (rank < remainder ? 1 : 0);

                // Assign all hyperspheres in [startIdx, endIdx) to this rank
                std::vector<Hypersphere> myHyperspheres;
                if (startIdx < endIdx && endIdx <= numSelected) {
                    myHyperspheres.assign(currentHyperspheres.begin() + startIdx, currentHyperspheres.begin() + endIdx);
                } else if (startIdx < endIdx && startIdx < numSelected) {
                    // Handle cases where endIdx exceeds numSelected
                    myHyperspheres.assign(currentHyperspheres.begin() + startIdx, currentHyperspheres.end());
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

                // Now, perform MPI reduction to find the global best
                struct {
                    double value;
                    int rank;
                } localMin = { localBestObjective, rank }, globalMin;

                int mpi_error = MPI_Reduce(&localMin, &globalMin, 1, MPI_DOUBLE_INT, MPI_MINLOC, 0, MPI_COMM_WORLD);
                if (mpi_error != MPI_SUCCESS) {
                    std::cerr << "MPI_Reduce failed with error code " << mpi_error << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, mpi_error);
                }

                // Broadcast the global minimum information to all ranks
                mpi_error = MPI_Bcast(&globalMin, 1, MPI_DOUBLE_INT, 0, MPI_COMM_WORLD);
                if (mpi_error != MPI_SUCCESS) {
                    std::cerr << "MPI_Bcast (globalMin) failed with error code " << mpi_error << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, mpi_error);
                }

                // Create a host mirror of globalBestPoint
                auto globalBestPointHost = Kokkos::View<double*, Kokkos::HostSpace>("globalBestPointHost", dimension);
                if (rank == globalMin.rank) {
                    // Create a host mirror for localBestPoint
                    auto localBestPointHost = Kokkos::create_mirror_view(localBestPoint);
                    Kokkos::deep_copy(localBestPointHost, localBestPoint);

                    // Copy localBestPointHost to globalBestPointHost
                    Kokkos::deep_copy(globalBestPointHost, localBestPointHost);
                }

                // Broadcast the global best point to all ranks using host data
                mpi_error = MPI_Bcast(globalBestPointHost.data(), dimension, MPI_DOUBLE, globalMin.rank, MPI_COMM_WORLD);
                if (mpi_error != MPI_SUCCESS) {
                    std::cerr << "MPI_Bcast (globalBestPoint) failed with error code " << mpi_error << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, mpi_error);
                }

                // Copy the data back to globalBestPoint on device
                Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
                    globalBestPoint("globalBestPoint", dimension);
                Kokkos::deep_copy(globalBestPoint, globalBestPointHost);

                // Only rank 0 outputs the result
                if (rank == 0) {
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

            std::cout << "Rank " << rank << " completed depth: " << depth << std::endl;
        }

        // Calculate total duration
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        // Only rank 0 prints the total execution time
        if (rank == 0) {
            std::cout << "Total Execution time: " << duration.count() << " seconds" << std::endl;
        }
    }
    // Finalize Kokkos and MPI
    Kokkos::finalize();
    MPI_Finalize();
    return 0;
}
