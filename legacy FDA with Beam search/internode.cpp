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

#ifdef KOKKOS_ENABLE_CUDA
#include <cuda_runtime.h>
#endif

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

// Helper function for Sphere function
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

// Reconstruct the center of a hypersphere
Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
reconstructCenter(const Hypersphere& hs,
                  const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initialCenter,
                  double initialRadius) {
    int dimension = hs.dimension;
    int depth = hs.depth;

    // Reconstruct center in host memory
    Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> centerHost("centerHost", dimension);

    // Copy initialCenter to centerHost
    auto initialCenterHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), initialCenter);
    Kokkos::deep_copy(centerHost, initialCenterHost);

    double r = initialRadius;
    for (int d = 0; d < depth; ++d) {
        double sqrt2 = sqrt(2.0);
        double r_prime = r / (1.0 + sqrt2);
        double offset = r - r_prime;
        int dim = hs.id[d].first;
        int sign = hs.id[d].second;

        centerHost(dim) += sign * offset;
        r = r_prime; // Update radius for next level
    }

    // Now, create device copy of centerHost
    Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> center("center", dimension);
    Kokkos::deep_copy(center, centerHost);

    return center;
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

// Function to flatten HypersphereIDs and prepare data for device processing
void prepareHypersphereData(
    const std::vector<Hypersphere>& hyperspheres,
    Kokkos::View<int*, mem_space>& depths,
    Kokkos::View<size_t*, mem_space>& idOffsets,
    Kokkos::View<int*, mem_space>& idDims,
    Kokkos::View<int*, mem_space>& idSigns) {

    size_t numHyperspheres = hyperspheres.size();

    // Allocate device views
    depths = Kokkos::View<int*, mem_space>("depths", numHyperspheres);
    idOffsets = Kokkos::View<size_t*, mem_space>("idOffsets", numHyperspheres + 1);

    // Calculate total size for flattened IDs
    size_t totalIDSize = 0;
    for (const auto& hs : hyperspheres) {
        totalIDSize += hs.id.size();
    }

    idDims = Kokkos::View<int*, mem_space>("idDims", totalIDSize);
    idSigns = Kokkos::View<int*, mem_space>("idSigns", totalIDSize);

    // Host mirrors for data preparation
    auto depthsHost = Kokkos::create_mirror_view(depths);
    auto idOffsetsHost = Kokkos::create_mirror_view(idOffsets);
    auto idDimsHost = Kokkos::create_mirror_view(idDims);
    auto idSignsHost = Kokkos::create_mirror_view(idSigns);

    // Flatten the IDs and fill depths and offsets
    size_t currentOffset = 0;
    for (size_t i = 0; i < numHyperspheres; ++i) {
        depthsHost(i) = hyperspheres[i].depth;
        idOffsetsHost(i) = currentOffset;
        for (const auto& pair : hyperspheres[i].id) {
            idDimsHost(currentOffset) = pair.first;
            idSignsHost(currentOffset) = pair.second;
            currentOffset++;
        }
    }
    idOffsetsHost(numHyperspheres) = currentOffset;

    // Copy data to device
    Kokkos::deep_copy(depths, depthsHost);
    Kokkos::deep_copy(idOffsets, idOffsetsHost);
    Kokkos::deep_copy(idDims, idDimsHost);
    Kokkos::deep_copy(idSigns, idSignsHost);
}

// Function to reconstruct centers and radii for all hyperspheres
void reconstructCentersAndRadii(
    size_t numHyperspheres,
    int dimension,
    double initialRadius,
    const Kokkos::View<int*, mem_space>& depths,
    const Kokkos::View<size_t*, mem_space>& idOffsets,
    const Kokkos::View<int*, mem_space>& idDims,
    const Kokkos::View<int*, mem_space>& idSigns,
    const Kokkos::View<double*, mem_space>& initialCenterDevice,
    Kokkos::View<double**, mem_space>& centers,
    Kokkos::View<double*, mem_space>& radii) {

    // Allocate centers and radii
    centers = Kokkos::View<double**, mem_space>("centers", numHyperspheres, dimension);
    radii = Kokkos::View<double*, mem_space>("radii", numHyperspheres);

    Kokkos::parallel_for("ReconstructCenters", Kokkos::RangePolicy<>(0, numHyperspheres), KOKKOS_LAMBDA(int i) {
        int depth = depths(i);
        size_t idStart = idOffsets(i);

        // Copy initialCenter to centers
        for (int j = 0; j < dimension; ++j) {
            centers(i, j) = initialCenterDevice(j);
        }

        double r = initialRadius;
        for (int d = 0; d < depth; ++d) {
            double sqrt2 = sqrt(2.0);
            double r_prime = r / (1.0 + sqrt2);
            double offset = r - r_prime;

            int dim = idDims(idStart + d);
            int sign = idSigns(idStart + d);

            centers(i, dim) += sign * offset;
            r = r_prime; // Update radius for next level
        }
        radii(i) = r;
    });
}

// Function to score all hyperspheres using hierarchical parallelism without shared memory
void scoreHyperspheres(
    size_t numHyperspheres,
    int dimension,
    const Kokkos::View<double**, mem_space>& centers,
    const Kokkos::View<double*, mem_space>& radii,
    const Kokkos::View<double*, mem_space>& bestSolutionDevice,
    double bestObjectiveValue,
    Kokkos::View<double*, mem_space>& kokkosScores) {

    // Initialize kokkosScores to zero
    Kokkos::deep_copy(kokkosScores, 0.0);

    // Number of points per hypersphere
    const int numPoints = 3;

    // Define team policy
    using team_policy = Kokkos::TeamPolicy<>;
    using member_type = team_policy::member_type;

    // Parallel loop over hyperspheres
    Kokkos::parallel_for("ScoreHyperspheres", team_policy(numHyperspheres, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& team) {
        const int i = team.league_rank(); // Each team works on one hypersphere

        double totalScore = 0.0;
        double r = radii(i);

        // Loop over the number of points
        for (int p = 0; p < numPoints; ++p) {

            // Since all points are initialized with 0.5, we can compute norm directly
            double norm = sqrt(dimension * 0.5 * 0.5);

            // Variables to accumulate fx and distance
            double fx = 0.0;
            double distance = 0.0;

            // Parallel reduction over dimensions for fx
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, dimension),
                [=](const int d, double& fx_local) {
                    double val = 0.5;
                    double point_d = centers(i, d) + (val / norm) * r;

                    // Objective function calculation (fx)
                    fx_local += computeSphereTerm(point_d);
                }, fx);

            // Parallel reduction over dimensions for distance
            Kokkos::parallel_reduce(
                Kokkos::TeamThreadRange(team, dimension),
                [=](const int d, double& distance_local) {
                    double val = 0.5;
                    double point_d = centers(i, d) + (val / norm) * r;

                    // Distance calculation
                    double diff = point_d - bestSolutionDevice(d);
                    distance_local += diff * diff;
                }, distance);

            distance = sqrt(distance);

            // Avoid division by zero
            if (distance == 0.0) {
                distance = 1e-10;
            }

            double c_dttb = (fx - bestObjectiveValue) / distance;

            totalScore += c_dttb;

            // Optional: team.team_barrier(); // Not strictly necessary here
        }

        // Compute mean score
        kokkosScores(i) = totalScore / numPoints;
    });
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

// Function to perform intensive local search
Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
intensiveLocalSearch(const Hypersphere& hs,
                     const Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>& initialCenter,
                     double initialRadius,
                     int maxIterations, double stepSize,
                     double phi, double omega_min) {

    // Reconstruct center
    auto center = reconstructCenter(hs, initialCenter, initialRadius);

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

                    double localBestValue = term_d(dim);
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




// Function to get available GPU memory
size_t get_available_memory() {
    size_t free_mem = 0;
    size_t total_mem = 0;
#ifdef KOKKOS_ENABLE_CUDA
    cudaMemGetInfo(&free_mem, &total_mem);
#else
    // For other backends, set free_mem to a large value or a default
    free_mem = SIZE_MAX;
#endif
    return free_mem;
}




// Function to parse command-line arguments for beta (beam width)
int parseBeta(int argc, char* argv[], int world_rank) {
    int beta = 50; // Default value
    if (argc > 1) {
        try {
            beta = std::stoi(argv[1]);
            if (beta <= 0) {
                if (world_rank == 0) {
                    std::cerr << "Beam width (beta) must be a positive integer. Using default value 50.\n";
                }
                beta = 50;
            }
        } catch (const std::invalid_argument& e) {
            if (world_rank == 0) {
                std::cerr << "Invalid beam width (beta) argument. Using default value 50.\n";
            }
        } catch (const std::out_of_range& e) {
            if (world_rank == 0) {
                std::cerr << "Beam width (beta) argument out of range. Using default value 50.\n";
            }
        }
    }
    return beta;
}



// Add a function to parse command-line arguments for dimension
int parseDimension(int argc, char* argv[], int world_rank) {
    int dimension = 20000; // Default value
    if (argc > 2) {
        try {
            dimension = std::stoi(argv[2]);
        } catch (const std::invalid_argument& e) {
            if (world_rank == 0) {
                std::cerr << "Invalid dimension argument. Using default value 20000.\n";
            }
        }
    }
    return dimension;
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

    // Parse beam_width (beta) and dimension from command-line arguments
    const int beta = parseBeta(argc, argv, world_rank);
    int dimension = parseDimension(argc, argv, world_rank);


    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        auto start = std::chrono::high_resolution_clock::now();

        // Define parameters for the search
        const int maxDepth = 5;
        // Parameters for applying the intensive local search starting from the center
        int maxIterations = 100;   // Number of iterations for the search
        double stepSize = 1.75;     // Initial step size (ω)
        double phi = 0.5;          // Factor to reduce step size
        double omega_min = 1e-20;  // Minimum step size before stopping

        // Initialize the search space
        Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
            lowerBounds("lowerBounds", dimension);
        Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
            upperBounds("upperBounds", dimension);
        Kokkos::deep_copy(lowerBounds, -5.12); // Example bounds
        Kokkos::deep_copy(upperBounds, 5.12);
        SearchSpaceBounds searchSpace(dimension, lowerBounds, upperBounds);

        // Seed for random number generation (unique per rank)
        uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count() + world_rank;
        Kokkos::Random_XorShift64_Pool<exec_space> randomPool(seed);

        // Compute initial hypersphere center
        Kokkos::View<double*, Kokkos::LayoutLeft, mem_space>
            initialCenter("initialCenter", dimension);
        Kokkos::parallel_for("ComputeInitialCenter", Kokkos::RangePolicy<exec_space>(0, dimension),
            KOKKOS_LAMBDA(const int j) {
                initialCenter(j) = -5.12 + (5.12 - (-5.12)) / 2.0; // Center at 0.0
            });

        // Compute radius: r = (U_j - L_j) / 2
        double initialRadius = 5.12; // Since upperBounds - lowerBounds = 10.24, r = 5.12

        // Initialize best solution and objective value
        double bestObjectiveValue = std::numeric_limits<double>::max();
        Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> bestSolution("bestSolution", dimension);
        Kokkos::deep_copy(bestSolution, initialCenter);

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

            // Initialize a vector to store local hyperspheres
            std::vector<Hypersphere> myHyperspheres;
            for (int i = localStartIdx; i < localEndIdx; ++i) {
                myHyperspheres.push_back(currentHyperspheres[i]);
            }

            // **Modified Code Starts Here**

            // Get available GPU memory
            size_t available_memory = get_available_memory();

            // Estimate per-hypersphere memory usage
            size_t per_hypersphere_memory = (dimension + 2) * sizeof(double);

            // Calculate max number of hyperspheres per chunk
            size_t max_num_hyperspheres = static_cast<size_t>(available_memory * 0.9) / per_hypersphere_memory;
            if (max_num_hyperspheres == 0) {
                max_num_hyperspheres = 1;
            }

            // Prepare to collect scores from all chunks
            std::vector<double> localScoresOnly;
            std::vector<Hypersphere> localHyperspheres;

            size_t totalHyperspheres = numLocalHyperspheres;

            for (size_t chunk_start = 0; chunk_start < totalHyperspheres; chunk_start += max_num_hyperspheres) {
                size_t chunk_size = std::min(max_num_hyperspheres, totalHyperspheres - chunk_start);

                // Get the hyperspheres for this chunk
                std::vector<Hypersphere> chunkHyperspheres(myHyperspheres.begin() + chunk_start, myHyperspheres.begin() + chunk_start + chunk_size);

                // Prepare hypersphere data
                Kokkos::View<int*, mem_space> depths_chunk;
                Kokkos::View<size_t*, mem_space> idOffsets_chunk;
                Kokkos::View<int*, mem_space> idDims_chunk;
                Kokkos::View<int*, mem_space> idSigns_chunk;

                prepareHypersphereData(chunkHyperspheres, depths_chunk, idOffsets_chunk, idDims_chunk, idSigns_chunk);

                // Reconstruct centers and radii
                Kokkos::View<double**, mem_space> centers_chunk;
                Kokkos::View<double*, mem_space> radii_chunk;

                reconstructCentersAndRadii(
                    chunk_size,
                    dimension,
                    initialRadius,
                    depths_chunk,
                    idOffsets_chunk,
                    idDims_chunk,
                    idSigns_chunk,
                    initialCenter,
                    centers_chunk,
                    radii_chunk);

                // Allocate scores view
                Kokkos::View<double*, mem_space> kokkosScores_chunk("scores", chunk_size);

                // Score hyperspheres in parallel
                scoreHyperspheres(
                    chunk_size,
                    dimension,
                    centers_chunk,
                    radii_chunk,
                    bestSolution,
                    bestObjectiveValue,
                    kokkosScores_chunk);

                // Copy scores back to host and process
                auto scoresHost_chunk = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), kokkosScores_chunk);

                // Collect scores and hyperspheres from the chunk
                for (size_t i = 0; i < chunk_size; ++i) {
                    localScoresOnly.push_back(scoresHost_chunk(i));
                    localHyperspheres.push_back(chunkHyperspheres[i]);
                }
            }

            // **Modified Code Ends Here**

            // Local selection of top 'beta' hyperspheres
            int localBeta = std::min(beta, static_cast<int>(localScoresOnly.size()));
            std::vector<int> localSelectedIndices = selectBestIndices(localScoresOnly, localBeta);

            std::vector<double> localTopScores(localBeta);
            std::vector<HypersphereID> localTopIDs(localBeta);
            for (int i = 0; i < localBeta; ++i) {
                int idx = localSelectedIndices[i];
                localTopScores[i] = localScoresOnly[idx];
                localTopIDs[i] = localHyperspheres[idx].id;
            }

            // Now, gather the top 'beta' hyperspheres within the node
            // Prepare data for MPI communication
            int localDataSize = localBeta;
            std::vector<int> allDataSizes(node_size);
            MPI_Allgather(&localDataSize, 1, MPI_INT, allDataSizes.data(), 1, MPI_INT, node_comm);

            int totalDataSize = std::accumulate(allDataSizes.begin(), allDataSizes.end(), 0);

            // Prepare displacements
            std::vector<int> displs(node_size, 0);
            for (int i = 1; i < node_size; ++i) {
                displs[i] = displs[i - 1] + allDataSizes[i - 1];
            }

            // Gather scores
            std::vector<double> allScores(totalDataSize);
            MPI_Allgatherv(localTopScores.data(), localDataSize, MPI_DOUBLE,
                           allScores.data(), allDataSizes.data(), displs.data(), MPI_DOUBLE, node_comm);

            // Serialize localTopIDs
            std::vector<int> localIDsSerialized;
            for (const auto& id : localTopIDs) {
                auto serializedID = serializeHypersphereID(id);
                localIDsSerialized.insert(localIDsSerialized.end(), serializedID.begin(), serializedID.end());
            }

            // Gather sizes of serialized IDs
            int localIDSize = localIDsSerialized.size();
            std::vector<int> allIDSizes(node_size);
            MPI_Allgather(&localIDSize, 1, MPI_INT, allIDSizes.data(), 1, MPI_INT, node_comm);

            int totalIDSize = std::accumulate(allIDSizes.begin(), allIDSizes.end(), 0);

            // Prepare displacements for IDs
            std::vector<int> displsIDs(node_size, 0);
            for (int i = 1; i < node_size; ++i) {
                displsIDs[i] = displsIDs[i - 1] + allIDSizes[i - 1];
            }

            // Gather serialized IDs
            std::vector<int> allIDsSerialized(totalIDSize);
            MPI_Allgatherv(localIDsSerialized.data(), localIDSize, MPI_INT,
                           allIDsSerialized.data(), allIDSizes.data(), displsIDs.data(), MPI_INT, node_comm);

            // Reconstruct all IDs
            std::vector<HypersphereID> allIDs(totalDataSize);
            size_t offset = 0;
            for (int i = 0; i < totalDataSize; ++i) {
                allIDs[i] = deserializeHypersphereID(allIDsSerialized, offset);
            }

            // Node-level selection of top 'beta' hyperspheres
            int nodeBeta = std::min(beta, totalDataSize);
            std::vector<int> nodeSelectedIndices = selectBestIndices(allScores, nodeBeta);

            std::vector<double> nodeTopScores(nodeBeta);
            std::vector<HypersphereID> nodeTopIDs(nodeBeta);
            for (int i = 0; i < nodeBeta; ++i) {
                int idx = nodeSelectedIndices[i];
                nodeTopScores[i] = allScores[idx];
                nodeTopIDs[i] = allIDs[idx];
            }

            // Node leaders exchange best hyperspheres across nodes
            if (node_rank == 0) {
                // Prepare data for inter-node communication
                int nodeLocalDataSize = nodeBeta;
                std::vector<int> nodeAllDataSizes(num_nodes);
                MPI_Allgather(&nodeLocalDataSize, 1, MPI_INT, nodeAllDataSizes.data(), 1, MPI_INT, node_leader_comm);

                int nodeTotalDataSize = std::accumulate(nodeAllDataSizes.begin(), nodeAllDataSizes.end(), 0);

                // Prepare displacements
                std::vector<int> nodeDispls(num_nodes, 0);
                for (int i = 1; i < num_nodes; ++i) {
                    nodeDispls[i] = nodeDispls[i - 1] + nodeAllDataSizes[i - 1];
                }

                // Gather scores among node leaders
                std::vector<double> nodeAllScores(nodeTotalDataSize);
                MPI_Allgatherv(nodeTopScores.data(), nodeLocalDataSize, MPI_DOUBLE,
                               nodeAllScores.data(), nodeAllDataSizes.data(), nodeDispls.data(), MPI_DOUBLE, node_leader_comm);

                // Serialize nodeTopIDs
                std::vector<int> nodeLocalIDsSerialized;
                for (const auto& id : nodeTopIDs) {
                    auto serializedID = serializeHypersphereID(id);
                    nodeLocalIDsSerialized.insert(nodeLocalIDsSerialized.end(), serializedID.begin(), serializedID.end());
                }

                // Gather sizes of serialized IDs among node leaders
                int nodeLocalIDSize = nodeLocalIDsSerialized.size();
                std::vector<int> nodeAllIDSizes(num_nodes);
                MPI_Allgather(&nodeLocalIDSize, 1, MPI_INT, nodeAllIDSizes.data(), 1, MPI_INT, node_leader_comm);

                int nodeTotalIDSize = std::accumulate(nodeAllIDSizes.begin(), nodeAllIDSizes.end(), 0);

                // Prepare displacements for IDs
                std::vector<int> nodeDisplsIDs(num_nodes, 0);
                for (int i = 1; i < num_nodes; ++i) {
                    nodeDisplsIDs[i] = nodeDisplsIDs[i - 1] + nodeAllIDSizes[i - 1];
                }

                // Gather serialized IDs among node leaders
                std::vector<int> nodeAllIDsSerialized(nodeTotalIDSize);
                MPI_Allgatherv(nodeLocalIDsSerialized.data(), nodeLocalIDSize, MPI_INT,
                               nodeAllIDsSerialized.data(), nodeAllIDSizes.data(), nodeDisplsIDs.data(), MPI_INT, node_leader_comm);

                // Reconstruct all IDs
                std::vector<HypersphereID> nodeAllIDs(nodeTotalDataSize);
                size_t offsetNode = 0;
                for (int i = 0; i < nodeTotalDataSize; ++i) {
                    nodeAllIDs[i] = deserializeHypersphereID(nodeAllIDsSerialized, offsetNode);
                }

                // Global selection of top 'beta' hyperspheres
                int globalBeta = std::min(beta, nodeTotalDataSize);
                std::vector<int> globalSelectedIndices = selectBestIndices(nodeAllScores, globalBeta);

                std::vector<Hypersphere> selectedHyperspheres;
                for (int idx : globalSelectedIndices) {
                    HypersphereID selectedID = nodeAllIDs[idx];
                    Hypersphere hs(dimension, depth, selectedID);
                    selectedHyperspheres.push_back(hs);
                }

                // Evaluate points of the best-scored hypersphere
                Hypersphere bestHypersphere = selectedHyperspheres[0];

                // Reconstruct the center and radius of the best hypersphere
                auto bestCenter = reconstructCenter(bestHypersphere, initialCenter, initialRadius);

                // Generate points around the center
                auto points = generateRandomPointsAroundCenter(bestCenter, initialRadius, dimension);

                // Evaluate the objective function at each point
                auto pointsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), points);

                double localBestObjective = std::numeric_limits<double>::max();
                Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> localBestSolution("localBestSolution", dimension);

                for (int i = 0; i < pointsHost.extent(0); ++i) {
                    auto x = Kokkos::subview(pointsHost, i, Kokkos::ALL());
                    double fx = sphereObjectiveFunction(x);

                    if (fx < localBestObjective) {
                        localBestObjective = fx;
                        for (int d = 0; d < dimension; ++d) {
                            localBestSolution(d) = x(d);
                        }
                    }
                }

                // Update bestSolution and bestObjectiveValue
                bestObjectiveValue = localBestObjective;
                Kokkos::deep_copy(bestSolution, localBestSolution);

                // Broadcast selected hyperspheres to all nodes
                int numSelected = selectedHyperspheres.size();
                MPI_Bcast(&numSelected, 1, MPI_INT, 0, node_comm);

                // Serialize selected hyperspheres' IDs
                std::vector<int> selectedIDsSerialized;
                for (const auto& hs : selectedHyperspheres) {
                    auto serializedID = serializeHypersphereID(hs.id);
                    selectedIDsSerialized.insert(selectedIDsSerialized.end(), serializedID.begin(), serializedID.end());
                }

                int selectedTotalIDSize = selectedIDsSerialized.size();
                MPI_Bcast(&selectedTotalIDSize, 1, MPI_INT, 0, node_comm);

                MPI_Bcast(selectedIDsSerialized.data(), selectedTotalIDSize, MPI_INT, 0, node_comm);

                // Reconstruct selected hyperspheres
                std::vector<Hypersphere> selectedHyperspheresAll;
                offset = 0;
                for (int i = 0; i < numSelected; ++i) {
                    HypersphereID selectedID = deserializeHypersphereID(selectedIDsSerialized, offset);
                    Hypersphere hs(dimension, depth, selectedID);
                    selectedHyperspheresAll.push_back(hs);
                }

                // Update currentHyperspheres for the next depth
                currentHyperspheres = std::move(selectedHyperspheresAll);
            } else {
                // Non-leader nodes receive selected hyperspheres
                int numSelected;
                MPI_Bcast(&numSelected, 1, MPI_INT, 0, node_comm);

                int selectedTotalIDSize;
                MPI_Bcast(&selectedTotalIDSize, 1, MPI_INT, 0, node_comm);

                std::vector<int> selectedIDsSerialized(selectedTotalIDSize);
                MPI_Bcast(selectedIDsSerialized.data(), selectedTotalIDSize, MPI_INT, 0, node_comm);

                // Reconstruct selected hyperspheres
                std::vector<Hypersphere> selectedHyperspheres;
                offset = 0;
                for (int i = 0; i < numSelected; ++i) {
                    HypersphereID selectedID = deserializeHypersphereID(selectedIDsSerialized, offset);
                    Hypersphere hs(dimension, depth, selectedID);
                    selectedHyperspheres.push_back(hs);
                }

                // Update currentHyperspheres for the next depth
                currentHyperspheres = std::move(selectedHyperspheres);
            }

            // Broadcast bestObjectiveValue and bestSolution
            MPI_Bcast(&bestObjectiveValue, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

            auto hostBestSolution = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), bestSolution);
            MPI_Bcast(hostBestSolution.data(), dimension, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            Kokkos::deep_copy(bestSolution, hostBestSolution);

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

                    // Define the known global minimum for Sphere function
                    const double SPHERE_GLOBAL_MIN = 0.0;

                    // Calculate the difference between the found value and the global minimum
                    double sphereDifference = globalMin.value - SPHERE_GLOBAL_MIN;

                    // Print the Sphere metric
                    std::cout << "Sphere Difference from Global Minimum: " << sphereDifference << std::endl;

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