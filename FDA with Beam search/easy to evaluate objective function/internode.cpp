#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <mpi.h>


// Hypersphere structure definition
struct Hypersphere {
    Kokkos::View<double*> center; // Coordinates of the center
    double radius;                // Radius of the hypersphere
    int dimension;                // Dimension of the space


    KOKKOS_INLINE_FUNCTION Hypersphere() = default;

    KOKKOS_INLINE_FUNCTION ~Hypersphere() = default;


    Hypersphere(int dim, double rad, const Kokkos::View<double*>& ctr)
        : dimension(dim), radius(rad), center(Kokkos::View<double*>("center", dim)) {
            Kokkos::deep_copy(center, ctr);
        }

    Hypersphere(int dim, double rad)
        : dimension(dim), radius(rad), center(Kokkos::View<double*>("center", dim)) {
            Kokkos::deep_copy(center, Kokkos::View<double*>("zeros", dim));
        }

    KOKKOS_INLINE_FUNCTION
    void setProperties(int dim, double rad, const Kokkos::View<double*>& ctr) {
        dimension = dim;
        radius = rad;
        center = ctr;
        }


};

// SearchSpaceBounds structure definition
struct SearchSpaceBounds {
    Kokkos::View<double*> lowerBounds;
    Kokkos::View<double*> upperBounds;
    int dimension;

    // Constructor
    SearchSpaceBounds(int n, Kokkos::View<double*> lBounds, Kokkos::View<double*> uBounds)
        : dimension(n), lowerBounds(lBounds), upperBounds(uBounds) {}
};

struct CreateHyperspheresFunctor {
    Kokkos::View<Hypersphere*> hyperspheres;
    Kokkos::View<double**, Kokkos::LayoutRight> allCenters;
    SearchSpaceBounds searchSpace;
    int n;

    CreateHyperspheresFunctor(
        Kokkos::View<Hypersphere*> hs,
        Kokkos::View<double**, Kokkos::LayoutRight> cntrs,
        SearchSpaceBounds ss,
        int num)
        : hyperspheres(hs), allCenters(cntrs), searchSpace(ss), n(num) {}

    KOKKOS_INLINE_FUNCTION void operator()(const int i) const {
        double minRange = searchSpace.upperBounds(0) - searchSpace.lowerBounds(0);
        for (int j = 1; j < searchSpace.dimension; ++j) {
            double range = searchSpace.upperBounds(j) - searchSpace.lowerBounds(j);
            if (range < minRange) minRange = range;
        }
        double radius = minRange / (2 * n);
        auto center = Kokkos::subview(allCenters, i, Kokkos::ALL);

        for (int d = 0; d < searchSpace.dimension; ++d) {
            double step = (searchSpace.upperBounds(d) - searchSpace.lowerBounds(d)) / n;
            center(d) = searchSpace.lowerBounds(d) + step / 2 + i * step;
        }

        // Set properties of the Hypersphere instead of constructing a new one
        hyperspheres(i).setProperties(searchSpace.dimension, radius, center);
    }
};

// Compute a single term of the Shubert function
KOKKOS_INLINE_FUNCTION
double computeShubertTerm(double x, int i) {
    return i * sin((i + 1) * x + i);
}

// Compute the Shubert function in 10 dimensions using Kokkos View
KOKKOS_INLINE_FUNCTION
double objectiveFunction(const Kokkos::View<double*, Kokkos::LayoutRight>& point) {
    double result = 1.0;
    for (int d = 0; d < 10000; ++d) {
        double x = point(d);
        double term = 0.0;
        for (int i = 1; i <= 5; ++i) {
            term += computeShubertTerm(x, i);
        }
        result *= term;
    }
    return result;
}

// Compute the Shubert function in 10 dimensions using std::vector
double objectiveFunction(const std::vector<double>& point) {
    double result = 1.0;
    for (int d = 0; d < 10000; ++d) {
        double x = point[d];
        double term = 0.0;
        for (int i = 1; i <= 5; ++i) {
            term += computeShubertTerm(x, i);
        }
        result *= term;
    }
    return result;
}

// Compute the Shubert function in 10 dimensions using raw pointer
double objectiveFunction(const double* point, int dimension) {
    double result = 1.0;
    for (int d = 0; d < 10000; ++d) {
        double x = point[d];
        double term = 0.0;
        for (int i = 1; i <= 5; ++i) {
            term += computeShubertTerm(x, i);
        }
        result *= term;
    }
    return result;
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



Kokkos::View<double**, Kokkos::LayoutLeft> generateRandomPointsInHypersphere(const Hypersphere& hs, int numPoints, Kokkos::Random_XorShift64_Pool<>& randomPool) {
    Kokkos::View<double**, Kokkos::LayoutLeft> points("points", numPoints, hs.dimension); // Specify layout

    Kokkos::parallel_for("GeneratePoints", Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>({0, 0}, {numPoints, hs.dimension}), KOKKOS_LAMBDA(const int i, const int j) {
        auto gen = randomPool.get_state();
        double lengthSquared;
        double scale;
        do {
            lengthSquared = 0.0;
            points(i, j) = gen.drand(-1.0, 1.0);
            lengthSquared += points(i, j) * points(i, j);
            scale = std::cbrt(gen.drand()) * hs.radius / std::sqrt(lengthSquared);
        } while (lengthSquared > 1.0 || lengthSquared == 0.0);

        points(i, j) = hs.center(j) + points(i, j) * scale;
        randomPool.free_state(gen);
    });

    return points;
}


Kokkos::View<double**, Kokkos::LayoutLeft> generateRandomPointsAroundCenter(const Hypersphere& hs, Kokkos::Random_XorShift64_Pool<>& randomPool) {
    Kokkos::View<double**, Kokkos::LayoutLeft> points("points", 3, hs.dimension);

    // Generate random direction
    Kokkos::parallel_for("GenerateRandomPoints", Kokkos::RangePolicy<>(0, hs.dimension), KOKKOS_LAMBDA(const int j) {
        // Obtain random number generator state
        auto gen = randomPool.get_state();

        double theta = gen.drand(0.0, 2 * M_PI);
        double phi = gen.drand(0.0, M_PI);
        double x = sin(phi) * cos(theta);
        double y = sin(phi) * sin(theta);
        double z = cos(phi);

        // Calculate the distance to the center (radius)
        double distance_to_center = hs.radius;

        // Release random number generator state
        randomPool.free_state(gen);

        // Generate points around the center
        points(0, j) = hs.center(j); // Center point
        points(1, j) = hs.center(j) + x * distance_to_center; // Point along x-direction
        points(2, j) = hs.center(j) + y * distance_to_center; // Point along y-direction
    });

    return points;
}



double scoreHypersphere(const Hypersphere& hs, const Kokkos::View<double*, Kokkos::LayoutLeft>& trueBest, int numSamples, Kokkos::Random_XorShift64_Pool<> randomPool) {
    std::cout << "about to generate random points to score this hypersphere" << std::endl;

    // Switch layout of points to LayoutLeft
    Kokkos::View<double**, Kokkos::LayoutLeft> points_left("points", 3, hs.dimension);
    auto points = generateRandomPointsAroundCenter(hs,randomPool);
    Kokkos::deep_copy(points_left, points);

    std::cout << "generated them now let's go for the parallel reduce" << std::endl;
    double totalScore = 0.0;
    auto start_time = std::chrono::high_resolution_clock::now();

    Kokkos::parallel_reduce("OuterReduce", Kokkos::MDRangePolicy<Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left>>(
        {0, 0}, {numSamples, hs.dimension}),
    KOKKOS_LAMBDA(int i, int j, double& localSum) {
        auto point = Kokkos::subview(points_left, i, j);
        double distanceToTrueBest = (point() - trueBest(j)) * (point() - trueBest(j));
        localSum += std::sqrt(distanceToTrueBest);
    },
    totalScore);


    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Parallel loop execution time: " << elapsed.count() << " seconds" << std::endl;

    return totalScore / numSamples; // Average distance to true best
}










struct HypersphereScore {
    Hypersphere hypersphere;
    double score;

    HypersphereScore(const Hypersphere& hs, double sc)
        : hypersphere(hs), score(sc) {}
};

std::vector<Hypersphere> selectBestHyperspheres(std::vector<HypersphereScore>& hypersphereScores, int beta) {
    beta = std::min(beta, static_cast<int>(hypersphereScores.size()));

    // Sort based on score
    std::sort(hypersphereScores.begin(), hypersphereScores.end(),
              [](const HypersphereScore& a, const HypersphereScore& b) {
                  return a.score < b.score;
              });

    // Select top Beta hyperspheres
    std::vector<Hypersphere> selectedHyperspheres;
    for (int i = 0; i < beta; ++i) {
        selectedHyperspheres.push_back(hypersphereScores[i].hypersphere);
    }

    return selectedHyperspheres;
}












Kokkos::View<double*> generatePointInHypersphere(const Hypersphere& hs) {
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    Kokkos::View<double*> point("point", hs.dimension);

    Kokkos::parallel_for("GeneratePoint", 1, KOKKOS_LAMBDA(const int&) {
        auto rand_gen = rand_pool.get_state();
        double lengthSquared;
        do {
            lengthSquared = 0.0;
            for (int i = 0; i < hs.dimension; ++i) {
                point(i) = rand_gen.drand(-1.0, 1.0) * hs.radius;
                lengthSquared += point(i) * point(i);
            }
        } while (lengthSquared > hs.radius * hs.radius);
        rand_pool.free_state(rand_gen);
    });

    return point;
}

std::vector<Hypersphere> createHyperspheresInHypersphere(const Hypersphere& parentHs, int n, double childRadius) {
    std::vector<Hypersphere> hyperspheres;

    for (int i = 0; i < n; ++i) {
        auto center = generatePointInHypersphere(parentHs);
        Hypersphere hs(parentHs.dimension, childRadius, center);
        hyperspheres.push_back(hs);
    }

    return hyperspheres;
}














Kokkos::View<double**> generateRandomPointsInHyperspheref(const Hypersphere& hs, int numPoints) {
    uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    Kokkos::Random_XorShift64_Pool<> rand_pool(seed);
    Kokkos::View<double**> points("random_points", numPoints, hs.dimension);

    Kokkos::parallel_for("GeneratePoints", numPoints, KOKKOS_LAMBDA(const int i) {
        auto rand_gen = rand_pool.get_state();
        double lengthSquared = 0.0;
        double scale = 0.0;
        do {
            lengthSquared = 0.0;
            for (int dim = 0; dim < hs.dimension; ++dim) {
                points(i, dim) = rand_gen.drand(-1.0, 1.0);
                lengthSquared += points(i, dim) * points(i, dim);
            }
            scale = std::cbrt(rand_gen.drand()) * hs.radius;
        } while (lengthSquared > 1.0 || lengthSquared == 0.0);

        for (int dim = 0; dim < hs.dimension; ++dim) {
            points(i, dim) = hs.center(dim) + points(i, dim) * scale / std::sqrt(lengthSquared);
        }

        rand_pool.free_state(rand_gen);
    });

    return points;
}

std::vector<Hypersphere> promisingHypersphereSearch(const Hypersphere& hs, int numCandidates, int numBest, double innerRadius, Kokkos::Random_XorShift64_Pool<>& randomPool) {
    auto d_candidateCenters = generateRandomPointsInHypersphere(hs, numCandidates, randomPool);
    auto h_candidateCenters = Kokkos::create_mirror_view(d_candidateCenters);
    Kokkos::deep_copy(h_candidateCenters, d_candidateCenters);

    std::vector<Hypersphere> bestCandidates;
    for (int i = 0; i < numBest; ++i) {
        Kokkos::View<double*> center("best_center_host", hs.dimension);
        auto h_center = Kokkos::create_mirror_view(center);
        for (int dim = 0; dim < hs.dimension; ++dim) {
            h_center(dim) = h_candidateCenters(i, dim);
        }
        Kokkos::deep_copy(center, h_center);
        bestCandidates.emplace_back(hs.dimension, innerRadius, center);
    }

    return bestCandidates;
}


void printHypersphere(const Hypersphere& hs) {
    auto h_center = Kokkos::create_mirror_view(hs.center);
    Kokkos::deep_copy(h_center, hs.center);
    std::cout << "Hypersphere Center: (";
    for (int i = 0; i < hs.dimension; ++i) {
        std::cout << h_center(i);
        if (i < hs.dimension - 1) std::cout << ", ";
    }
    std::cout << "), Radius: " << hs.radius << std::endl;
}








Kokkos::View<double**> localSearchMultipleStarts(const Hypersphere& hs, const Kokkos::View<double**>& initialPoints, int maxIterations, double stepSize ) {
    Kokkos::Random_XorShift64_Pool<> rand_pool(std::chrono::high_resolution_clock::now().time_since_epoch().count());

    // Pre-allocate bestSolutions with the same dimensions as initialPoints
    Kokkos::View<double**> bestSolutions("bestSolutions", initialPoints.extent(0), hs.dimension);

    // Directly copy initialPoints to bestSolutions before entering parallel region
    Kokkos::deep_copy(bestSolutions, initialPoints);

    // local copy for calculations, avoiding dynamic allocation or deep_copy
    Kokkos::View<double*> localBestPoint("localBestPoint", hs.dimension);

    Kokkos::parallel_for("LocalSearchMultipleStarts", Kokkos::TeamPolicy<>(initialPoints.extent(0), Kokkos::AUTO), KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
        int i = team.league_rank();
        auto rand_gen = rand_pool.get_state(team.league_rank());

        // Initialize localBestPoint with initial values
        for (int dim = 0; dim < hs.dimension; ++dim) {
            localBestPoint(dim) = bestSolutions(i, dim);
        }

        double bestValue = objectiveFunction(localBestPoint);

        for (int iter = 0; iter < maxIterations; ++iter) {
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension), [&](int dim) {
                double step = rand_gen.drand(-stepSize, stepSize);
                localBestPoint(dim) += step; // Update local copy
            });

            double neighborValue = objectiveFunction(localBestPoint);
            if (neighborValue < bestValue) {
                bestValue = neighborValue;
            } else {
                // Revert the changes if neighbor is not better
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension), [&](int dim) {
                    localBestPoint(dim) -= stepSize; // Simplified reversal
                });
            }
        }

        // Copy the local best point back to the global memory
        for (int dim = 0; dim < hs.dimension; ++dim) {
            bestSolutions(i, dim) = localBestPoint(dim);
        }
    });

    return bestSolutions;
}

struct MinObjectiveData {
    double& minObjective;
    int& minPointIdx;

    MinObjectiveData(double& objective, int& pointIdx)
        : minObjective(objective), minPointIdx(pointIdx) {}
};


// Define a tolerance value
const double epsilon = 1000;

// Compare two floating-point numbers with tolerance
bool isEqual(double a, double b) {
    return std::abs(a - b) < epsilon;
}


int main(int argc, char* argv[]) {
    MPI_Init (& argc ,& argv );

    int num_procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    // Split communicator by shared memory type
    MPI_Comm node_comm;
    int split_type = MPI_COMM_TYPE_SHARED;
    MPI_Comm_split_type(MPI_COMM_WORLD, split_type, rank, MPI_INFO_NULL, &node_comm);
    int num_procs_intra, rank_intra;
    // Get the number of processes in the node communicator
    MPI_Comm_size(node_comm, &num_procs_intra);
    MPI_Comm_rank(node_comm, &rank_intra);


    Kokkos::initialize(argc, argv);
    {   auto start = std::chrono::high_resolution_clock::now();
        // Define parameters for the search
        const int dimension = 10000;
        const double innerRadius = 2.0;
        const int beta = 5;
        const int maxDepth = 5;
        const int numSamples = 10000; // Number of points sampled in each final leaf for exploitation
        std::vector<Kokkos::View<double**>> finalSolutions;

        // Initialize the search space
        Kokkos::View<double*> lowerBounds("lowerBounds", dimension), upperBounds("upperBounds", dimension);
        Kokkos::deep_copy(lowerBounds, -10.0); // Example bounds
        Kokkos::deep_copy(upperBounds, 10.0);
        SearchSpaceBounds searchSpace(dimension, lowerBounds, upperBounds);

        // Seed for random number generation
        uint64_t seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
        Kokkos::Random_XorShift64_Pool<> randomPool(seed);

        // True best for scoring
        Kokkos::View<double*> trueBest("trueBest", dimension);
        Kokkos::deep_copy(trueBest, 0);

        // Generate initial set of hyperspheres
        const int nHyperspheres = 5; // Determine the number of hyperspheres you need
        Kokkos::View<Hypersphere*> hyperspheres("hyperspheres", nHyperspheres);
        Kokkos::View<double**, Kokkos::LayoutRight> allCenters("allCenters", nHyperspheres, dimension);
        std::cout << "i am here " << std::endl;

        // Create and execute the functor to initialize hyperspheres
        CreateHyperspheresFunctor functor(hyperspheres, allCenters, searchSpace, 7);
        Kokkos::parallel_for("CreateHyperspheres", nHyperspheres, functor);
        std::cout << "first hypersphers " << std::endl;

        // Create a mirror view for host access
        auto hyperspheresHostMirror = Kokkos::create_mirror_view(hyperspheres);

        // Deep copy the data to the host
        Kokkos::deep_copy(hyperspheresHostMirror, hyperspheres);

        // Convert Kokkos::View to std::vector for easier manipulation
        std::vector<Hypersphere> currentHyperspheres(nHyperspheres);
        for (int i = 0; i < nHyperspheres; ++i) {
            currentHyperspheres[i] = hyperspheresHostMirror(i);
        }

        std::cout << "about to the enter the loop" << std::endl;







        // Main iterative process
        for (int depth = 0; depth <= maxDepth - 1; ++depth) {
            // Score current set of hyperspheres
            std::cout << "i am before scoring the hyperspheres" << std::endl;
            std::vector<HypersphereScore> scores;
            for (auto& hs : currentHyperspheres) {
                double score = scoreHypersphere(hs, trueBest, numSamples, randomPool);
                std::cout << "scored the curent hypersphere" << std::endl;
                scores.emplace_back(hs, score);
            }
            std::cout << "i am after scoring them" << std::endl;

            // Select the best hyperspheres based on scores
            auto selectedHyperspheres = selectBestHyperspheres(scores, beta);

            std::cout << "i selected the best hyperspheres now about to decide to explore or exploit" << std::endl;

            // Check for maximum depth
            if (depth == maxDepth - 1) {
                // Exploit selected hyperspheres
                // Get the number of available GPUs

                // Calculate number of samples per process
                int samples_per_process = numSamples / num_procs;

                // Calculate the remaining samples
                int remaining_samples = numSamples % num_procs;

                if (rank < remaining_samples) {
                    samples_per_process++;
                }

                for (auto& hs : selectedHyperspheres) {
                    // Generate initial points within the hypersphere
                    auto initialPoints = generateRandomPointsInHypersphere(hs, samples_per_process, randomPool); // Adjust numSamples if needed
                    // Apply the local search algorithm to find local optimal
                    auto solutions = localSearchMultipleStarts(hs, initialPoints, 1, 2); // Fill in maxIterations and stepSize as appropriate
                    finalSolutions.push_back(solutions);
                }
            } else {
                // Explore selected hyperspheres
                std::cout << "i am in the exploration" << std::endl;
                std::vector<Hypersphere> nextHyperspheres;
                for (auto& hs : selectedHyperspheres) {
                    // Use promisingHypersphereSearch to find promising regions within the hypersphere
                    std::cout << "i am before  the promiscing hypersphere search" << std::endl;
                    auto promisingRegions = promisingHypersphereSearch(hs, 5, 5, innerRadius , randomPool);
                    std::cout << "i am after the promiscing hypersphere search" << std::endl; // Fill in numCandidates and numBest as appropriate
                    // The result of promisingHypersphereSearch gives us the next hyperspheres to explore
                    nextHyperspheres.insert(nextHyperspheres.end(), promisingRegions.begin(), promisingRegions.end());
                }
                currentHyperspheres = std::move(nextHyperspheres);

            }
            std::cout << "Completed depth: " << depth << std::endl;
        }

        // Compute the dimensions
        const size_t numRowsPerView = finalSolutions[0].extent(0);
        const size_t numCols = finalSolutions[0].extent(1);
        const size_t numViews = finalSolutions.size();
        const size_t totalRows = numRowsPerView * numViews;

        // Create a new view for concatenated data in CUDA space
        Kokkos::View<double**,  Kokkos::HostSpace> concatenatedViewHost("concatenatedViewHost", totalRows, numCols);




        // Copy data from finalSolutions to concatenatedView
        for (size_t i = 0; i < numViews; ++i) {
            auto currentView = Kokkos::create_mirror_view(finalSolutions[i]);
            deep_copy(currentView, finalSolutions[i]);


            // Copy data from current view to concatenatedViewHost
            for (size_t j = 0; j < numRowsPerView; ++j) {
                for (size_t k = 0; k < numCols; ++k) {
                    concatenatedViewHost(i * numRowsPerView + j, k) = currentView(j, k);
                }
            }
        }


        auto concatenatedView = create_mirror_view_and_copy(Kokkos::CudaSpace(), concatenatedViewHost);

        double minObjective = std::numeric_limits<double>::max();
        int minPointIdx = -1;
        MinObjectiveData data{minObjective, minPointIdx};

        // Create a view to store coordinates of the point with min objective
        Kokkos::View<double*, Kokkos::HostSpace> minPointCoords("minPointCoords", numCols);


        Kokkos::parallel_scan("MinObjectiveScan", Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, totalRows-1),
            KOKKOS_LAMBDA(int i, double &update, const bool final) {
                // Access data from the mirrored view
                double objective1 = objectiveFunction(Kokkos::subview(concatenatedViewHost, i, Kokkos::ALL()));
                double objective2 = objectiveFunction(Kokkos::subview(concatenatedViewHost, i + 1, Kokkos::ALL()));
                double minObjectivePair = objective1 < objective2 ? objective1 : objective2;

                if (minObjectivePair < update) {
                    update = minObjectivePair;
                }

                if (final && update < data.minObjective) {
                    data.minObjective = update;
                    data.minPointIdx = (objective1 < objective2) ? i : (i + 1);

                    // Store coordinates of the point with min objective
                    for (size_t k = 0; k < numCols; ++k) {
                        minPointCoords(k) = concatenatedViewHost(data.minPointIdx, k);
                    }
                }
            }, minObjective);

    minObjective = data.minObjective;
    minPointIdx = data.minPointIdx;
    double minObjective1 = data.minObjective ;


    // Synchronize minObjective across all intra MPI processes
    MPI_Allreduce(MPI_IN_PLACE, &minObjective, 1, MPI_DOUBLE, MPI_MIN, node_comm);

    // Create a new communicator containing all processes satisfying the condition
    MPI_Comm new_comm;
    MPI_Comm_split(MPI_COMM_WORLD, (minObjective1 == minObjective) ? 1 : MPI_UNDEFINED, rank, &new_comm);


    if (minObjective1 == minObjective) {

        // Create a new communicator containing all processes satisfying the condition


        // Example usage of the new communicator
        int new_rank, new_size;
        MPI_Comm_rank(new_comm, &new_rank);
        MPI_Comm_size(new_comm, &new_size);


        double minObjective2 = data.minObjective ;

        MPI_Allreduce(MPI_IN_PLACE, &minObjective1, 1, MPI_DOUBLE, MPI_MIN, new_comm);


        if(minObjective2==minObjective1){
            printf("Minimum objective value: %e\n", minObjective);

            for (size_t k = 0; k < numCols; ++k) {
                printf("%lf ", minPointCoords(k));
            }
            // Calculate duration
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;

        // Output the execution time
        std::cout << "Execution time: " << duration.count() << " seconds" << std::endl ;
        }
      }
     // Calculate duration
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;


    }

    Kokkos::finalize();
    MPI_Finalize ();
    return 0;
    }