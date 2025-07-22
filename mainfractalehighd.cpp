#include <mpi.h>
#include <iostream>
#include <chrono>
#include <limits>
#include <numeric>
#include <vector>
#include <algorithm>

#include <Kokkos_Core.hpp>

#include "HPCFunctions.hpp"
#include "GPfunctions.hpp"

struct MinAcqIndex {
    double value;
    int index;

    KOKKOS_INLINE_FUNCTION
    MinAcqIndex() : value(DBL_MAX), index(-1) {}

    KOKKOS_INLINE_FUNCTION
    MinAcqIndex(const double val, const int idx) : value(val), index(idx) {}

    // Comparison operator needed by Kokkos
    KOKKOS_INLINE_FUNCTION
    bool operator<(const MinAcqIndex& rhs) const {
        return value < rhs.value;
    }

    // Join operation for reduction
    KOKKOS_INLINE_FUNCTION
    void operator+=(const MinAcqIndex& src) {
        if (src.value < value) {
            value = src.value;
            index = src.index;
        }
    }
};

namespace Kokkos {
template <>
struct reduction_identity<MinAcqIndex> {
    KOKKOS_FORCEINLINE_FUNCTION static MinAcqIndex max() {
        return MinAcqIndex(-DBL_MAX, -1);
    }

    KOKKOS_FORCEINLINE_FUNCTION static MinAcqIndex min() {
        return MinAcqIndex(DBL_MAX, -1);
    }
};
} // namespace Kokkos


void buildPcaMatrix(Kokkos::View<double**, Kokkos::LayoutRight,
                                 Kokkos::HostSpace>& hostPca,
                    int reducedDim, int origDim)
{
    for (int rd = 0; rd < reducedDim; ++rd)
        for (int j = 0; j < origDim; ++j)
            hostPca(rd,j) = (rd == j ? 1.0 : 0.0);
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
    int budget = parseBudget(argc, argv, world_rank);
    int expenseFactor  = parseExpenseFactor(argc, argv, world_rank);
    const ObjectiveType objTy = parseObjectiveType (argc, argv, /*idx=*/5, world_rank);

    // Initialize Kokkos
    Kokkos::initialize(argc, argv);
    {
        auto start = std::chrono::high_resolution_clock::now();


        using ExecSpace = Kokkos::DefaultExecutionSpace;
        using MemSpace  = ExecSpace::memory_space;


        // Define parameters for the search
        const int maxDepth = 5;
        // Parameters for applying the intensive local search starting from the center
        int maxIterations = 100;   // Number of iterations for the search
        double stepSize = 1.75;    // Initial step size (ω)
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



        // --- Initial Training with Embedding ---
        const int numGPPoints = 50;
        const int reducedDimension = std::min(10, dimension); // reduce dimension to 10 or lower

        // ---------- host-side PCA basis -----------------------------------
        Kokkos::View<double**, Kokkos::LayoutRight, Kokkos::HostSpace>
            h_pcaMat("h_pcaMat", reducedDimension, dimension);

        buildPcaMatrix(h_pcaMat, reducedDimension, dimension);   // <-- your routine

        // ---------- device copy (ordinary local, NOT static !) ------------
        Kokkos::View<double**, Kokkos::LayoutRight, MemSpace>
            d_pcaMat("d_pcaMat", reducedDimension, dimension);

        Kokkos::deep_copy(d_pcaMat, h_pcaMat);

        // Latin Hypercube Sampling in original space
        std::vector<double> originalTrainingPoints = GP::latinHypercubeSample(
            numGPPoints, dimension, lowerBounds, upperBounds, 123);

        // Embed training points using PCA
        Kokkos::Profiling::pushRegion("GP_PCA_embed");

        Kokkos::View<double*, Kokkos::HostSpace> h_trainFlat("h_trainFlat", numGPPoints * dimension);
for (size_t i = 0; i < originalTrainingPoints.size(); ++i)
    h_trainFlat(i) = originalTrainingPoints[i];

Kokkos::View<double*, MemSpace> d_trainFlat("d_trainFlat", numGPPoints * dimension);
Kokkos::deep_copy(d_trainFlat, h_trainFlat);
        //------------------------------------------------------------------------
// (A) copy the 50 original points to a device view
//------------------------------------------------------------------------
Kokkos::View<double**, MemSpace> d_train("d_train",
                                         numGPPoints, dimension);


Kokkos::parallel_for("copyTrain",
    Kokkos::RangePolicy<ExecSpace>(0, numGPPoints * dimension),
    KOKKOS_LAMBDA(const int idx) {
        const int i = idx / dimension;
        const int j = idx % dimension;
        d_train(i, j) = d_trainFlat(idx);
    });

//------------------------------------------------------------------------
// (B) allocate output [numGPPoints × reducedDimension]
//------------------------------------------------------------------------
Kokkos::View<double**, MemSpace> d_trainEmb(
        "d_trainEmb", numGPPoints, reducedDimension);

//------------------------------------------------------------------------
// (C) same projection kernel we used for candidates
//------------------------------------------------------------------------
using team_policy = Kokkos::TeamPolicy<ExecSpace>;
using member_type = team_policy::member_type;

Kokkos::parallel_for("PCA_project_train",
     team_policy(numGPPoints, Kokkos::AUTO),
     KOKKOS_LAMBDA(const member_type& team) {

    const int i = team.league_rank();                 // training index

    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, reducedDimension),
        [&](const int rd) {
            double dot = 0.0;
            for (int j = 0; j < dimension; ++j)
                dot += d_train(i, j) * d_pcaMat(rd, j);
            d_trainEmb(i, rd) = dot;
        });
});

//------------------------------------------------------------------------
// (D) bring the result back to a std::vector that the CPU-GP code expects
//------------------------------------------------------------------------
auto h_trainEmb =
    Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_trainEmb);

std::vector<double> embeddedTrainingPoints(numGPPoints * reducedDimension);
for (int i = 0; i < numGPPoints; ++i)
    for (int rd = 0; rd < reducedDimension; ++rd)
        embeddedTrainingPoints[i * reducedDimension + rd] = h_trainEmb(i, rd);

        
        Kokkos::Profiling::popRegion();

        // Train GP model in embedded space
        const double sigma_f = 1.0;
        const double l = 1.0;
        const double sigma_n = 1e-3;

        
        


        GP::Model gpModel = GP::trainGP(
            embeddedTrainingPoints, numGPPoints, reducedDimension, sigma_f, l, sigma_n, expenseFactor, objTy );


        auto d_gpModel = GP::makeDeviceCopy(gpModel, sigma_f, l, sigma_n);
        
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
        double initialRadius = 5.12; // bounds are [-5.12, 5.12], so radius = 5.12

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

            // Get available GPU memory (if CUDA)
            size_t available_memory = get_available_memory();

            // Estimate a naive "per-hypersphere" usage
            size_t per_hypersphere_memory = (dimension + 2) * sizeof(double);

            per_hypersphere_memory =
    // centers_chunk
    dimension * sizeof(double)
  + // radii_chunk
    sizeof(double)
  + // candidatePointsOriginal + d_candidates
    2 * (dimension * sizeof(double))
  + // embedding
    reducedDimension * sizeof(double)
  + // GP buffers: mu, var, acq
    3 * sizeof(double)
  + // used flags
    sizeof(bool);

            // Calculate max number of hyperspheres per chunk
            size_t max_num_hyperspheres = static_cast<size_t>(available_memory * 0.1) / per_hypersphere_memory;
            if (max_num_hyperspheres == 0) {
                max_num_hyperspheres = 1;
            }

            // We'll collect local (scores, hyperspheres) chunk by chunk
            std::vector<double> localScoresOnly;
            std::vector<Hypersphere> localHyperspheres;

            size_t totalHyperspheres = numLocalHyperspheres;

            /*---------------------------------------------------------*/
            /* 1)  Compteur global : points qu'il reste à évaluer     */
            /*---------------------------------------------------------*/
            int remaining = budget;          // ← quota à consommer

            for (size_t chunk_start = 0;
                 chunk_start < totalHyperspheres && remaining > 0;
                 chunk_start += max_num_hyperspheres) {
                size_t chunk_size = std::min(max_num_hyperspheres, totalHyperspheres - chunk_start);

                std::vector<Hypersphere> chunkHyperspheres(
                    myHyperspheres.begin() + chunk_start,
                    myHyperspheres.begin() + chunk_start + chunk_size);

                // Prepare hypersphere data
                Kokkos::View<int*, mem_space> depths_chunk;
                Kokkos::View<size_t*, mem_space> idOffsets_chunk;
                Kokkos::View<int*, mem_space> idDims_chunk;
                Kokkos::View<int*, mem_space> idSigns_chunk;

                prepareHypersphereData(chunkHyperspheres,
                                       depths_chunk,
                                       idOffsets_chunk,
                                       idDims_chunk,
                                       idSigns_chunk);

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














































































                    {

                        std::cout << "I entered the prescoring block" << std::endl;
                         
                    
                        // Copy candidate centers to host
                        auto candHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), centers_chunk);
                        int numCandidates = candHost.extent(0);


                        /*-----------------------------------------------*/
                        /* 2)  Budget pour CE chunk                      */
                        /*-----------------------------------------------*/
                        int B = std::min(remaining, numCandidates);

                        // -------- BuildCandidates : copy + GPU projection -------------------
// Étape 1: copie vers un View host
Kokkos::View<double*, Kokkos::HostSpace> h_candidatePointsOriginal("h_candidatePointsOriginal", numCandidates * dimension);
for (int i = 0; i < numCandidates; ++i)
    for (int d = 0; d < dimension; ++d)
        h_candidatePointsOriginal(i * dimension + d) = candHost(i, d);

// Étape 2: transfert vers device
Kokkos::View<double*, MemSpace> d_candidatePointsOriginal("d_candidatePointsOriginal", numCandidates * dimension);
Kokkos::deep_copy(d_candidatePointsOriginal, h_candidatePointsOriginal);

// Étape 3: remplissage du tableau 2D
Kokkos::View<double**, MemSpace> d_candidates("d_candidates", numCandidates, dimension);
Kokkos::parallel_for("copyCandidates",
    Kokkos::RangePolicy<ExecSpace>(0, numCandidates * dimension),
    KOKKOS_LAMBDA(const int idx){
        const int i = idx / dimension;
        const int j = idx % dimension;
        d_candidates(i, j) = d_candidatePointsOriginal(idx);
    });


/* (2) allocate output in reduced space ----------------------------- */
Kokkos::View<double**, MemSpace> d_embedded(
        "d_embedded", numCandidates, reducedDimension);

/* (3) project each centre with the cached PCA basis ---------------- */
using team_policy = Kokkos::TeamPolicy<ExecSpace>;
using member_type = team_policy::member_type;

Kokkos::parallel_for("PCA_project",
     team_policy(numCandidates, Kokkos::AUTO),
     KOKKOS_LAMBDA(const member_type& team)
{
    const int i = team.league_rank();   // candidate index
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(team, reducedDimension),
        [&](const int rd){
            double dot = 0.0;
            for (int j = 0; j < dimension; ++j)
                dot += d_candidates(i,j) * d_pcaMat(rd,j);
            d_embedded(i, rd) = dot;
        });
});

/* (4) mirror back to host because the CPU GP code expects a vector -- */
auto h_embedded = Kokkos::create_mirror_view_and_copy(
                      Kokkos::HostSpace(), d_embedded);
std::vector<double> candidatePointsEmbedded(numCandidates * reducedDimension);
for (int i = 0; i < numCandidates; ++i)
    for (int rd = 0; rd < reducedDimension; ++rd)
        candidatePointsEmbedded[i*reducedDimension + rd] = h_embedded(i,rd);


                        
                        // Add verification:
                        for (double val : candidatePointsEmbedded) {
                            if (!std::isfinite(val)) {
                                std::cerr << "[ERROR] PCA embedding produced invalid (NaN or Inf) values." << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                            }
                        }
                    
                        // Predict in embedded space
                        std::vector<double> h_mu(numCandidates), h_var(numCandidates);
                        // Initialize device views
                        Kokkos::View<double*> d_mu("d_mu", numCandidates);
                        Kokkos::View<double*> d_var("d_var", numCandidates);
                        Kokkos::View<double*> d_acq("d_acq", numCandidates);
                        Kokkos::View<bool*>   d_used("d_used", numCandidates);
                    
                        auto h_muView     = Kokkos::create_mirror_view(d_mu);
                        auto h_varView    = Kokkos::create_mirror_view(d_var);
                        auto h_acqView    = Kokkos::create_mirror_view(d_acq);
                        auto h_usedView   = Kokkos::create_mirror_view(d_used);

                        /* -------------------  add region wrapper  ------------------- */
                        Kokkos::Profiling::pushRegion("GP_predict");            // <-- push here
                        /* ------------------------------------------------------------ */
                        /* ------------- device views already exist ------------------ */


/* candidates are in d_embedded (numCandidates × reducedDimension) */
GP::batchPredictGP(d_gpModel, d_embedded, d_mu, d_var);   // <<< GPU >>>



for (int i = 0; i < numCandidates; ++i) {
    h_mu[i]  = h_muView(i);
    h_var[i] = h_varView(i);
}
                        /* -------------------  end of region  ------------------------ */
                        Kokkos::Profiling::popRegion();                         // <-- pop here
                        /* ------------------------------------------------------------ */
                        
                    
                        double kappa = 2.0;
                        for(int i = 0; i < numCandidates; i++){
                            h_muView(i)   = h_mu[i];
                            h_varView(i)  = h_var[i];
                            double sig    = std::sqrt(h_var[i]);
                            h_acqView(i)  = h_mu[i] - kappa * sig;  // LCB
                            h_usedView(i) = false;
                        }
                    
                        Kokkos::deep_copy(d_mu,   h_muView);
                        Kokkos::deep_copy(d_var,  h_varView);
                        Kokkos::deep_copy(d_acq,  h_acqView);
                        Kokkos::deep_copy(d_used, h_usedView);
                    
                        std::vector<int> chosenIndices;
                        chosenIndices.reserve(B);

                        std::cout << "pick loop" << std::endl;
                    
                        for (int pick = 0; pick < B; pick++) {
                            MinAcqIndex result;
                            Kokkos::parallel_reduce(
                                "FindMinLCBValueAndIndex",
                                Kokkos::RangePolicy<ExecSpace>(0, numCandidates),
                                KOKKOS_LAMBDA(const int i, MinAcqIndex& local_min) {
                                    if (!d_used(i) && d_acq(i) < local_min.value) {
                                        local_min.value = d_acq(i);
                                        local_min.index = i;
                                    }
                                },
                                Kokkos::Min<MinAcqIndex>(result)
                            );
                    
                            Kokkos::fence();
                            int bestIdxHost = result.index;
                            
                            
                    
                            Kokkos::parallel_for("MarkUsed", Kokkos::RangePolicy<ExecSpace>(bestIdxHost, bestIdxHost + 1),
                                                 KOKKOS_LAMBDA(const int i) { d_used(i) = true; });
                            Kokkos::fence();

                            if (result.index < 0 || result.index >= numCandidates) {
                                std::cerr << "[ERROR] GPU selection produced invalid index: " << result.index << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                            }
                    
                            chosenIndices.push_back(bestIdxHost);
                        }
                    
                        int actualB = chosenIndices.size();

                        /* --------------------------------------------------------------------
                * TOP-UP THE BEAM IF   actualB < beta
                * ------------------------------------------------------------------*/
                int needExtra = beta - actualB;          // how many we still need
                if (needExtra > 0)
                {
                    /* 1.  Host copies that we can edit -------------------------------- */
                    auto h_acqHost  = Kokkos::create_mirror_view_and_copy(
                                        Kokkos::HostSpace(), d_acq);
                    auto h_usedHost = Kokkos::create_mirror_view_and_copy(
                                        Kokkos::HostSpace(), d_used);

                    /* 2.  Rank all candidates by LCB (lowest first) ------------------- */
                    std::vector<int> ranked(numCandidates);
                    std::iota(ranked.begin(), ranked.end(), 0);
                    std::sort(ranked.begin(), ranked.end(),
                            [&](int a,int b){ return h_acqHost(a) < h_acqHost(b); });

                    /* 3.  Pick 'needExtra' best still-unused indices ------------------ */
                    for (int k = 0; k < numCandidates && needExtra > 0; ++k)
                    {
                        int idx = ranked[k];
                        if (!h_usedHost(idx))            // still unused?
                        {
                            chosenIndices.push_back(idx);
                            h_usedHost(idx) = true;      // mark so we never take it twice
                            --needExtra;
                        }
                    }

                    /* 4.  Copy the updated “used’’ flags back to the device ----------- */
                    Kokkos::deep_copy(d_used, h_usedHost);
                    actualB = static_cast<int>(chosenIndices.size());
                    
                }
                /* ------------------------------------------------------------------ */

                    
                        Kokkos::View<double**, mem_space> centers_topB("centers_topB", actualB, dimension);
                        Kokkos::View<double*,  mem_space> radii_topB("radii_topB", actualB);
                    
                        auto h_centersB = Kokkos::create_mirror_view(centers_topB);
                        auto h_radiiB   = Kokkos::create_mirror_view(radii_topB);

                        // After creating centers_chunk:
                        std::cout << "chunkHyperspheres.size(): " << chunkHyperspheres.size() 
                        << ", numCandidates: " << numCandidates << std::endl;
                    
                        for(int i = 0; i < actualB; i++){
                            int cidx = chosenIndices[i];

                            // Robust boundary check
                            if (cidx < 0 || cidx >= chunkHyperspheres.size()) {
                                std::cerr << "[ERROR] cidx out-of-bounds: " << cidx 
                                        << ", chunkHyperspheres.size() = " << chunkHyperspheres.size() << std::endl;
                                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                            }
                            auto & hs = chunkHyperspheres[cidx]; 

                           
                    
                            auto centerView = reconstructCenter(hs, initialCenter, initialRadius);
                            auto centerHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), centerView);
                    
                            for(int d = 0; d < dimension; d++)
                                h_centersB(i,d) = centerHost(d);
                    
                            double r = initialRadius;
                            for(int dd = 0; dd < hs.depth; dd++)
                                r = r / (1.0 + std::sqrt(2.0));
                            h_radiiB(i) = r;
                        }
                    
                        Kokkos::deep_copy(centers_topB, h_centersB);
                        Kokkos::deep_copy(radii_topB,   h_radiiB);
                    
                        Kokkos::View<double*, mem_space> topScores("topScores", actualB);
                        scoreHyperspheres(
                                        actualB,
                                        dimension,
                                        centers_topB,
                                        radii_topB,
                                        bestSolution,
                                        bestObjectiveValue,
                                        topScores,
                                        expenseFactor,
                                        objTy  // <— bien qualifié
                                    );
                    
                        auto hostScores = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), topScores);


                        int preScored  = B;
                        int extraPicks = actualB - preScored;

                        static long long globalPts = 0;
                        globalPts += 3LL * actualB;              // 3 sample-points per centre

                        if (world_rank == 0) {
                                        std::cout << "[DBG] depth="       << depth
                          << "  candidates="      << numCandidates
                          << "  picked="          << actualB
                          << "  (scored="     << preScored
                          << ", pre-scored="          << extraPicks << ")"
                          << "  scoredPoints="    << 3*actualB
                          << "  cumPoints="       << globalPts
                          << '\n';
             }

                        // IMMEDIATELY ADD THIS BLOCK:
                        for(int i = 0; i < actualB; i++){
                            localScoresOnly.push_back(hostScores(i));
                            localHyperspheres.push_back(chunkHyperspheres[chosenIndices[i]]);
                        }

                        /*-----------------------------------------------*/
                        /* 3)  Décrémente le compteur et stop si quota OK*/
                        /*-----------------------------------------------*/
                        remaining -= actualB;          // actualB == B
                        if (remaining == 0) {
                            break;   // ← quota atteint ➜ sortir de la boucle chunk
                        }
                    
                        std::vector<double> newTrainingPoints;
                        newTrainingPoints.reserve(actualB * reducedDimension);
                    
                        for(int i = 0; i < actualB; i++){
                            int cidx = chosenIndices[i];
                            for (int d = 0; d < reducedDimension; d++)
                                newTrainingPoints.push_back(candidatePointsEmbedded[cidx*reducedDimension + d]);
                        }
                        Kokkos::Profiling::pushRegion("GP_update");


                         // right before:  gpModel = GP::updateGP(…)
                        embeddedTrainingPoints.insert(embeddedTrainingPoints.end(),
                                                    newTrainingPoints.begin(),
                                                    newTrainingPoints.end());

                        gpModel = GP::updateGP(gpModel, newTrainingPoints, actualB, sigma_f, l, sigma_n, expenseFactor, objTy );
                        d_gpModel = GP::makeDeviceCopy(gpModel, sigma_f, l, sigma_n);


                        // ─── problem-specific log-grid search over (σƒ, ℓ, σₙ) ─────────────────────
double initRadius = 5.12;   // half the search-space width
std::vector<double> l_grid  = {
    initRadius / 1e2,    // very short lengthscale
    initRadius / 1e1,    // short
    initRadius,          // medium
    initRadius * 1e1     // long
};
std::vector<double> sf_grid = { 0.1, 1.0, 10.0 };      // small/med/large signal var
std::vector<double> sn_grid = { 1e-6, 1e-3, 1e-1 };    // low/med/high noise var

double bestNLL = std::numeric_limits<double>::infinity();
double best_l  = l;         // your current defaults
double best_sf = sigma_f;
double best_sn = sigma_n;

for (double sn : sn_grid) {
  for (double sf : sf_grid) {
    for (double lg : l_grid) {
      double nll = GP::negLogMarginalLikelihood(
                       gpModel, sf, lg, sn);
      if (nll < bestNLL) {
        bestNLL = nll;
        best_sf = sf;
        best_l  = lg;
        best_sn = sn;
      }
    }
  }
}

// retrain on N≈50 points (still cheap)
gpModel = GP::trainGP(
    embeddedTrainingPoints,
    gpModel.N,
    reducedDimension,
    best_sf, best_l, best_sn,
    expenseFactor,
    objTy);

d_gpModel = GP::makeDeviceCopy(
    gpModel, best_sf, best_l, best_sn);

// update exploration coefficient per Šrivas et al.
double beta_t = 2.0 * std::log(
    gpModel.N * gpModel.N * M_PI * M_PI / 6.0);
kappa = std::sqrt(beta_t);
// ─────────────────────────────────────────────────────────────────────────





                        Kokkos::Profiling::popRegion();
                    }
                    

























































































                }


            
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

            // Now gather top 'beta' hyperspheres within the node
            int localDataSize = localBeta;
            std::vector<int> allDataSizes(node_size);
            MPI_Allgather(&localDataSize, 1, MPI_INT, allDataSizes.data(), 1, MPI_INT, node_comm);

            int totalDataSize = std::accumulate(allDataSizes.begin(), allDataSizes.end(), 0);

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
                    if (idx < 0 || idx >= static_cast<int>(nodeAllIDs.size())) {
                        std::cerr << "[Error] idx out-of-bounds. idx=" << idx 
                                  << ", nodeAllIDs.size()=" << nodeAllIDs.size() << std::endl;
                        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                    }
                    HypersphereID selectedID = nodeAllIDs[idx];
                    selectedHyperspheres.emplace_back(dimension, depth, selectedID);
                }
                
                if (selectedHyperspheres.empty()) {
                    std::cerr << "[Error] selectedHyperspheres is empty after filling loop." << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
                }
                
                Hypersphere bestHypersphere = selectedHyperspheres[0]; // safe now

                // Reconstruct the center and radius of the best hypersphere
                auto bestCenter = reconstructCenter(bestHypersphere, initialCenter, initialRadius);

                // Generate points around the center
                auto points = generateRandomPointsAroundCenter(bestCenter, initialRadius, dimension);

                // Evaluate objective function at each point
                auto pointsHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), points);

                double localBestObjective = std::numeric_limits<double>::max();
                Kokkos::View<double*, Kokkos::LayoutLeft, Kokkos::HostSpace> localBestSolution("localBestSolution", dimension);

                for (int i = 0; i < pointsHost.extent(0); ++i) {
                    auto x = Kokkos::subview(pointsHost, i, Kokkos::ALL());
                    double fx = computeObjective(x, expenseFactor, objTy);
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

                std::vector<Hypersphere> myHyperspheres;
                for (int i = localStartIdx; i < localEndIdx; ++i) {
                    myHyperspheres.push_back(currentHyperspheres[i]);
                }

                double localBestObjective = std::numeric_limits<double>::max();
                Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> localBestPoint("localBestPoint", dimension);
                Kokkos::deep_copy(localBestPoint, 0.0);

                // Process all assigned hyperspheres
                for (auto& hs : myHyperspheres) {
                    // Perform intensive local search
                    auto solution = intensiveLocalSearch(hs,
                                     initialCenter,
                                     initialRadius,
                                     maxIterations,
                                     stepSize,
                                     phi,
                                     omega_min,
                                     objTy,          // <- new
                                     expenseFactor); // <- new

                    // Compute the objective value on device
                    double objValue = computeObjectiveValue(solution, expenseFactor, objTy);

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

// Extract the best objective
double nodeBestObjective = nodeBestMin.value;

// Gather the best point across all ranks in this node
Kokkos::View<double*, Kokkos::HostSpace> nodeBestPointHost("nodeBestPointHost", dimension);
if (node_rank == nodeBestMin.rank) {
  // Only the rank that held the best point copies it into host view
  auto localPointHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), localBestPoint);
  for (int d = 0; d < dimension; ++d) {
    nodeBestPointHost(d) = localPointHost(d);
  }
}
// Broadcast from the winning rank to everyone in node_comm
MPI_Bcast(nodeBestPointHost.data(), dimension, MPI_DOUBLE, nodeBestMin.rank, node_comm);

// Copy it back to device
Kokkos::View<double*, mem_space> nodeBestPoint("nodeBestPoint", dimension);
Kokkos::deep_copy(nodeBestPoint, nodeBestPointHost);


                // Node leaders do an inter-node reduction
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

                // Broadcast globalMin to all processes on the node
                MPI_Bcast(&globalMin, 1, MPI_DOUBLE_INT, 0, node_comm);

                // Now node leader with global best broadcasts its best point to the other node leaders
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

                // Then the node leader broadcasts that best point to all local ranks
                MPI_Bcast(globalBestPointHost.data(), dimension, MPI_DOUBLE, 0, node_comm);

                // Put that in a device view
                Kokkos::View<double*, Kokkos::LayoutLeft, mem_space> globalBestPoint("globalBestPoint", dimension);
                Kokkos::deep_copy(globalBestPoint, globalBestPointHost);

                // Only the root process (rank 0) prints the final result
                if (world_rank == 0) {
                    std::cout << "Minimum objective value: " << globalMin.value << std::endl;

                    // Sphere function global minimum is 0.0
                    double sphereDifference = globalMin.value - 0.0;
                    std::cout << "Difference from Global Minimum: " << sphereDifference << std::endl;

                    // Compute Euclidean distance to the known optimum (all zeros)
                    double dist2 = 0.0;
                    for (int k = 0; k < dimension; ++k) {
                        double xk = globalBestPointHost(k);
                        dist2 += xk * xk;
                    }
                    double euclidDist = std::sqrt(dist2);
                    std::cout << "Euclidean distance to optimum: " << euclidDist << std::endl;
                    

                    // Output summary stats about solution
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

                    // Print the entire final solution
                    std::cout << "Entire solution found: [";
                    for (size_t k = 0; k < globalBestPointHost.extent(0); ++k) {
                        std::cout << globalBestPointHost(k)
                                  << ((k < globalBestPointHost.extent(0) - 1) ? ", " : "");
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
                    nextHyperspheres.insert(nextHyperspheres.end(),
                                            childHyperspheres.begin(),
                                            childHyperspheres.end());
                }
                currentHyperspheres = std::move(nextHyperspheres);
            }

            if (world_rank == 0) {
                std::cout << "Completed depth: " << depth << std::endl;
            }
        }

        

        
    

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    if (world_rank == 0) {
            std::cout << "Total Execution time: " << duration.count() << " seconds" << std::endl;
        }

}

Kokkos::finalize();
    MPI_Finalize();
    return 0;}