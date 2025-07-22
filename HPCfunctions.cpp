#include "HPCFunctions.hpp"
#include <iostream>
#include <cmath>        // std::cos, std::sin, std::sqrt
#include <limits>       // std::numeric_limits

// --------------------------------------------------
// 1) HPC and Data Structures
// --------------------------------------------------
// Define the inline function here (instead of just a declaration).




// À la fin de HPCFunctions.cpp

// Assurez-vous que computeRastriginTerm est déclaré juste avant.



// --------------------------------------------------
// 2) Serialization
// --------------------------------------------------
std::vector<int> serializeHypersphereID(const HypersphereID& id){
    std::vector<int> s;
    s.push_back((int)id.size());
    for(const auto& p : id){
        s.push_back(p.first);
        s.push_back(p.second);
    }
    return s;
}

HypersphereID deserializeHypersphereID(const std::vector<int>& data, size_t& offset){
    HypersphereID id;
    int idSize = data[offset++];
    for(int i=0; i< idSize; i++){
        int dim  = data[offset++];
        int sign = data[offset++];
        id.emplace_back(dim, sign);
    }
    return id;
}


// --------------------------------------------------
// 3) HPC Functions
// --------------------------------------------------
Kokkos::View<double*, mem_space>
reconstructCenter(const Hypersphere& hs,
                  const Kokkos::View<double*, mem_space>& initCenter,
                  double initRadius)
{
    int D     = hs.dimension;
    int depth = hs.depth;

    // We'll do a host mirror copy to do the arithmetic
    Kokkos::View<double*, Kokkos::HostSpace> hostC("hostC", D);
    auto initCenterHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), initCenter);
    Kokkos::deep_copy(hostC, initCenterHost);

    double r = initRadius;
    for(int d=0; d< depth; d++){
        double sqrt2 = std::sqrt(2.0);
        double rPrime = r / (1.0 + sqrt2);
        double offset = r - rPrime;
        int   dim  = hs.id[d].first;
        int   sign = hs.id[d].second;
        hostC(dim) += sign * offset;
        r = rPrime;
    }

    Kokkos::View<double*, mem_space> c("c", D);
    Kokkos::deep_copy(c, hostC);
    return c;
}

Kokkos::View<double**, mem_space>
generateRandomPointsAroundCenter(const Kokkos::View<double*, mem_space>& center,
                                 double radius, int dimension)
{
    const int numPoints = 3;
    Kokkos::View<double**, mem_space> points("points", numPoints, dimension);

    auto policy = Kokkos::TeamPolicy<>(numPoints, Kokkos::AUTO);

    // Fill
    Kokkos::parallel_for("GenRandPoints", policy,
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team){
        int i = team.league_rank();
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dimension),
          [&](int d){
            // Hard-coded for demonstration
            double val = 0.5;
            points(i,d) = val;
        });
        team.team_barrier();
    });

    // Normalize and scale
    Kokkos::parallel_for("NormalizePoints", policy,
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team){
        int i = team.league_rank();
        double norm=0.0;
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, dimension),
          [&](int d, double& loc){
            loc += points(i,d) * points(i,d);
          },
          norm
        );
        norm = std::sqrt(norm);

        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, dimension),
          [&](int d){
            double scaled = (points(i,d)/norm)* radius;
            points(i,d) = center(d) + scaled;
        });
        team.team_barrier();
    });

    return points;
}

void prepareHypersphereData(const std::vector<Hypersphere>& hList,
                            Kokkos::View<int*,    mem_space>& depths,
                            Kokkos::View<size_t*, mem_space>& idOffsets,
                            Kokkos::View<int*,    mem_space>& idDims,
                            Kokkos::View<int*,    mem_space>& idSigns)
{
    size_t n = hList.size();
    depths    = Kokkos::View<int*,    mem_space>("depths", n);
    idOffsets = Kokkos::View<size_t*, mem_space>("idOffsets", n+1);

    size_t totalIDSize = 0;
    for(const auto& hs : hList){
        totalIDSize += hs.id.size();
    }
    idDims  = Kokkos::View<int*, mem_space>("idDims",  totalIDSize);
    idSigns = Kokkos::View<int*, mem_space>("idSigns", totalIDSize);

    auto dHost   = Kokkos::create_mirror_view(depths);
    auto offHost = Kokkos::create_mirror_view(idOffsets);
    auto dimHost = Kokkos::create_mirror_view(idDims);
    auto signHost= Kokkos::create_mirror_view(idSigns);

    size_t currentOff = 0;
    for(size_t i=0; i< n; i++){
        dHost(i)   = hList[i].depth;
        offHost(i) = currentOff;
        for(const auto& p : hList[i].id){
            dimHost(currentOff)  = p.first;
            signHost(currentOff) = p.second;
            currentOff++;
        }
    }
    offHost(n) = currentOff;

    Kokkos::deep_copy(depths,    dHost);
    Kokkos::deep_copy(idOffsets, offHost);
    Kokkos::deep_copy(idDims,    dimHost);
    Kokkos::deep_copy(idSigns,   signHost);
}

void reconstructCentersAndRadii(size_t nHypers,
                                int dimension,
                                double initRadius,
                                const Kokkos::View<int*, mem_space>& depths,
                                const Kokkos::View<size_t*, mem_space>& idOffsets,
                                const Kokkos::View<int*, mem_space>& idDims,
                                const Kokkos::View<int*, mem_space>& idSigns,
                                const Kokkos::View<double*, mem_space>& initCenterDev,
                                Kokkos::View<double**, mem_space>& centers,
                                Kokkos::View<double*,  mem_space>& radii)
{
    centers = Kokkos::View<double**, mem_space>("centers", nHypers, dimension);
    radii   = Kokkos::View<double*,  mem_space>("radii",   nHypers);

    Kokkos::parallel_for("ReconstructCenters",
      Kokkos::RangePolicy<>(0, nHypers),
      KOKKOS_LAMBDA(int i){
        int depth   = depths(i);
        size_t st   = idOffsets(i);

        // Copy initCenter to row
        for(int d=0; d< dimension; d++){
            centers(i,d) = initCenterDev(d);
        }
        double r = initRadius;
        for(int dd=0; dd< depth; dd++){
            double sqrt2 = std::sqrt(2.0);
            double rP    = r/(1.0+ sqrt2);
            double off   = r- rP;
            int dim      = idDims(st + dd);
            int sign     = idSigns(st + dd);
            centers(i, dim) += sign* off;
            r = rP;
        }
        radii(i) = r;
    });
}

void scoreHyperspheres(size_t nHypers,
                       int dimension,
                       const Kokkos::View<double**, mem_space>& centers,
                       const Kokkos::View<double*,  mem_space>& radii,
                       const Kokkos::View<double*,  mem_space>& bestSolutionDev,
                       double bestObjVal,
                       Kokkos::View<double*, mem_space>& scores,
                       int expenseFactor,
                       ObjectiveType objType)
{
  Kokkos::deep_copy(scores, 0.0);
  const int numPoints = 3;

  using policy = Kokkos::TeamPolicy<>;
  Kokkos::parallel_for("ScoreHypers",
    policy(nHypers, Kokkos::AUTO),
    KOKKOS_LAMBDA(const policy::member_type& team){
      int i = team.league_rank();
      double totalScore = 0.0;
      double r = radii(i);

      for(int p = 0; p < numPoints; ++p){
        double fx   = 0.0;
        double dist = 0.0;

        // 1) on calcule la valeur de l'objectif en chaque dimension
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, dimension),
          [&](int d, double& loc){
            double xi  = centers(i,d) + 0.5 * r;
            double term = 0.0;

            // Switch générique
            if(objType == ObjectiveType::Sphere) {
              term = computeSphereTerm(xi, expenseFactor);
            }
            else if(objType == ObjectiveType::Rastrigin) {
              term = computeRastriginTerm(xi, expenseFactor);
            }
            else /* Ellipsoid */ {
              term = computeEllipsoidTerm(xi, d, dimension, expenseFactor);
            }

            loc += term;
          },
          fx
        );

        // 2) distance euclidienne au meilleur connu
        Kokkos::parallel_reduce(
          Kokkos::TeamThreadRange(team, dimension),
          [&](int d, double& loc){
            double xi    = centers(i,d) + 0.5 * r;
            double diff  = xi - bestSolutionDev(d);
            loc += diff * diff;
          },
          dist
        );
        dist = sqrt(dist);
        if(dist == 0.0) dist = 1e-10;

        totalScore += (fx - bestObjVal) / dist;
      }

      scores(i) = totalScore / numPoints;
    }
  );
}





// ==================================================================
//  Intensive local search – generic version
//  • identical control flow to your tuned code
//  • per‑coordinate “partial term” picked by ObjectiveType
// ==================================================================
Kokkos::View<double*, mem_space>
intensiveLocalSearch(const Hypersphere&                    hs,
                     const Kokkos::View<double*, mem_space>& initCenter,
                     double                                 initRadius,
                     int                                    maxIters,
                     double                                 stepSize,
                     double                                 phi,
                     double                                 omegaMin,
                     ObjectiveType                          objType,
                     int                                    expenseFactor)
{
  // ---------------- initialisation ----------------
  auto c = reconstructCenter(hs, initCenter, initRadius);
  Kokkos::View<double*, mem_space> bestSol("bestSol", hs.dimension);
  Kokkos::deep_copy(bestSol, c);

  auto policy = Kokkos::TeamPolicy<exec_space>(1, Kokkos::AUTO);

  Kokkos::View<double*, mem_space> localB ("localB" , hs.dimension);
  Kokkos::View<double*, mem_space> newVals("newVals", hs.dimension);
  Kokkos::View<double*, mem_space> termD  ("termD"  , hs.dimension);
  Kokkos::View<bool*,   mem_space> improved("imp"   , 1);

  // ---------------- device kernel -----------------
  Kokkos::parallel_for("ILS", policy,
    KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team)
  {
    // helper: one partial term
    auto partial = [&](double x, int dim)->double {
      switch (objType)
      {
        case ObjectiveType::Sphere:
          return computeSphereTerm(x, expenseFactor);

        case ObjectiveType::Rastrigin:
          return computeRastriginTerm(x, expenseFactor);

        case ObjectiveType::Ellipsoid:
          return computeEllipsoidTerm(x, dim, hs.dimension, expenseFactor);
      }
      return 0.0; // unreachable, silences warnings
    };

    // 1) copy initial point into localB
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension),
      [&](int d){ localB(d) = bestSol(d); });
    team.team_barrier();

    // 2) pre‑compute termD for each dimension
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension),
      [&](int d){ termD(d) = partial(localB(d), d); });
    team.team_barrier();

    // 3) total value
    double bestVal = 0.0;
    Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, hs.dimension),
      [&](int d, double& s){ s += termD(d); }, bestVal);
    team.team_barrier();

    // ------------- main pattern search loop ----------
    double omega = stepSize;
    for(int iter = 0; iter < maxIters; ++iter)
    {
      if(team.team_rank() == 0) improved(0) = false;
      team.team_barrier();

      // for each coordinate try ± omega
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension),
        [&](int d)
      {
        double x  = localB(d);
        double xp = x + omega;
        double xm = x - omega;

        double oldP = termD(d);
        double p1   = partial(xp, d);
        double p2   = partial(xm, d);

        double bestP = oldP;
        double bestX = x;

        if(p1 < bestP){ bestP = p1; bestX = xp; improved(0) = true; }
        if(p2 < bestP){ bestP = p2; bestX = xm; improved(0) = true; }

        newVals(d) = bestX;
        termD (d)  = bestP;
      });
      team.team_barrier();

      // update localB
      Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension),
        [&](int d){ localB(d) = newVals(d); });
      team.team_barrier();

      // recompute total value
      double newVal = 0.0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, hs.dimension),
        [&](int d, double& s){ s += termD(d); }, newVal);
      bestVal = newVal;
      team.team_barrier();

      // adapt step size on master thread
      if(team.team_rank() == 0){
        if(!improved(0)){
          omega /= phi;
          if(omega < omegaMin) iter = maxIters; // break
        }
      }
      team.team_barrier();
    }

    // copy localB → bestSol
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, hs.dimension),
      [&](int d){ bestSol(d) = localB(d); });
  });

  return bestSol;
}



std::vector<Hypersphere> decomposeHypersphere(const Hypersphere& parent){
    std::vector<Hypersphere> result;
    int D       = parent.dimension;
    int newDepth= parent.depth + 1;
    for(int k=0; k< D; k++){
        for(int sign=-1; sign<=1; sign+=2){
            HypersphereID childID = parent.id;
            childID.emplace_back(k, sign);
            Hypersphere h(D, newDepth, childID);
            result.push_back(h);
        }
    }
    return result;
}

std::vector<int> selectBestIndices(const std::vector<double>& scores, int beta){
    std::vector<int> inds(scores.size());
    std::iota(inds.begin(), inds.end(), 0);
    beta = std::min(beta, (int)scores.size());
    std::partial_sort(inds.begin(), inds.begin()+ beta, inds.end(),
      [&](int a,int b){
        return scores[a] < scores[b];
    });
    inds.resize(beta);
    return inds;
}

size_t get_available_memory(){
    size_t free_mem = SIZE_MAX;
#ifdef KOKKOS_ENABLE_CUDA
    // For CUDA, query free GPU memory
    size_t total_mem = 0;
    cudaMemGetInfo(&free_mem, &total_mem);
#endif
    return free_mem;
}

int parseBeta(int argc, char* argv[], int wrank){
    int beta = 50;
    if(argc > 1){
        try{
            beta = std::stoi(argv[1]);
            if(beta <= 0){
                if(wrank == 0)
                    std::cerr<<"Invalid beta. using 50.\n";
                beta=50;
            }
        }catch(...){
            if(wrank==0)
                std::cerr<<"Invalid beta arg. using 50\n";
        }
    }
    return beta;
}

int parseDimension(int argc, char* argv[], int wrank){
    int dim = 20000;
    if(argc>2){
        try{
            dim = std::stoi(argv[2]);
        }catch(...){
            if(wrank==0)
                std::cerr<<"Invalid dimension arg. using 20000.\n";
        }
    }
    return dim;
}

int parseBudget(int argc, char* argv[], int rank) {
  if (argc < 4) {
      if (rank == 0) {
          std::cerr << "Usage: " << argv[0] << " <beta> <dimension> <budget> [expenseFactor]" << std::endl;
      }
      MPI_Finalize();
      exit(EXIT_FAILURE);
  }
  return std::stoi(argv[3]);
}

// FIXED: Now reading argv[4] if it exists
int parseExpenseFactor(int argc, char* argv[], int rank) {
    int expenseFactor = 1;
    if (argc > 4) {
        try {
            expenseFactor = std::stoi(argv[4]);
            // optional checks
        } catch (...) {
            if (rank == 0) {
                std::cerr << "Invalid expense factor. Using default=1.\n";
            }
        }
    }
    return expenseFactor;
}



ObjectiveType parseObjectiveType(int argc, char* argv[], int idx, int wrank)
{
  if(argc <= idx) return ObjectiveType::Sphere;          // default

  std::string s(argv[idx]);
  for(auto& c : s) c = std::tolower(c);

  if(s == "rastrigin")  return ObjectiveType::Rastrigin;
  if(s == "ellipsoid")  return ObjectiveType::Ellipsoid;
  if(s != "sphere" && wrank == 0)
      std::cerr << "Unknown objective \"" << argv[idx]
                << "\" – falling back to Sphere.\n";
  return ObjectiveType::Sphere;
}


// ------------------------------------------------------------------
// Ellipsoidal computeObjectiveValue
// ------------------------------------------------------------------
double computeEllipsoidObjectiveValue(const Kokkos::View<double*, mem_space>& solution, int expensefactor){
    // mirror back to host for simplicity
    auto h_sol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution);
    int D = static_cast<int>(h_sol.extent(0));
    double sum = 0.0;
    for(int i = 0; i < D; ++i){
      double xi = h_sol(i);
      sum += computeEllipsoidTerm(xi, i, D, expensefactor);
    }
    return sum;
  }
  



// ------------------------------------------------------------------
// Host-side wrapper: copies data to host then calls the generic kernel
// ------------------------------------------------------------------
double computeObjectiveValue(const Kokkos::View<double*, mem_space>& solution,
                             int               expenseFactor,
                             ObjectiveType     objType)
{
    // Mirror the device view so we can read it safely on the host
    auto hostSol = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), solution);

    // Re-use the template dispatcher that already exists
    return computeObjective(hostSol, expenseFactor, objType);
}

