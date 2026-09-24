#pragma once
#include "FQOMpi.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>

// Version 1 callback protocol used by the Python driver. One native process
// exchanges batches over its private pipes; ordinary optimizer logs coexist
// on stdout. Distributed jobs use the in-process Python/VQE adapter instead.
inline void fqo_stdio_many(const double* x, int d, int n, double* out) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::cout << "\n@FQO_EVAL {\"points\":[" << std::setprecision(17);
    for (int i=0;i<n;++i) {
        if (i) std::cout << ',';
        std::cout << '[';
        for (int j=0;j<d;++j) {
            if(j) std::cout << ',';
            std::cout << x[size_t(i)*d+j];
        }
        std::cout << ']';
    }
    std::cout << "]}" << std::endl;
    for (int i=0;i<n;++i) {
        if (!(std::cin >> out[i]) || !std::isfinite(out[i])) {
            std::cerr << "External objective returned invalid data or closed its input pipe.\n";
            MPI_Abort(MPI_COMM_WORLD,1);
        }
    }
}
inline double fqo_stdio_one(const double* x,int d) {
    double value; fqo_stdio_many(x,d,1,&value); return value;
}
