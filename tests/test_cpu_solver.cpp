#include "FQOCpuSolver.hpp"
#include <cmath>
#include <iostream>
int main() {
    // Symmetric matrix with known eigenvalues 1 and 3.
    double a[]={2,1,1,2}, w[2]; int info=0;
    gpu_syevd(nullptr,2,a,2,w,nullptr,0,&info);
    if(info || std::abs(w[0]-1)>1e-12 || std::abs(w[1]-3)>1e-12) return 1;
    for(int j=0;j<2;++j) {
        if(std::abs(2*a[2*j]+a[2*j+1]-w[j]*a[2*j])>1e-12) return 2;
    }
    double b[]={4,2,2,3};
    gpu_potrf(nullptr,2,b,2,nullptr,0,&info);
    if(info || std::abs(b[0]-2)>1e-12 || std::abs(b[1]-1)>1e-12 || std::abs(b[3]-std::sqrt(2.0))>1e-12) return 3;
    double indefinite[]={1,2,2,1};
    gpu_potrf(nullptr,2,indefinite,2,nullptr,0,&info);
    if(!info) return 4;
    std::cout << "CPU eigensolver/Cholesky checks passed\n";
}
