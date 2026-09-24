#pragma once
#ifdef FQO_ENABLE_MPI
#include <mpi.h>
#else
// Single-process implementation of the collective operations used by FQO.
// This is deliberately not an MPI runtime; distributed launches are rejected.
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
using MPI_Comm = int;
using MPI_Request = int;
inline constexpr int MPI_COMM_WORLD=0, MPI_COMM_TYPE_SHARED=0, MPI_INFO_NULL=0;
inline constexpr int MPI_UNDEFINED=-1, MPI_REQUEST_NULL=0;
inline constexpr int MPI_THREAD_SINGLE=0, MPI_THREAD_FUNNELED=1,
    MPI_THREAD_SERIALIZED=2, MPI_THREAD_MULTIPLE=3;
inline constexpr int MPI_DOUBLE=1, MPI_INT=2, MPI_LONG_LONG=3,
    MPI_UNSIGNED_LONG_LONG=4, MPI_DOUBLE_INT=5;
inline constexpr int MPI_SUM=0, MPI_MIN=1, MPI_MAX=2, MPI_MINLOC=3;
inline constexpr int* MPI_STATUS_IGNORE=nullptr;
inline constexpr int* MPI_STATUSES_IGNORE=nullptr;
inline size_t fqo_mpi_size(int type) {
    struct DoubleInt { double value; int rank; };
    switch(type) {
        case MPI_DOUBLE: return sizeof(double);
        case MPI_INT: return sizeof(int);
        case MPI_LONG_LONG: return sizeof(long long);
        case MPI_UNSIGNED_LONG_LONG: return sizeof(unsigned long long);
        case MPI_DOUBLE_INT: return sizeof(DoubleInt);
        default: std::abort();
    }
}
inline int MPI_Init_thread(int*, char***, int, int* provided) {
    for (const char* name : {"OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_SIZE", "SLURM_NTASKS"}) {
        const char* value=std::getenv(name);
        if (value && std::strtol(value, nullptr, 10)>1) {
            std::cerr << "This FQO build has MPI disabled. Rebuild with -DFQO_ENABLE_MPI=ON for multiple ranks.\n";
            std::exit(1);
        }
    }
    *provided=MPI_THREAD_SINGLE; return 0;
}
inline int MPI_Finalize() { return 0; }
[[noreturn]] inline int MPI_Abort(MPI_Comm, int code) { std::exit(code ? code : 1); }
inline int MPI_Comm_rank(MPI_Comm, int* rank) { *rank=0; return 0; }
inline int MPI_Comm_size(MPI_Comm, int* size) { *size=1; return 0; }
inline int MPI_Comm_split(MPI_Comm, int color, int, MPI_Comm* out) { *out=color; return 0; }
inline int MPI_Comm_split_type(MPI_Comm, int, int, int, MPI_Comm* out) { *out=0; return 0; }
inline int MPI_Comm_free(MPI_Comm* comm) { *comm=MPI_UNDEFINED; return 0; }
inline int MPI_Barrier(MPI_Comm) { return 0; }
inline int MPI_Bcast(void*, int, int, int, MPI_Comm) { return 0; }
inline double MPI_Wtime() {
    using Clock=std::chrono::steady_clock;
    return std::chrono::duration<double>(Clock::now().time_since_epoch()).count();
}
inline int MPI_Allreduce(const void* s, void* r, int n, int t, int, MPI_Comm) {
    if (n && s!=r) std::memcpy(r,s,size_t(n)*fqo_mpi_size(t)); return 0;
}
inline int MPI_Reduce(const void* s, void* r, int n, int t, int op, int, MPI_Comm c) {
    return MPI_Allreduce(s,r,n,t,op,c);
}
inline int MPI_Allgather(const void* s, int n, int t, void* r, int, int, MPI_Comm c) {
    return MPI_Allreduce(s,r,n,t,0,c);
}
inline int MPI_Gather(const void* s, int n, int t, void* r, int m, int u, int, MPI_Comm c) {
    return MPI_Allgather(s,n,t,r,m,u,c);
}
inline int MPI_Allgatherv(const void* s, int n, int t, void* r, const int*, const int* d, int u, MPI_Comm c) {
    if (!n) return 0;
    return MPI_Allreduce(s,static_cast<char*>(r)+size_t(d[0])*fqo_mpi_size(u),n,t,0,c);
}
inline int MPI_Scatterv(const void* s, const int*, const int* d, int t, void* r, int n, int u, int, MPI_Comm c) {
    if (!n) return 0;
    return MPI_Allreduce(static_cast<const char*>(s)+size_t(d[0])*fqo_mpi_size(t),r,n,u,0,c);
}
inline int MPI_Iallgather(const void* s, int n, int t, void* r, int m, int u, MPI_Comm c, MPI_Request* q) {
    *q=1; return MPI_Allgather(s,n,t,r,m,u,c);
}
inline int MPI_Iallgatherv(const void* s, int n, int t, void* r, const int* m, const int* d, int u, MPI_Comm c, MPI_Request* q) {
    *q=1; return MPI_Allgatherv(s,n,t,r,m,d,u,c);
}
inline int MPI_Waitall(int n, MPI_Request* q, int*) {
    for(int i=0;i<n;++i) q[i]=MPI_REQUEST_NULL; return 0;
}
inline int MPI_Testany(int n, MPI_Request* q, int* index, int* flag, int*) {
    *index=MPI_UNDEFINED; *flag=1;
    for(int i=0;i<n;++i) if(q[i]!=MPI_REQUEST_NULL) { *index=i; q[i]=MPI_REQUEST_NULL; break; }
    return 0;
}
#endif
