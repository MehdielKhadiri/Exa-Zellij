#include "py_vqe_shim.hpp"
#include <stdexcept>

extern "C" {
int init_vqe_bridge(const char*, const char*, const char*) { return 1; }
int init_python_objective(const char*, const char*, int) { return 1; }
int vqe_num_params() { return 0; }
bool vqe_is_ready() { return false; }
const char* vqe_last_error() {
    return "Python adapter disabled. Rebuild with -DFQO_ENABLE_PYTHON=ON, or use the Python API callback adapter.";
}
double vqe_eval_trampoline(const double*, int) {
    throw std::runtime_error(vqe_last_error());
}
void vqe_eval_many_trampoline(const double*, int, int, double*) {
    throw std::runtime_error(vqe_last_error());
}
}
