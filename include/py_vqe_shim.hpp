#pragma once
extern "C" {
  // Fail-fast aware API: return 0 on success, nonzero on failure.
  // If nonzero is returned, vqe_is_ready()==false and vqe_last_error() holds a message.
  int  init_vqe_bridge(const char* py_dir, const char* mol, const char* basis);
  int  init_python_objective(const char* py_dir, const char* module, int dimension);
  int    vqe_num_params();
  double vqe_eval_trampoline(const double* x, int D);
  void   vqe_eval_many_trampoline(const double* X, int D, int N, double* out);
  // New helpers for early-abort logic
  bool        vqe_is_ready();           // true after a successful init
  const char* vqe_last_error();         // human-readable error string (never null; may be "")
}
