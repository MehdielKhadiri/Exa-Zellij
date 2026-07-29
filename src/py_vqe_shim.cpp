#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <atomic>

namespace fs = std::filesystem;
namespace py = pybind11;




// ─────────────────────────────────────────────────────────────
// Fail-fast state
// ─────────────────────────────────────────────────────────────
static std::atomic<bool> g_ready{false};
static std::string       g_last_err;   // non-empty when a recent failure happened

static inline void set_fail(const std::string& msg) {
    g_ready    = false;
    g_last_err = msg;
}
static inline void set_ok() {
    g_ready    = true;
    g_last_err.clear();
}




struct PyVQE {
    py::scoped_interpreter guard;
    py::object mod;
    py::object eval_fn;
    py::object eval_batch_fn;
    int n_params = 0;

    static void append_path(py::list sys_path, const fs::path& p) {
        try { sys_path.append(p.string()); } catch (...) {}
    }

    PyVQE(const std::string& user_dir,
          const std::string& mol,
          const std::string& basis)
    : guard{} {
        try {
            py::gil_scoped_acquire gil;
            auto sys = py::module_::import("sys");
            auto sys_path = sys.attr("path").cast<py::list>();

            // 1) user-provided dir
            if (!user_dir.empty()) append_path(sys_path, user_dir);

            // 2) compile-time source dir fallback
        #ifdef VQE_PY_DIR
            append_path(sys_path, fs::path(VQE_PY_DIR));
        #endif

            // 3) env var
            if (const char* envp = std::getenv("VQE_PY_DIR")) {
                append_path(sys_path, fs::path(envp));
            }

            // 4) CWD
            append_path(sys_path, fs::current_path());

            // 5) executable dir (best-effort)
            try {
                fs::path exe = fs::read_symlink("/proc/self/exe").parent_path();
                append_path(sys_path, exe);
            } catch (...) {}

            // Import + init
            // Use importlib and force a reload so we never get a stale module
            auto importlib = py::module_::import("importlib");
            mod = importlib.attr("import_module")("vqe_bridge");
            importlib.attr("reload")(mod);
            py::object modfile = mod.attr("__file__");
            py::print("[VQE shim] using module:", modfile);
            py::print("[VQE shim] args mol/basis passed from C++:", mol, basis);
            n_params = mod.attr("init")(mol, basis).cast<int>();
            eval_fn       = mod.attr("eval");
            eval_batch_fn = py::none();
            // eval_batch may not exist in older vqe_bridge; guard it
            if (py::hasattr(mod, "eval_batch"))
                eval_batch_fn = mod.attr("eval_batch");
            // success path: clear any previous error
            set_ok();
        } catch (const py::error_already_set& e) {
            std::cerr << "[VQE shim] Python import/init failed: " << e.what() << "\n"
                      << "  Tried user_dir, VQE_PY_DIR(def), env VQE_PY_DIR, CWD, exe dir.\n"
                      << "  Fix: set VQE_PY_DIR or pass a valid folder to init_vqe_bridge().\n";
            set_fail(e.what());
            n_params = 0;
        } catch (const std::exception& e) {
            std::cerr << "[VQE shim] unexpected std::exception: " << e.what() << "\n";
            set_fail(e.what());
            n_params = 0;
        } catch (...) {
            std::cerr << "[VQE shim] unknown error during Python init\n";
            set_fail("unknown error during Python init");
            n_params = 0;
        }
    }

    double eval(const double* x, int D) {
    if (!mod || n_params == 0) return 1e300;
    py::gil_scoped_acquire gil;

    // zero-copy 1D view: shape={D}, strides={sizeof(double)}
    py::array_t<double> arr({D}, {sizeof(double)}, x);

    py::object out = eval_fn(arr);
    return out.cast<double>();
}

void eval_many(const double* X, int D, int N, double* out) {
    if (!mod || n_params == 0) {
        std::fill(out, out + N, 1e300);
        return;
    }
    py::gil_scoped_acquire gil;

    if (!eval_batch_fn.is_none()) {
        // Build a zero-copy view over the row-major C++ batch.
        py::array_t<double> arr({N, D}, {sizeof(double)*D, sizeof(double)}, X);
        try {
            py::object out_obj = eval_batch_fn(arr);
            py::array res = out_obj.cast<py::array>();
            if (res.ndim() != 1 || res.shape(0) != N)
                throw std::runtime_error("eval_batch returned wrong shape");
            std::memcpy(out, res.data(), sizeof(double) * N);
        } catch (const std::exception& e) {
            std::cerr << "[VQE shim] batched evaluation failed: "
                      << e.what() << std::endl;
            std::fill(out, out + N, 1e300);
        }
    } else {
        for (int n = 0; n < N; ++n) {
            const double* row = X + size_t(n) * size_t(D);
            out[n] = eval(row, D);
        }
    }
}


};

static PyVQE* g_py = nullptr;

extern "C" {

int init_vqe_bridge(const char* py_dir, const char* mol, const char* basis) {
    // Recreate from scratch so we don’t carry a half-broken interpreter
    if (g_py) { delete g_py; g_py = nullptr; }
    g_py = new PyVQE(py_dir ? std::string(py_dir) : std::string{},
                     mol ? mol : "LiH",
                     basis ? basis : "STO-3G");
    if (!g_py) {
        set_fail("allocation failure (PyVQE == nullptr)");
        return 2;
    }
    if (g_py->n_params <= 0) {
        if (g_last_err.empty())
            set_fail("vqe_bridge.init returned n_params <= 0");
        return 1;
    }
    set_ok();
    return 0;
}

int vqe_num_params() {
    return g_py ? g_py->n_params : 0;
}

double vqe_eval_trampoline(const double* x, int D) {
    return g_py ? g_py->eval(x, D) : 1e300;
}


// New: batched trampoline
void vqe_eval_many_trampoline(const double* X, int D, int N, double* out) {
    if (!g_py || !g_ready.load()) { std::fill(out, out+N, 1e300); return; }
    g_py->eval_many(X, D, N, out);
}


bool vqe_is_ready() {
    return g_ready.load();
}

const char* vqe_last_error() {
    return g_last_err.empty() ? "" : g_last_err.c_str();
}


} // extern "C"
