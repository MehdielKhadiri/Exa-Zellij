# vqe_bridge.py
# JAX + PennyLane VQE bridge with per-(mol,basis,device,dtype,shape) persistent cache

from __future__ import annotations
import os, json, glob, hashlib
from pathlib import Path
import numpy as np
import jax
from jax import numpy as jnp
import pennylane as qml

# optional: jaxlib info (if present)
try:
    import jaxlib  # type: ignore
    _JAXLIB_VER = getattr(jaxlib, "__version__", "unknown")
except Exception:
    _JAXLIB_VER = "unknown"

# prefer modern cache API, fall back if needed
try:
    from jax.experimental import compilation_cache as cc  # type: ignore
except Exception:  # very old JAX
    cc = None  # type: ignore

# ---------- helpers ----------

def _cache_root() -> str:
    # canonical writable cache root
    return os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")))


def _jax_cache_root() -> str:
    """Return a worker-private JAX cache root when one is configured."""
    return os.environ.get(
        "VQE_JAX_CACHE_ROOT",
        os.path.join(_cache_root(), "jax_persistent_cache"),
    )

def _safe_token(s: str) -> str:
    # stable, filesystem-safe token
    s = s.strip().lower()
    out = []
    for ch in s:
        if ch.isalnum() or ch in "._+-":
            out.append(ch)
        elif ch.isspace():
            out.append("_")
        # else drop
    return "".join(out) or "na"

def _key_for_cache(mol: str, basis: str, n_qubits: int, n_params: int,
                   dev_label: str, dtype: jnp.dtype, state_dtype: str,
                   wrap_mode: str) -> str:
    """
    Build a directory name key that uniquely identifies the compiled executable set.
    """
    dtype_tok = "f32" if dtype == jnp.float32 else "f64"
    # include versions to avoid ABI mismatch after upgrades
    pl_ver   = getattr(qml, "__version__", "unknown")
    jax_ver  = getattr(jax, "__version__", "unknown")
    jaxlib_v = _JAXLIB_VER

    parts = [
        f"mol={_safe_token(mol)}",
        f"basis={_safe_token(basis)}",
        f"q={n_qubits}",
        f"p={n_params}",
        f"dev={_safe_token(dev_label)}",
        f"dt={dtype_tok}",
        f"state={_safe_token(state_dtype)}",
        f"wrap={_safe_token(wrap_mode)}",
        f"jax={_safe_token(jax_ver)}",
        f"jaxlib={_safe_token(jaxlib_v)}",
        f"pl={_safe_token(pl_ver)}",
    ]
    key = "__".join(parts)
    if len(key) > 140:
        h = hashlib.sha1(key.encode()).hexdigest()[:16]
        key = key[:120] + "__" + h
    return key

def _set_jax_cache_dir(dirpath: str) -> None:
    os.makedirs(dirpath, exist_ok=True)
    if cc is not None and hasattr(cc, "set_cache_dir"):
        cc.set_cache_dir(dirpath)          # JAX >= 0.4.21
    elif cc is not None and hasattr(cc, "initialize_cache"):
        cc.initialize_cache(dirpath)       # older experimental API
    else:
        jax.config.update("jax_compilation_cache_dir", dirpath)
    print(f"[VQE] JAX persistent cache: {dirpath}", flush=True)
    print(f"[VQE] JAX version: {jax.__version__}", flush=True)

def _write_meta(dirpath: str, meta: dict) -> None:
    try:
        with open(os.path.join(dirpath, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, sort_keys=True)
    except Exception:
        pass

def _debug_qchem_cache(molname: str, basis: str) -> None:
    """Lightweight listing of matching local HDF5s (helps confirm offline readiness)."""
    try:
        root  = _cache_root()
        patt  = f"*{molname.lower()}*{basis.lower().replace('-', '')}*.h5"
        print(f"[VQE debug] XDG_CACHE_HOME={root}")
        qroots = [
            Path(root) / "datasets" / "qchem",
            Path(root) / "pennylane" / "datasets" / "qchem",
        ]
        print(f"[VQE debug] scanning: {qroots}  pattern: {patt}")
        for p in sorted(candidate for qroot in qroots for candidate in qroot.rglob("*.h5")):
            try:
                import h5py
                with h5py.File(str(p), "r"):
                    pass
                sz = p.stat().st_size
                print(f"[VQE debug] H5 OK  {sz} bytes  ->  {p}")
            except Exception as e:
                print(f"[VQE debug] H5 BROKEN -> {p}  ({e})")
    except Exception:
        pass

# ---------- import reference construction code ----------
from vqe_cost_reference import (
    load_molecule, build_qubit_hamiltonian, make_uccsd_ansatz
)

# ---------- module session ----------
_session: dict = {}  # holds compiled callables + config

# ---------- API used by C++ ----------

def init(molname: str = "LiH", basis: str = "STO-3G") -> int:
    """
    Build the fixed-geometry cost E(theta) and prepare compiled single/batched evaluators.
    Returns the number of parameters.
    """
    # (a) dataset sanity (does not modify global cache)
    _debug_qchem_cache(molname, basis)

    # (b) build molecular model & Hamiltonian
    mol, e_ref = load_molecule(molname, basis)
    H, n_qubits = build_qubit_hamiltonian(mol)
    ansatz, n_params, hf_state, s_wires, d_wires = make_uccsd_ansatz(mol, n_qubits)
    print(f"[VQE py] init(mol={molname}, basis={basis})", flush=True)

    # (c) pick a device. Parameter and statevector precision are independent:
    # the historical LUMI path used FP32 JAX parameters while Lightning-Kokkos
    # retained its complex128 statevector representation.
    requested_state_dtype = os.environ.get("VQE_STATE_DTYPE", "auto").strip().lower()
    state_dtype_aliases = {
        "auto": None,
        "complex64": np.complex64,
        "c64": np.complex64,
        "fp32": np.complex64,
        "float32": np.complex64,
        "complex128": np.complex128,
        "c128": np.complex128,
        "fp64": np.complex128,
        "float64": np.complex128,
    }
    if requested_state_dtype not in state_dtype_aliases:
        raise ValueError(
            f"Unsupported VQE_STATE_DTYPE={requested_state_dtype!r}; "
            "expected auto, complex64, or complex128"
        )
    requested_c_dtype = state_dtype_aliases[requested_state_dtype]

    def _pick_device():
        requested = os.environ.get("VQE_DEVICE", "").strip()
        candidates = (
            [requested]
            if requested
            else ["lightning.kokkos", "lightning.gpu", "lightning.qubit", "default.qubit"]
        )
        for name in candidates:
            try:
                kwargs = {}
                if requested_c_dtype is not None:
                    if not name.startswith("lightning."):
                        raise ValueError(
                            "Explicit VQE_STATE_DTYPE requires a Lightning device"
                        )
                    kwargs["c_dtype"] = requested_c_dtype
                return qml.device(name, wires=n_qubits, **kwargs)
            except Exception:
                pass
        raise RuntimeError(
            "No requested PennyLane device is available: "
            + ", ".join(candidates)
        )

    dev = _pick_device()
    dev_label = (getattr(dev, "short_name", None)
                 or getattr(dev, "name", None)
                 or dev.__class__.__name__)
    on_gpu = str(dev_label).lower() in ("lightning.kokkos", "lightning.gpu")

    # (d) dtype policy
    requested_dtype = os.environ.get("VQE_DTYPE", "auto").strip().lower()
    if requested_dtype not in {"auto", "fp32", "float32", "fp64", "float64"}:
        raise ValueError(f"Unsupported VQE_DTYPE={requested_dtype!r}")
    use_fp64 = (
        requested_dtype in {"fp64", "float64"}
        or (requested_dtype == "auto" and not on_gpu)
    )
    jax.config.update("jax_enable_x64", use_fp64)
    dtype = jnp.float64 if use_fp64 else jnp.float32
    device_c_dtype = getattr(dev, "c_dtype", getattr(dev, "_c_dtype", None))
    if device_c_dtype is not None:
        state_dtype = np.dtype(device_c_dtype).name
    elif requested_c_dtype is not None:
        state_dtype = np.dtype(requested_c_dtype).name
    else:
        # JAX-native simulators follow the enabled JAX precision.
        state_dtype = "complex128" if use_fp64 else "complex64"
    wrap_mode = os.environ.get("VQE_WRAP_MODE", "principal").strip().lower()
    if wrap_mode not in {"principal", "none"}:
        raise ValueError(f"Unsupported VQE_WRAP_MODE={wrap_mode!r}")

    # (e) configure a *dedicated* persistent cache directory for this circuit
    cache_root  = _cache_root()
    jax_cache   = _jax_cache_root()
    cache_key   = _key_for_cache(
        molname, basis, n_qubits, n_params, str(dev_label), dtype,
        state_dtype, wrap_mode
    )
    cache_dir   = os.path.join(jax_cache, cache_key)
    _set_jax_cache_dir(cache_dir)
    _write_meta(cache_dir, {
        "mol": molname,
        "basis": basis,
        "n_qubits": n_qubits,
        "n_params": n_params,
        "device": str(dev_label),
        "dtype": "float32" if dtype == jnp.float32 else "float64",
        "parameter_dtype": "float32" if dtype == jnp.float32 else "float64",
        "state_dtype": state_dtype,
        "wrap_mode": wrap_mode,
        "jax": getattr(jax, "__version__", "unknown"),
        "jaxlib": _JAXLIB_VER,
        "pennylane": getattr(qml, "__version__", "unknown"),
    })

    # (f) angle handling.  The historical PFDA-BS path searches directly in
    # [-pi, pi] and therefore keeps the wrapping default.  Quantum-gradient
    # rules, however, must be allowed to query shifted angles outside that
    # interval.  VQE_WRAP_MODE=none disables wrapping without changing the
    # candidate domain enforced by the optimizer.
    def _wrap(theta):
        if wrap_mode == "none":
            return theta
        # map angles to (-pi, pi]; supports vector or batch
        return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    # (g) QNode definition (no compilation yet)
    @qml.qnode(dev, interface="jax")
    def circuit(theta):
        ansatz(_wrap(theta), wires=dev.wires)
        return qml.expval(H)

    @qml.qnode(dev, interface="jax")
    def state_circuit(theta):
        ansatz(_wrap(theta), wires=dev.wires)
        return qml.state()

    # (h) compile single and batched evaluators
    single  = jax.jit(circuit)                 # (D,) -> ()
    batched = jax.jit(jax.vmap(circuit))       # (B,D) -> (B,)

    # (i) store session
    _session.clear()
    _session["mol"]       = molname
    _session["basis"]     = basis
    _session["n_qubits"]  = n_qubits
    _session["n_params"]  = n_params
    _session["n_electrons"] = int(mol.n_electrons)
    _session["n_orbitals"] = int(mol.n_orbitals)
    _session["hamiltonian_backend"] = (
        "openfermion" if int(mol.n_electrons) % 2 else "pyscf"
    )
    _session["fci_energy"] = float(e_ref)
    _session["single"]    = single
    _session["batched"]   = batched
    _session["state_circuit"] = state_circuit
    _session["dtype"]     = dtype
    _session["state_dtype"] = state_dtype
    _session["dev_label"] = str(dev_label)
    _session["on_gpu"]    = bool(on_gpu)
    _session["cache_dir"] = cache_dir
    _session["B_PAD"]     = int(os.environ.get("VQE_BATCH", "1"))
    _session["wrap_mode"] = wrap_mode

    # (j) status line
    print(f"[VQE] init: device={_session['dev_label']} parameter_dtype={dtype} "
          f"state_dtype={state_dtype} "
          f"(jax_enable_x64={use_fp64}) n_params={n_params}", flush=True)

    return n_params


def metadata() -> dict:
    """Return immutable metadata for the initialized molecular objective."""
    if not _session:
        raise RuntimeError("Call init() first")
    return {
        "molecule": _session["mol"],
        "basis": _session["basis"],
        "n_qubits": int(_session["n_qubits"]),
        "n_params": int(_session["n_params"]),
        "n_electrons": int(_session["n_electrons"]),
        "n_orbitals": int(_session["n_orbitals"]),
        "hamiltonian_backend": _session["hamiltonian_backend"],
        "fci_energy": float(_session["fci_energy"]),
        "device": _session["dev_label"],
        "dtype": "float32" if _session["dtype"] == jnp.float32 else "float64",
        "parameter_dtype": (
            "float32" if _session["dtype"] == jnp.float32 else "float64"
        ),
        "state_dtype": _session["state_dtype"],
        "cache_dir": _session["cache_dir"],
        "wrap_mode": _session["wrap_mode"],
    }


def options() -> dict[str, list[str]]:
    """
    Return {molname: [bases...]} discovered in the PennyLane 'qchem' dataset.
    (May require network if not using local cache; not used by the C++ path.)
    """
    by_mol = {}
    try:
        ds = qml.data.load("qchem", attributes=["molecule"])
    except Exception:
        return {}
    def _get(rec, key, default=None):
        for src in (
            getattr(rec, key, None),
            getattr(rec, "metadata", {}).get(key) if hasattr(rec, "metadata") else None,
            getattr(rec, "kwargs", {}).get(key) if hasattr(rec, "kwargs") else None,
        ):
            if src is not None:
                return src
        return default
    for rec in ds:
        molname = _get(rec, "molname")
        if molname is None:
            mol = getattr(rec, "molecule", None)
            if mol is not None and hasattr(mol, "name") and mol.name:
                molname = str(mol.name)
            elif mol is not None and hasattr(mol, "symbols"):
                molname = "".join(list(mol.symbols))
            else:
                molname = "UNKNOWN_MOLECULE"
        basis = _get(rec, "basis", "UNKNOWN_BASIS")
        by_mol.setdefault(str(molname), set()).add(str(basis))
    return {m: sorted(list(bases)) for m, bases in sorted(by_mol.items())}


def print_options() -> None:
    opts = options()
    if not opts:
        print("[vqe_bridge] No entries found in qchem dataset.")
        return
    print("=== Available molecules / bases (qml.data: 'qchem') ===")
    for m, bases in opts.items():
        print(f"  {m}: {', '.join(bases)}")


def eval(theta_list) -> float:
    """
    Evaluate E(theta_list) -> float.
    """
    if not _session:
        raise RuntimeError("Call init() first")
    D = _session["n_params"]
    theta = jnp.asarray(theta_list, dtype=_session["dtype"])
    if theta.shape[0] != D:
        raise ValueError(f"theta has {int(theta.shape[0])} params, expected {D}")
    val = _session["single"](theta)
    return float(val)


def eval_batch(theta_2d):
    """
    Evaluate a batch: theta_2d shape must be (B, D).
    Padding to a fixed block size P = VQE_BATCH is used to keep compiled shapes stable.
    Returns a NumPy 1-D array of length B.
    """
    import numpy as np
    if not _session:
        raise RuntimeError("Call init() first")

    D = _session["n_params"]
    theta = jnp.asarray(theta_2d, dtype=_session["dtype"])
    if theta.ndim != 2 or int(theta.shape[1]) != D:
        raise ValueError(f"theta must be (B,{D})")

    B = int(theta.shape[0])
    P = _session["B_PAD"]

    # one-time warmup for (P,D)
    if "_warm" not in _session:
        dummy = jnp.zeros((P, D), dtype=_session["dtype"])
        _ = _session["batched"](dummy)  # compile for (P,D)
        _session["_warm"] = True
        print(f"[VQE] warmup compiled for batch={P}", flush=True)

    out = []
    i = 0
    while i < B:
        n = min(P, B - i)
        block = theta[i:i+n]
        if n < P:
            pad = jnp.zeros((P - n, D), dtype=_session["dtype"])
            block = jnp.concatenate([block, pad], axis=0)
        y = jax.device_get(_session["batched"](block))
        out.append(y[:n])  # drop padding
        i += n

    return np.concatenate(out, axis=0)


def state_metric_diag(theta_list):
    """Return the diagonal Fubini--Study metric for the UCCSD state.

    This exact statevector implementation is used only by the small
    knowledge-driven QNG benchmark.  The benchmark accounts for the quantum
    work separately using the standard diagonal-metric circuit count; the
    state Jacobian is an efficient noiseless-emulator implementation, not a
    claim that a physical QPU exposes its statevector.
    """
    import numpy as np

    if not _session:
        raise RuntimeError("Call init() first")
    D = _session["n_params"]
    theta = jnp.asarray(theta_list, dtype=_session["dtype"])
    if theta.ndim != 1 or int(theta.shape[0]) != D:
        raise ValueError(f"theta must be ({D},)")

    if "metric_diag" not in _session:
        state_circuit = _session["state_circuit"]

        def _metric_diag(x):
            psi = state_circuit(x)

            def _real_imag_state(z):
                state = state_circuit(z)
                return jnp.concatenate((jnp.real(state), jnp.imag(state)))

            # Forward mode scales with the number of UCCSD parameters rather
            # than the (usually much larger) real-imaginary state dimension.
            jac_ri = jax.jacfwd(_real_imag_state)(x)
            n_amplitudes = int(psi.shape[0])
            jac = jac_ri[:n_amplitudes] + 1j * jac_ri[n_amplitudes:]
            tangent_norm = jnp.sum(jnp.abs(jac) ** 2, axis=0)
            projection = jnp.sum(jnp.conj(psi)[:, None] * jac, axis=0)
            diag = jnp.real(tangent_norm - jnp.abs(projection) ** 2)
            return jnp.maximum(diag, jnp.asarray(0.0, dtype=diag.dtype))

        _session["metric_diag"] = jax.jit(_metric_diag)

    result = jax.device_get(_session["metric_diag"](theta))
    return np.asarray(result, dtype=np.float64)
