# vqe_bridge.py
# JAX + PennyLane VQE bridge with per-(mol,basis,device,dtype,shape) persistent cache

from __future__ import annotations
import os, json, glob, hashlib
from pathlib import Path
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
                   dev_label: str, dtype: jnp.dtype) -> str:
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
        qroot = os.path.join(root, "datasets", "qchem")
        patt  = f"*{molname.lower()}*{basis.lower().replace('-', '')}*.h5"
        print(f"[VQE debug] XDG_CACHE_HOME={root}")
        print(f"[VQE debug] scanning: {qroot}  pattern: {patt}")
        for p in sorted(Path(qroot).glob(patt)):
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

    # (c) pick a device
    def _pick_device():
        for name in ("lightning.kokkos", "lightning.gpu", "lightning.qubit", "default.qubit"):
            try:
                return qml.device(name, wires=n_qubits)
            except Exception:
                pass
        raise RuntimeError("No PennyLane device available")

    dev = _pick_device()
    dev_label = (getattr(dev, "short_name", None)
                 or getattr(dev, "name", None)
                 or dev.__class__.__name__)
    on_gpu = str(dev_label).lower() in ("lightning.kokkos", "lightning.gpu")

    # (d) dtype policy
    jax.config.update("jax_enable_x64", not on_gpu)
    dtype = jnp.float32 if on_gpu else jnp.float64

    # (e) configure a *dedicated* persistent cache directory for this circuit
    cache_root  = _cache_root()
    jax_cache   = os.path.join(cache_root, "jax_persistent_cache")
    cache_key   = _key_for_cache(molname, basis, n_qubits, n_params, str(dev_label), dtype)
    cache_dir   = os.path.join(jax_cache, cache_key)
    _set_jax_cache_dir(cache_dir)
    _write_meta(cache_dir, {
        "mol": molname,
        "basis": basis,
        "n_qubits": n_qubits,
        "n_params": n_params,
        "device": str(dev_label),
        "dtype": "float32" if dtype == jnp.float32 else "float64",
        "jax": getattr(jax, "__version__", "unknown"),
        "jaxlib": _JAXLIB_VER,
        "pennylane": getattr(qml, "__version__", "unknown"),
    })

    # (f) angle wrapping helper
    def _wrap(theta):
        # map angles to (-pi, pi]; supports vector or batch
        return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    # (g) QNode definition (no compilation yet)
    @qml.qnode(dev, interface="jax")
    def circuit(theta):
        ansatz(_wrap(theta), wires=dev.wires)
        return qml.expval(H)

    # (h) compile single and batched evaluators
    single  = jax.jit(circuit)                 # (D,) -> ()
    batched = jax.jit(jax.vmap(circuit))       # (B,D) -> (B,)

    # (i) store session
    _session.clear()
    _session["mol"]       = molname
    _session["basis"]     = basis
    _session["n_qubits"]  = n_qubits
    _session["n_params"]  = n_params
    _session["single"]    = single
    _session["batched"]   = batched
    _session["dtype"]     = dtype
    _session["dev_label"] = str(dev_label)
    _session["on_gpu"]    = bool(on_gpu)
    _session["cache_dir"] = cache_dir
    _session["B_PAD"]     = int(os.environ.get("VQE_BATCH", "1"))

    # (j) status line
    print(f"[VQE] init: device={_session['dev_label']} dtype={dtype} "
          f"(jax_enable_x64={not on_gpu}) n_params={n_params}", flush=True)

    return n_params


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
