# vqe_cost_reference.py
# Minimal reference to obtain the VQE cost function E(theta) (and optionally E(theta, x))
# Requires: pennylane, pennylane-qchem, jax, (lightning is recommended but default.qubit also works)

from __future__ import annotations
import sys, json, functools, os
from typing import Dict, List, Tuple
import pennylane as qml
import jax
from jax import numpy as jnp
from pennylane.pauli import pauli_sentence

jax.config.update("jax_enable_x64", True)

# ----------------------------- CLI/defaults ----------------------------------
MOL_NAME_DEFAULT = "LiH"
BASIS_DEFAULT    = "STO-3G"

def parse_cli():
    if len(sys.argv) >= 3:
        return sys.argv[1], sys.argv[2]
    return MOL_NAME_DEFAULT, BASIS_DEFAULT

# ---------------------- 1) Dataset + Molecule + H -----------------------------
# ---------------------- 1) Dataset + Molecule + H -----------------------------
# ---------------------- 1) Dataset + Molecule + H -----------------------------
def load_molecule(molname: str, basis: str):
    """
    Offline (VQE_OFFLINE=1): open the local HDF5 under $XDG_CACHE_HOME/datasets/qchem
    with zero network traffic. Online: use qml.data.load("qchem", ...).
    Returns (molecule, fci_energy).
    """
    import os
    from pathlib import Path
    from pennylane.data import Dataset
    import glob

    offline = os.environ.get("VQE_OFFLINE", "0") == "1"

    if offline:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home()))
        qroot = root / "datasets" / "qchem"

        # --- exact map (use raw strings for any '\a' sequences) ---
        exact = {
            ("HeH+", "STO-3G"): r"heh-cation-heh+-sto-3g-0.775-n\a-4.h5",
            ("H3+",  "STO-3G"): r"h3-cation-h3+-sto-3g-0.874-∡ hhh = 60°-6.h5",
            ("H4",   "STO-3G"): r"h4-molecule-h4-sto-3g-0.88-∡ hhh = 180°-8.h5",
            ("H5",   "STO-3G"): r"h5-molecule-h5-sto-3g-1.0-∡ hhh = 180°-10.h5",
            ("H6",   "STO-3G"): r"h6-molecule-h6-sto-3g-0.92-∡ hhh = 180°-12.h5",
        }

        fname = exact.get((molname, basis))
        path = (qroot / fname) if fname else None

        # If not mapped or missing, try robust discovery (handles BEL vs '\a')
        if path is None or not path.exists():
            patt = f"{molname.lower()}*-{basis.lower()}-*.h5"
            cands = sorted(Path(p) for p in glob.glob(str(qroot / patt)))
            # Prefer the one ending with '-20.h5' (Li2 has 20 qubits)
            cands = [p for p in cands if p.name.endswith("-20.h5")] or cands
            if not cands:
                raise FileNotFoundError(f"No local HDF5 found for {molname}/{basis} under {qroot}")
            path = cands[0]

        # Last-resort swap: if user typed literal '\a', try BEL variant and vice versa
        if not path.exists():
            s = str(path)
            alt = Path(s.replace(r"\a", "\a")) if r"\a" in s else Path(s.replace("\a", r"\a"))
            if alt.exists():
                path = alt

        ds = Dataset.open(str(path), mode="r")
        try:
            mol = getattr(ds, "molecule")
            efi = float(getattr(ds, "fci_energy"))
        finally:
            ds.close()
        return mol, efi

    # ---------- Online (small scale) ----------
    data = qml.data.load(
        "qchem",
        molname=molname,
        basis=basis,
        attributes=["molecule", "fci_energy"],
    )[-1]
    return data.molecule, float(data.fci_energy)






def build_qubit_hamiltonian(mol: qml.qchem.Molecule):
    """Jordan–Wigner qubit Hamiltonian for this Molecule (fixed geometry)."""
    H, n_qubits = qml.qchem.molecular_hamiltonian(
        mol,
        method="pyscf",            # robust on Linux/macOS; use "openfermion" if you have that stack
        mapping="jordan_wigner",
    )
    return H, n_qubits

# ------------------- 2) UCCSD ansatz + dimensions ----------------------------
def make_uccsd_ansatz(mol: qml.qchem.Molecule, n_qubits: int):
    """Return (ansatz callable, n_params, hf_state, s_wires, d_wires)."""
    n_spin_orbs = mol.n_orbitals * 2
    hf_state    = qml.qchem.hf_state(mol.n_electrons, n_spin_orbs)
    singles, doubles = qml.qchem.excitations(mol.n_electrons, n_spin_orbs)
    s_wires, d_wires = qml.qchem.excitations_to_wires(singles, doubles)
    ansatz = functools.partial(qml.UCCSD, init_state=hf_state,
                               s_wires=s_wires, d_wires=d_wires)
    n_params = len(singles) + len(doubles)
    return ansatz, n_params, hf_state, s_wires, d_wires

# ------------------- 3) Cost E(theta) for fixed geometry ---------------------
def make_cost_fixed_geometry(mol: qml.qchem.Molecule, basis_label: str):
    """
    Returns:
      cost(theta) -> float energy (Hartree)
      meta dict with: n_qubits, n_params, electrons, orbitals, geometry (Bohr), basis, symbols
    """
    H, n_qubits = build_qubit_hamiltonian(mol)
    ansatz, n_params, hf_state, s_wires, d_wires = make_uccsd_ansatz(mol, n_qubits)

    # Prefer lightning.kokkos (GPU/ROCm) → lightning.qubit → default.qubit
    def _pick_dev():
        for name in ("lightning.kokkos", "lightning.qubit", "default.qubit"):
            try:
                return qml.device(name, wires=n_qubits)
            except Exception:
                pass
        raise RuntimeError("No PennyLane device available")
    dev = _pick_dev()

    def _wrap(theta):
        # map to (-pi, pi]; works for vector or batch
        return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    @qml.qnode(dev, interface="jax")
    def _circuit(theta):
        ansatz(_wrap(theta), wires=dev.wires)     # includes HF init
        return qml.expval(H)

    def cost(theta):
        theta = jnp.asarray(theta, dtype=jnp.float64)
        return float(_circuit(theta))

    meta = {
        "basis": basis_label,
        "n_qubits": n_qubits,
        "n_params": n_params,
        "n_electrons": mol.n_electrons,
        "n_orbitals": mol.n_orbitals,
        "symbols": list(mol.symbols),         # list[str]
        "charge": int(mol.charge),
        "geometry_bohr": jnp.asarray(mol.coordinates).tolist(),  # numeric 2D list
        "description": f"VQE cost E(theta) for {basis_label} at fixed geometry",
    }
    return cost, meta, H

# -------- 4) Optional: E(theta, x) re-building H(x) per call (slower) --------
def make_cost_theta_x(symbols: List[str], charge: int,
                      n_qubits: int, hf_state, s_wires, d_wires):
    """Return cost(theta, x_bohr) -> float energy (rebuilds H(x) each call)."""
    # Ensure integer dtype for init_state (what UCCSD wants)
    import numpy as _np
    hf_state_int = _np.asarray(hf_state, dtype=_np.int64)

    # Portable device (geometry changes each call so lightning doesn't help much here)
    dev = qml.device("default.qubit", wires=n_qubits)

    # Build the UCCSD ansatz exactly like in the fixed-geometry case
    ansatz = functools.partial(
        qml.UCCSD,
        init_state=hf_state_int,
        s_wires=s_wires,
        d_wires=d_wires,
    )

    def _wrap(theta):
        return (theta + jnp.pi) % (2.0 * jnp.pi) - jnp.pi

    @qml.qnode(dev, interface="jax")
    def _circ(theta, H):
        # No separate BasisState – UCCSD applies the HF init when init_state is provided
        ansatz(_wrap(theta), wires=range(n_qubits))
        return qml.expval(H)

    def cost(theta, x_bohr):
        mol = qml.qchem.Molecule(symbols, jnp.asarray(x_bohr, dtype=jnp.float64), charge=charge)
        H, _ = qml.qchem.molecular_hamiltonian(
            mol, method="pyscf", mapping="jordan_wigner"
        )
        return float(_circ(jnp.asarray(theta, dtype=jnp.float64), H))

    return cost


# -------------------- 5) Export H as JSON for porting -------------------------
def export_pauli_terms(H):
    """Return a JSON-serializable dict of Pauli terms for any PennyLane Operator H."""
    nq = len(H.wires)
    ps = pauli_sentence(H)  # works for qml.Hamiltonian or generic Operator

    # map various representations to a single character
    def op_to_char(op):
        # op might be 'X'/'Y'/'Z' or a PL operator with .name like 'PauliX'
        if isinstance(op, str):
            c = op.upper()
            if c in ("X", "Y", "Z"):
                return c
        name = getattr(op, "name", None)
        if name in ("PauliX", "X"):
            return "X"
        if name in ("PauliY", "Y"):
            return "Y"
        if name in ("PauliZ", "Z"):
            return "Z"
        raise ValueError(f"Unrecognized Pauli op representation: {op!r}")

    def masks(pstr: str):
        x = y = z = 0
        for i, ch in enumerate(pstr):
            bit = 1 << i
            if ch == "X":
                x |= bit
            elif ch == "Y":
                y |= bit
            elif ch == "Z":
                z |= bit
        return x, y, z

    terms = []
    for pword, coeff in ps.items():
        # Build a full-length Pauli string with 'I' on untouched wires.
        chars = ["I"] * nq
        # pword.items() yields (wire, op) where op may be 'X'/'Y'/'Z' or an operator object
        for w, op in pword.items():
            idx = int(w)  # Wire -> int
            chars[idx] = op_to_char(op)
        pstr = "".join(chars)
        xmask, ymask, zmask = masks(pstr)
        # coeff can be complex but should be real for Hermitian H; take real part safely
        coeff_real = float(jnp.real(jnp.asarray(coeff)))
        terms.append({
            "coeff": coeff_real,
            "pauli": pstr,
            "xmask": int(xmask),
            "ymask": int(ymask),
            "zmask": int(zmask),
        })

    return {"n_qubits": nq, "num_terms": len(terms), "terms": terms}


def save_pauli_json(path: str, H: qml.Hamiltonian):
    with open(path, "w") as f:
        json.dump(export_pauli_terms(H), f, indent=2)

# ------------------------------- 6) Demo --------------------------------------
if __name__ == "__main__":
    molname, basis = parse_cli()

    # Load & build
    mol, e_ref = load_molecule(molname, basis)
    basis_lbl = basis
    dataset_name = molname


    H, n_qubits = build_qubit_hamiltonian(mol)
    cost, meta, H = make_cost_fixed_geometry(mol, basis_lbl)
    ansatz, n_params, hf_state, s_wires, d_wires = make_uccsd_ansatz(mol, n_qubits)

    print("=== VQE cost definition (fixed geometry) ===")
    print(f"Molecule:        {dataset_name}")
    print(f"Basis:           {basis_lbl}")
    print(f"Electrons/Orbs:  {mol.n_electrons}e / {mol.n_orbitals}o")
    print(f"Qubits:          {meta['n_qubits']}")
    print(f"Params (UCCSD):  {meta['n_params']}")
    print(f"Reference (FCI): {e_ref:.10f} Ha")
    print("Cost signature:  E(theta) -> <psi(theta)|H|psi(theta)>")

    # Test the cost once (HF start: zeros usually best)
    theta0 = jnp.zeros(n_params, dtype=jnp.float64)
    e0 = cost(theta0)
    print(f"E(zeros) = {e0:.10f} Ha  (Δ vs FCI = {e0 - e_ref:+.6f} Ha)")

    # Export H so you can implement ⟨H⟩ in C++/Kokkos
    out_json = f"{molname.lower()}_{basis_lbl.lower().replace('-', '')}_pauli.json"
    save_pauli_json(out_json, H)
    print(f"Saved qubit Hamiltonian to: {out_json}")

    # (Optional) build E(theta, x) if you want geometry dependence later
    cost_theta_x = make_cost_theta_x(meta["symbols"], meta["charge"], n_qubits,
                                     hf_state, s_wires, d_wires)
    e_same = cost_theta_x(theta0, meta["geometry_bohr"])
    print(f"E(zeros, x_init) = {e_same:.10f} Ha  (should match E(zeros) within numerical tolerance)")
