#!/usr/bin/env python3
"""Checkpointed 24/30-qubit VQE statevector feasibility campaign.

The target is LiH/cc-pVDZ with the doubly occupied Li core orbital frozen.
Selecting two active electrons in 15 spatial orbitals produces a real
30-qubit molecular Hamiltonian while keeping the UCCSD parameter count small
enough for a multi-hour preflight.  Every completed objective call is
atomically committed, so a killed worker resumes at the first missing
candidate instead of repeating successful evaluations.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import socket
import time
from pathlib import Path
from typing import Any


ANGSTROM_TO_BOHR = 1.0 / 0.529177210903


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def signature(args: argparse.Namespace) -> str:
    fields = {
        "protocol": "LiH-cc-pVDZ-active-space-statevector-v1",
        "basis": args.basis,
        "bond_angstrom": args.bond_angstrom,
        "active_electrons": args.active_electrons,
        "active_orbitals": args.active_orbitals,
        "precision": args.precision,
        "seed": args.seed,
        "candidate_count": args.candidate_count,
        "domain_scale": args.domain_scale,
    }
    encoded = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def candidate_indices(args: argparse.Namespace) -> list[int]:
    if args.candidate_indices:
        values = [int(value) for value in args.candidate_indices.split(",") if value]
    else:
        values = list(range(args.worker_id, args.candidate_count, args.worker_count))
    if any(value < 0 or value >= args.candidate_count for value in values):
        raise ValueError("candidate index lies outside [0, candidate_count)")
    return values


def make_candidates(count: int, dimension: int, seed: int, half_width: float):
    import numpy as np
    from scipy.stats import qmc

    points = np.zeros((count, dimension), dtype=np.float64)
    if count > 1:
        exponent = int(math.ceil(math.log2(count - 1)))
        sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
        unit = sampler.random_base2(exponent)[: count - 1]
        points[1:] = qmc.scale(unit, -half_width, half_width)
    return points


def choose_device(qml, np, n_qubits: int, precision: str):
    requested = os.environ.get("VQE_DEVICE", "").strip()
    names = [requested] if requested else [
        "lightning.kokkos",
        "lightning.gpu",
        "default.qubit",
    ]
    errors: list[str] = []
    c_dtype = np.complex64 if precision == "complex64" else np.complex128
    for name in names:
        try:
            kwargs = {"c_dtype": c_dtype} if name.startswith("lightning.") else {}
            return qml.device(name, wires=n_qubits, **kwargs), errors, name
        except Exception as exc:  # pragma: no cover - depends on remote plugins
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
    raise RuntimeError("no statevector device available; " + " | ".join(errors))


def exact_active_space_reference(
    np,
    core_constant,
    one_mo,
    two_mo_physicist,
    active_electrons: int,
    multiplicity: int,
) -> float:
    """Return the exact FCI energy in the same active-space Hamiltonian.

    ``_pyscf_integrals`` returns its two-electron tensor in physicists'
    notation for PennyLane.  PySCF's direct FCI solver expects chemists'
    notation, so the conversion is explicitly reversed here.  For the
    two-electron 24/30-qubit controls the determinant space is tiny even
    though the dense qubit statevector is large.
    """

    from pyscf import fci

    spin = int(multiplicity) - 1
    if (active_electrons + spin) % 2 != 0:
        raise ValueError("active electron count and multiplicity are inconsistent")
    n_alpha = (active_electrons + spin) // 2
    n_beta = active_electrons - n_alpha
    two_mo_chemist = np.swapaxes(two_mo_physicist, 1, 3)
    energy, _ = fci.direct_spin1.kernel(
        np.asarray(one_mo),
        np.asarray(two_mo_chemist),
        int(one_mo.shape[0]),
        (n_alpha, n_beta),
        ecore=float(np.asarray(core_constant).reshape(-1)[0]),
    )
    return float(energy)


def build_active_space_hamiltonian(
    qml,
    np,
    molecule,
    active_electrons: int,
    active_orbitals: int,
):
    """Build and validate the requested PySCF active-space Hamiltonian.

    PennyLane 0.41.1's high-level ``molecular_hamiltonian`` path reported the
    full 38-wire LiH/cc-pVDZ register for the 12- and 15-orbital requests on
    Grid'5000.  Constructing from the active integral tensors makes the
    register reduction explicit and lets us validate it before any large
    statevector is allocated.
    """

    expected_qubits = 2 * active_orbitals
    coordinates = np.asarray(molecule.coordinates, dtype=float).reshape(-1)
    core_constant, one_mo, two_mo = qml.qchem.openfermion_pyscf._pyscf_integrals(
        molecule.symbols,
        coordinates,
        molecule.charge,
        molecule.mult,
        molecule.basis_name,
        active_electrons,
        active_orbitals,
    )

    expected_one_shape = (active_orbitals, active_orbitals)
    expected_two_shape = (active_orbitals,) * 4
    one_shape = tuple(int(value) for value in one_mo.shape)
    two_shape = tuple(int(value) for value in two_mo.shape)
    if one_shape != expected_one_shape or two_shape != expected_two_shape:
        raise RuntimeError(
            "active-space integral reduction failed: "
            f"requested {active_orbitals} orbitals, "
            f"one_mo shape={one_shape}, two_mo shape={two_shape}"
        )

    fermionic = qml.qchem.fermionic_observable(core_constant, one_mo, two_mo)
    hamiltonian = qml.jordan_wigner(fermionic, tol=1.0e-10).simplify()
    wire_indices = [int(wire) for wire in hamiltonian.wires]
    invalid_wires = [
        wire for wire in wire_indices if wire < 0 or wire >= expected_qubits
    ]
    if invalid_wires:
        raise RuntimeError(
            "active-space Hamiltonian escaped the requested register: "
            f"expected wires 0..{expected_qubits - 1}, "
            f"observed invalid wires={invalid_wires[:8]}"
        )

    diagnostics = {
        "builder": "explicit-pyscf-active-integrals",
        "one_mo_shape": list(one_shape),
        "two_mo_shape": list(two_shape),
        "operator_wire_count": len(wire_indices),
        "operator_wire_min": min(wire_indices) if wire_indices else None,
        "operator_wire_max": max(wire_indices) if wire_indices else None,
    }
    if os.environ.get("VQE_COMPUTE_EXACT_REFERENCE", "0") == "1":
        diagnostics["exact_active_space_energy_hartree"] = exact_active_space_reference(
            np,
            core_constant,
            one_mo,
            two_mo,
            active_electrons,
            molecule.mult,
        )
    return hamiltonian, expected_qubits, diagnostics


def run_worker(args: argparse.Namespace) -> int:
    indices = candidate_indices(args)
    run_signature = signature(args)
    output_root = args.output_root
    candidate_root = output_root / "candidates"

    pending: list[int] = []
    for index in indices:
        path = candidate_root / f"candidate_{index:04d}.json"
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("status") == "OK":
                    if payload.get("signature") != run_signature:
                        raise RuntimeError(f"signature mismatch in {path}")
                    print(f"skip completed candidate={index}", flush=True)
                    continue
            except json.JSONDecodeError:
                pass
        pending.append(index)

    if not pending:
        print("all assigned candidates already checkpointed", flush=True)
        return 0

    heartbeat = output_root / "heartbeats" / f"worker_{args.worker_id:02d}.json"
    atomic_json(
        heartbeat,
        {
            "status": "INITIALIZING",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "worker_id": args.worker_id,
            "pending": pending,
            "updated_at": time.time(),
        },
    )

    import numpy as np
    import jax
    import jax.numpy as jnp
    import pennylane as qml

    use_x64 = args.precision == "complex128"
    jax.config.update("jax_enable_x64", use_x64)
    parameter_dtype = jnp.float64 if use_x64 else jnp.float32
    state_bytes = 16 if use_x64 else 8

    build_start = time.perf_counter()
    coordinates = np.array(
        [[0.0, 0.0, 0.0], [0.0, 0.0, args.bond_angstrom * ANGSTROM_TO_BOHR]],
        dtype=float,
    )
    molecule = qml.qchem.Molecule(
        ["Li", "H"],
        coordinates,
        charge=0,
        mult=1,
        basis_name=args.basis,
    )
    hamiltonian, n_qubits, hamiltonian_diagnostics = build_active_space_hamiltonian(
        qml,
        np,
        molecule,
        args.active_electrons,
        args.active_orbitals,
    )
    expected_qubits = 2 * args.active_orbitals
    if int(n_qubits) != expected_qubits:
        raise RuntimeError(f"expected {expected_qubits} qubits, got {n_qubits}")

    hf_state = qml.qchem.hf_state(args.active_electrons, n_qubits)
    singles, doubles = qml.qchem.excitations(args.active_electrons, n_qubits)
    single_wires, double_wires = qml.qchem.excitations_to_wires(singles, doubles)
    n_params = len(single_wires) + len(double_wires)
    device, device_errors, device_name = choose_device(
        qml, np, n_qubits, args.precision
    )
    device_label = (
        getattr(device, "short_name", None)
        or getattr(device, "name", None)
        or device.__class__.__name__
    )

    if device_name.startswith("lightning."):
        # Lightning executes the tape in compiled C++/CUDA or C++/Kokkos.
        # Avoid wrapping the complete UCCSD circuit and tens of thousands of
        # Hamiltonian terms in one XLA graph: that compiler graph, rather than
        # the 8-GiB complex64 statevector, exhausted the previous 30-qubit
        # preflight before useful GPU work began.
        @qml.qnode(device, interface=None, diff_method=None)
        def energy(parameters):
            qml.UCCSD(
                parameters,
                wires=range(n_qubits),
                s_wires=single_wires,
                d_wires=double_wires,
                init_state=hf_state,
            )
            return qml.expval(hamiltonian)

        @qml.qnode(device, interface=None, diff_method=None)
        def hartree_fock_energy():
            # Candidate 0 is exactly the zero-amplitude UCCSD reference.
            # Preparing its Hartree--Fock determinant directly is
            # mathematically identical to applying every UCCSD excitation at
            # angle zero, while avoiding hundreds of no-op state updates in
            # the 30-qubit backend preflight.
            qml.BasisState(hf_state, wires=range(n_qubits))
            return qml.expval(hamiltonian)

        native_parameter_dtype = np.float64 if use_x64 else np.float32

        def evaluate(point):
            native_point = np.asarray(point, dtype=native_parameter_dtype)
            if not np.any(native_point):
                return float(hartree_fock_energy())
            return float(energy(native_point))

        execution_mode = "native-lightning"
    else:
        @qml.qnode(device, interface="jax")
        def energy(parameters):
            qml.UCCSD(
                parameters,
                wires=range(n_qubits),
                s_wires=single_wires,
                d_wires=double_wires,
                init_state=hf_state,
            )
            return qml.expval(hamiltonian)

        compiled_energy = jax.jit(energy)

        def evaluate(point):
            return float(
                jax.device_get(
                    compiled_energy(jnp.asarray(point, dtype=parameter_dtype))
                )
            )

        execution_mode = "jax-monolithic-jit"
    build_time = time.perf_counter() - build_start
    half_width = min(
        args.domain_scale * math.sqrt(3.0 / max(1, n_params)), math.pi / 8.0
    )
    candidates = make_candidates(
        args.candidate_count, n_params, args.seed, half_width
    )
    term_count = len(hamiltonian.terms()[0])

    metadata = {
        "protocol": "LiH-cc-pVDZ-active-space-statevector-v1",
        "signature": run_signature,
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "worker_id": args.worker_id,
        "worker_count": args.worker_count,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "pennylane": qml.__version__,
        "jax_devices": [str(item) for item in jax.devices()],
        "device": str(device_label),
        "device_requested": device_name,
        "device_selection_errors": device_errors,
        "execution_mode": execution_mode,
        "molecule": "LiH",
        "basis": args.basis,
        "bond_angstrom": args.bond_angstrom,
        "active_electrons": args.active_electrons,
        "active_orbitals": args.active_orbitals,
        "n_qubits": int(n_qubits),
        "n_parameters": n_params,
        "single_excitations": len(single_wires),
        "double_excitations": len(double_wires),
        "hamiltonian_terms": term_count,
        "hamiltonian_construction": hamiltonian_diagnostics,
        "precision": args.precision,
        "parameter_dtype": str(parameter_dtype),
        "raw_statevector_bytes": (1 << int(n_qubits)) * state_bytes,
        "raw_statevector_gib": ((1 << int(n_qubits)) * state_bytes) / (1 << 30),
        "candidate_count": args.candidate_count,
        "assigned_candidates": indices,
        "domain_half_width": half_width,
        "candidate_generator": "zero then scrambled Sobol(seed) in bounded trust box",
        "hamiltonian_and_qnode_build_time_s": build_time,
    }
    atomic_json(
        output_root / "metadata" / f"worker_{args.worker_id:02d}.json", metadata
    )

    for index in pending:
        path = candidate_root / f"candidate_{index:04d}.json"
        point = candidates[index]
        candidate_hash = hashlib.sha256(
            np.asarray(point, dtype=np.float64).tobytes(order="C")
        ).hexdigest()
        atomic_json(
            heartbeat,
            {
                "status": "EVALUATING",
                "host": socket.gethostname(),
                "pid": os.getpid(),
                "worker_id": args.worker_id,
                "candidate": index,
                "updated_at": time.time(),
            },
        )
        started = time.perf_counter()
        value = evaluate(point)
        elapsed = time.perf_counter() - started
        if not math.isfinite(value):
            raise RuntimeError(f"candidate {index} returned non-finite energy {value}")
        exact_reference = hamiltonian_diagnostics.get(
            "exact_active_space_energy_hartree"
        )
        if exact_reference is not None:
            variational_tolerance = float(
                os.environ.get(
                    "VQE_VARIATIONAL_TOLERANCE_HA",
                    "0.005" if args.precision == "complex64" else "1e-8",
                )
            )
            if value < float(exact_reference) - variational_tolerance:
                raise RuntimeError(
                    "variational-bound violation: "
                    f"measured {value:.12f} Ha below exact active-space "
                    f"reference {float(exact_reference):.12f} Ha by more than "
                    f"{variational_tolerance:.6g} Ha"
                )
        atomic_json(
            path,
            {
                "status": "OK",
                "signature": run_signature,
                "candidate_index": index,
                "candidate_sha256_float64": candidate_hash,
                "energy_hartree": value,
                "evaluation_wall_time_s": elapsed,
                "includes_jit_compilation": index == pending[0],
                "parameter_l2_norm": float(np.linalg.norm(point)),
                "max_absolute_parameter": float(np.max(np.abs(point))),
                "host": socket.gethostname(),
                "worker_id": args.worker_id,
                "device": str(device_label),
                "precision": args.precision,
                "execution_mode": execution_mode,
                "n_qubits": int(n_qubits),
                "n_parameters": n_params,
                "exact_active_space_energy_hartree": exact_reference,
                "completed_at": time.time(),
            },
        )
        print(
            f"checkpoint candidate={index} energy={value:.12f} wall_s={elapsed:.6f}",
            flush=True,
        )

    atomic_json(
        heartbeat,
        {
            "status": "DONE",
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "worker_id": args.worker_id,
            "updated_at": time.time(),
        },
    )
    return 0


def aggregate(root: Path) -> int:
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("runs/*/*/candidates/candidate_*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        payload["relative_path"] = path.relative_to(root).as_posix()
        payload["profile"] = path.parts[-4]
        records.append(payload)

    fields = sorted({key for record in records for key in record})
    csv_path = root / "candidate_results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(csv_path)

    successful = [record for record in records if record.get("status") == "OK"]
    by_profile: dict[str, dict[str, Any]] = {}
    for record in successful:
        path_parts = Path(str(record["relative_path"])).parts
        key = f"{path_parts[1]}/{path_parts[2]}"
        entry = by_profile.setdefault(
            key, {"successful_candidates": 0, "best_energy_hartree": math.inf}
        )
        entry["successful_candidates"] += 1
        entry["best_energy_hartree"] = min(
            float(entry["best_energy_hartree"]), float(record["energy_hartree"])
        )
    atomic_json(
        root / "campaign_summary.json",
        {
            "protocol": "LiH-cc-pVDZ-active-space-statevector-v1",
            "recorded_candidates": len(records),
            "successful_candidates": len(successful),
            "profiles": by_profile,
            "updated_at": time.time(),
        },
    )
    print(json.dumps(by_profile, sort_keys=True), flush=True)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--campaign-root", type=Path)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--worker-count", type=int, default=1)
    parser.add_argument("--candidate-indices", default="")
    parser.add_argument("--candidate-count", type=int, default=19)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--basis", default="cc-pvdz")
    parser.add_argument("--bond-angstrom", type=float, default=1.57)
    parser.add_argument("--active-electrons", type=int, default=2)
    parser.add_argument("--active-orbitals", type=int, default=15)
    parser.add_argument("--precision", choices=("complex64", "complex128"), default="complex64")
    parser.add_argument("--domain-scale", type=float, default=0.30)
    args = parser.parse_args()
    if args.aggregate_only:
        if args.campaign_root is None:
            parser.error("--aggregate-only requires --campaign-root")
    elif args.output_root is None:
        parser.error("worker mode requires --output-root")
    if args.worker_count < 1 or not 0 <= args.worker_id < args.worker_count:
        parser.error("worker id/count are inconsistent")
    if args.active_orbitals < 1 or args.active_electrons < 1:
        parser.error("active-space sizes must be positive")
    return args


def main() -> int:
    args = parse_args()
    if args.aggregate_only:
        return aggregate(args.campaign_root)
    try:
        return run_worker(args)
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "host": socket.gethostname(),
            "worker_id": args.worker_id,
            "updated_at": time.time(),
        }
        atomic_json(
            args.output_root / "failures" / f"worker_{args.worker_id:02d}_{int(time.time())}.json",
            failure,
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
