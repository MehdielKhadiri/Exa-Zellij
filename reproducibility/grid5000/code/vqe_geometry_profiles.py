#!/usr/bin/env python3
"""Controlled low-qubit geometries for VQE domain studies."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ANGSTROM_TO_BOHR = 1.8897261254578281


@dataclass(frozen=True)
class GeometryProfile:
    molecule: object
    fci_energy: float
    metadata: dict[str, object]


def linear_h4(spacing_angstrom: float, basis: str = "STO-3G") -> GeometryProfile:
    """Return an equally spaced linear H4 chain and an exact PySCF reference."""
    if spacing_angstrom <= 0.0:
        raise ValueError("H4 spacing must be positive")

    import pennylane as qml
    from pyscf import fci, gto, scf

    z_angstrom = (np.arange(4, dtype=np.float64) - 1.5) * spacing_angstrom
    atom = [("H", (0.0, 0.0, float(z))) for z in z_angstrom]
    pyscf_mol = gto.M(
        atom=atom,
        basis=basis.lower(),
        unit="Angstrom",
        charge=0,
        spin=0,
        verbose=0,
    )
    mean_field = scf.RHF(pyscf_mol)
    mean_field.conv_tol = 1e-12
    mean_field.max_cycle = 200
    hf_energy = float(mean_field.kernel())
    if not mean_field.converged:
        mean_field = mean_field.newton()
        mean_field.conv_tol = 1e-12
        hf_energy = float(mean_field.kernel())
    if not mean_field.converged:
        raise RuntimeError(f"RHF did not converge for linear H4 at R={spacing_angstrom}")

    fci_solver = fci.FCI(mean_field)
    fci_solver.conv_tol = 1e-12
    fci_energy = float(fci_solver.kernel()[0])

    coordinates_bohr = np.zeros((4, 3), dtype=np.float64)
    coordinates_bohr[:, 2] = z_angstrom * ANGSTROM_TO_BOHR
    molecule = qml.qchem.Molecule(
        ["H", "H", "H", "H"],
        coordinates_bohr,
        charge=0,
        mult=1,
        basis_name=basis.lower(),
        unit="bohr",
        name=f"H4_linear_R{spacing_angstrom:.3f}",
    )
    return GeometryProfile(
        molecule=molecule,
        fci_energy=fci_energy,
        metadata={
            "geometry_profile": "linear-h4",
            "spacing_angstrom": float(spacing_angstrom),
            "pyscf_hf_energy": hf_energy,
            "pyscf_fci_energy": fci_energy,
            "correlation_energy_ha": hf_energy - fci_energy,
            "coordinates_bohr": coordinates_bohr.tolist(),
        },
    )
