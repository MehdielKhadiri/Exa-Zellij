#!/usr/bin/env python3
"""Pre-stage qchem datasets before timed optimizer runs."""

from __future__ import annotations

import argparse

from vqe_cost_reference import load_molecule


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--molecules", nargs="+", default=["HeH+", "H3+", "H4"])
    parser.add_argument("--basis", default="STO-3G")
    args = parser.parse_args()
    for molecule in args.molecules:
        mol, fci = load_molecule(molecule, args.basis)
        print(
            f"staged molecule={molecule} basis={args.basis} "
            f"electrons={mol.n_electrons} orbitals={mol.n_orbitals} fci={float(fci):.12f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
