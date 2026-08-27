#!/usr/bin/env python3
"""Declared PFDA-BS robustness design used by launch and analysis scripts."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from run_vqe_pfda_equal_count import pfda_evaluation_count


MOLECULES = ("HeH+", "H3+", "H4")
DIMENSIONS = {"HeH+": 3, "H3+": 8, "H4": 26}
SEEDS = tuple(range(5))


@dataclass(frozen=True)
class Variant:
    name: str
    factor: str
    level: str
    reference: str = "default"
    depth: int = 2
    beam_width: int = 100
    samples_per_child: int = 1
    sampling_strategy: str = "center-symmetric"
    ils_iterations: int = 1
    ils_contraction: float = 0.8

    def calls(self, dimension: int) -> int:
        count = pfda_evaluation_count(
            dimension,
            self.depth,
            self.beam_width,
            self.samples_per_child,
            self.ils_iterations,
        )
        return int(count["total"])

    def cli(self) -> list[str]:
        return [
            "--depth", str(self.depth),
            "--beam-width", str(self.beam_width),
            "--samples-per-child", str(self.samples_per_child),
            "--sampling-strategy", self.sampling_strategy,
            "--ils-iterations", str(self.ils_iterations),
            "--ils-contraction", str(self.ils_contraction),
        ]


VARIANTS = (
    Variant("default", "reference", "D=2, beta=100, m=1, maxiter=1"),
    Variant("depth_1", "maximum depth", "1", depth=1),
    Variant("depth_3", "maximum depth", "3", depth=3),
    Variant("beam_25", "beam width", "25", beam_width=25),
    Variant("beam_200", "beam width", "200", beam_width=200),
    Variant("samples_2", "samples per region", "2", samples_per_child=2),
    Variant("samples_4", "samples per region", "4", samples_per_child=4),
    Variant("ils_0", "local-search iterations", "0", ils_iterations=0),
    Variant("ils_2", "local-search iterations", "2", ils_iterations=2),
    Variant(
        "sampling_uniform",
        "sampling strategy",
        "uniform hypersphere",
        sampling_strategy="uniform-ball",
    ),
    Variant(
        "sampling_sobol",
        "sampling strategy",
        "scrambled Sobol hypersphere",
        sampling_strategy="sobol-ball",
    ),
    Variant(
        "contraction_05",
        "ILS contraction",
        "0.5 at two iterations",
        reference="ils_2",
        ils_iterations=2,
        ils_contraction=0.5,
    ),
    Variant(
        "contraction_095",
        "ILS contraction",
        "0.95 at two iterations",
        reference="ils_2",
        ils_iterations=2,
        ils_contraction=0.95,
    ),
)
VARIANT_BY_NAME = {variant.name: variant for variant in VARIANTS}


def molecule_slug(molecule: str) -> str:
    return molecule.replace("+", "plus")


def jobs() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant in VARIANTS:
        for molecule in MOLECULES:
            for seed in SEEDS:
                rows.append(
                    {
                        "variant": variant.name,
                        "molecule": molecule,
                        "molecule_slug": molecule_slug(molecule),
                        "seed": seed,
                        "evaluations": variant.calls(DIMENSIONS[molecule]),
                    }
                )
    # Largest-processing-time-first ordering gives the six dynamic workers a
    # balanced queue while preserving deterministic, resumable job identities.
    rows.sort(
        key=lambda row: (
            -int(row["evaluations"]),
            str(row["variant"]),
            str(row["molecule"]),
            int(row["seed"]),
        )
    )
    return rows


def serializable_protocol() -> dict[str, object]:
    return {
        "protocol": "PFDA-BS-hyperparameter-robustness-v1",
        "molecules": list(MOLECULES),
        "dimensions": DIMENSIONS,
        "seeds": list(SEEDS),
        "variants": [asdict(variant) for variant in VARIANTS],
        "comparison": {
            "native_final": "Each structural configuration runs to its native call count.",
            "fixed_call": "All configurations are also compared at the smallest native count for each molecule.",
            "geometry": "Canonical hyperspheres are held fixed because geometry defines the PFDA-BS algorithm rather than a tunable hyperparameter.",
        },
    }
