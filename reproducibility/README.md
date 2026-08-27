# Reproducibility bundle

This directory connects the public implementation to the experiments reported
for the PFDA-BS VQE study.  It supports two different verification levels:

1. **Artifact verification** recomputes the reported comparison counts,
   robustness statements, LUMI-G speedups, efficiencies, and energy errors
   from the included CSV/JSON files without using an accelerator.
2. **Experimental reproduction** reruns the objective evaluations with the
   included Python drivers and Grid'5000 OAR or LUMI-G SLURM launch material.

Run the dependency-free verification from the repository root:

```bash
python reproducibility/verify_results.py
```

The command validates checksums and should end with
`reproducibility verification: OK`.

For a curated archive containing only Grid'5000 and LUMI-G evidence, use:

```bash
python reproducibility/verify_results.py --scopes grid5000,lumi_g
```

## What is included

```text
reproducibility/
├── verify_results.py
├── SHA256SUMS
├── grid5000/
│   ├── code/       Sequential VQE baselines, PFDA-BS driver, and analyzers
│   ├── oar/        Resumable Nantes/Ecotaxe launch, staging, and worker files
│   └── results/    Run-level and aggregate sequential result tables
├── local/
│   └── results/    Complete 195-run single-GPU robustness experiment
└── lumi_g/
    ├── README.md
    ├── raw_evidence/ Hash-indexed excerpts from the recovered scheduler backup
    └── results/      Published scaling and solution-quality coordinates
```

## Evidence and provenance

### Grid'5000 black-box equal-call comparison

`grid5000/results/black_box_equal_call/runs_long.csv` is the retained run-level
measurement table.  It contains objective-call checkpoints, energies, FCI
references, absolute errors, and wall times.  The paper result compares the
median stochastic-baseline error with deterministic PFDA-BS at the final
matched checkpoint.  The verifier recovers 25 wins, 14 ties, and one loss at a
`1e-7` Ha tie tolerance.

The original execution manifests are included after replacing only personal
filesystem locations.  Grid'5000 job identifiers and Ecotaxe hostnames are
retained because they are useful provenance, not credentials.

### Grid'5000 knowledge-driven comparison

`grid5000/results/knowledge_driven/matched_quantum_work_curves.csv` and
`quantum_aware_summary.csv` are the retained aggregate outputs.  The complete
eight-molecule paper table is in `paper_final_errors.csv`; the verifier checks
all locally retained unrounded raw conditions against that table and recovers
8 wins, 15 ties, and one loss.

The local archive does not contain the individual H5/LiH quantum-aware run
logs.  Those two rows are therefore explicitly labeled as paper-table values,
not raw logs.  Rerun support remains present in `grid5000/code/` and
`grid5000/oar/`.

### Robustness experiment

The sensitivity experiment was executed on one NVIDIA RTX 3050 Laptop GPU,
not on Grid'5000.  Its truthful provenance is therefore under `local/`.
`runs.csv` contains all 195 measurements; `paired_effects.csv` contains the 12
one-factor perturbations.  The Grid'5000 robustness launcher is also included
so that the same resumable design can be reproduced on an Ecotaxe allocation.

### LUMI-G experiments

The existing LUMI-G scripts remain under `scripts/lumi/`.  The numerical
coordinates used by the manuscript are transcribed into machine-readable CSVs
under `lumi_g/results/` and are internally rechecked by the verifier.  A
recovered LUMI-G home-directory backup was audited after the initial assembly.
`lumi_g/raw_evidence/` records exact source-file hashes and verbatim terminal
excerpts for the source logs that were found.  It directly supports five of
the eight scaling timestamps and seven of the nine solution-quality energies,
and includes scheduler completion records for the 8,000-rank endpoint and the
HF, OH-, and LiH quality runs.

The backup does not contain every specific log used by the final manuscript.
Unsupported coordinates therefore remain explicitly labeled
**paper-derived**; the evidence index states `exact`, `energy_only`,
`scheduler_only`, or `not_recovered` rather than silently mixing provenance.
See `lumi_g/README.md`.

## Reproduction boundary

The repository does not redistribute Grid'5000 or LUMI container images,
PennyLane datasets, scheduler accounts, or site modules.  These are
site-controlled dependencies.  Launchers accept `PROJECT_ROOT`, `IMAGE`, and
`RESULT_ROOT` overrides and write resumable `RUN_SUCCESS` markers plus
machine-readable summaries.  Personal keys, credentials, access tokens, and
home-directory identities are not included.

The large per-evaluation traces are omitted from Git because the curated
run-level CSVs contain every value needed to reconstruct the reported tables.
Reruns generate the full traces in their chosen `RESULT_ROOT`.
