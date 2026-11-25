# Reproducibility bundle

This directory contains the PFDA-BS VQE results and the scripts used for the
experiments. You can check the saved results or repeat the experiments:

1. **Check the saved results:** recompute comparisons, robustness results,
   LUMI-G speedups, efficiencies and energy errors from the CSV/JSON files.
   This works on CPU.
2. **Repeat the experiments:** run the Python drivers through the Grid'5000
   OAR or LUMI-G Slurm job scripts.

Run the dependency-free verification from the repository root:

```bash
python reproducibility/verify_results.py
```

The command validates checksums and should end with
`reproducibility verification: OK`.

To check only the Grid'5000 and LUMI-G results, use:

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
    ├── raw_evidence/ Hash-indexed excerpts from archived execution records
    └── results/      Published scaling and solution-quality coordinates
```

## Results and execution records

### Grid'5000 black-box equal-call comparison

`grid5000/results/black_box_equal_call/runs_long.csv` is the retained run-level
measurement table.  It contains objective-call checkpoints, energies, FCI
references, absolute errors, and wall times.  The reported comparison uses the
median stochastic-baseline error with deterministic PFDA-BS at the final
matched checkpoint.  The verifier reproduces 25 wins, 14 ties, and one loss at a
`1e-7` Ha tie tolerance.

The execution records keep the Grid'5000 job identifiers and Ecotaxe
hostnames. Personal filesystem paths have been replaced.

### Grid'5000 knowledge-driven comparison

`grid5000/results/knowledge_driven/matched_quantum_work_curves.csv` and
`quantum_aware_summary.csv` are the retained aggregate outputs.  The complete
eight-molecule result table is in `paper_final_errors.csv`; the verifier checks
all retained unrounded conditions against that table and reproduces
8 wins, 15 ties, and one loss.

The archived experiment data do not include the individual H5/LiH
quantum-aware run logs. Those two rows are therefore identified as reported
aggregate values rather than raw-log values. Rerun support remains present in `grid5000/code/` and
`grid5000/oar/`.

### Robustness experiment

The sensitivity experiment ran on one NVIDIA RTX 3050 Laptop GPU. Its
results are stored under `local/`.
`runs.csv` contains all 195 measurements; `paired_effects.csv` contains the 12
one-factor perturbations.  The Grid'5000 robustness launcher is also included
so that the same resumable design can be reproduced on an Ecotaxe allocation.

### LUMI-G experiments

The LUMI-G job scripts are in `scripts/lumi/`. The result tables are in
`lumi_g/results/`, where the verifier checks their numerical consistency.
`lumi_g/raw_evidence/` records source-file hashes and verbatim result-bearing
excerpts from the archived execution records. It directly supports five of
the eight scaling timestamps and seven of the nine solution-quality energies,
and includes scheduler completion records for the 8,000-rank endpoint and the
HF, OH-, and LiH quality runs.

Detailed source traces are not available for every reported coordinate. The
evidence index therefore records only archived source evidence and labels each
entry as `exact`, `energy_only`, or `scheduler_only`. The remaining coordinates
are retained in the result tables and documented explicitly in
`lumi_g/README.md`.

## Running the experiments again

You need access to the site's container images, datasets and modules, and
an account for submitting jobs. Set `PROJECT_ROOT`, `IMAGE` and `RESULT_ROOT`
for your environment. The job scripts save result summaries and mark finished
runs with `RUN_SUCCESS`, so later jobs can continue from the remaining work.

The repository keeps the CSV measurements needed to reproduce the paper
tables. The larger per-evaluation logs are generated when you repeat the
experiments and are saved under `RESULT_ROOT`.
