# LUMI-G result provenance

The executable, HIP configuration, GPU-selection wrapper, and SLURM template
used for LUMI-G are maintained in `../../scripts/lumi/` and documented in
`../../docs/lumi-g.md`.

This directory contains the reported numerical results:

- `results/strong_scaling.csv` records the four reference-to-target strong
  scaling pairs, including node/GPU counts, wall times, speedups, and
  efficiencies.
- `results/solution_quality.csv` records the nine-molecule distributed
  solution-quality table, including the FCI values needed to recompute both
  absolute and relative errors.
- `raw_evidence/raw_log_evidence.csv` indexes the corresponding archived
  execution records, including their original sizes and SHA-256 hashes.
- `raw_evidence/log_excerpts.txt` retains the exact terminal lines needed to
  connect those source logs to the published energies and timings.

`../verify_results.py` recomputes every speedup and efficiency, checks the
eight-GPU-per-node mapping, confirms the 8,000-GPU/60.98% endpoint, and
recomputes all nine energy errors.

## Provenance boundary

The complete experiment snapshot also contains software environments, caches,
container images, and exploratory runs that are not required to validate the
reported results. This archive therefore retains a compact evidence index: it
names each relevant source record, stores its byte size and SHA-256 digest, and
preserves the result-bearing lines.

The archived detailed logs exactly support the 1-to-10-node and
100-to-1,000-node (`maxiter=10`) scaling pairs, the 100-node reference of the
`maxiter=100` pair, and seven solution-quality energies.  Scheduler records
also corroborate the HF, OH-, and LiH execution envelopes.  The 8,000-rank
`maxiter=100` scheduler stdout confirms successful execution on 1,000 nodes,
although its detailed internal-timing trace is not available in the archived
experiment data. The same applies to the detailed
10-to-100-node scaling traces and the OH-/HF final-energy traces. Their
reported coordinates remain available in the machine-readable result tables,
while the evidence index makes the distinction between direct detailed-log,
energy-only, and scheduler-only support explicit.
