# LUMI-G results

The executable, HIP configuration, GPU-selection wrapper, and SLURM template
used for LUMI-G are maintained in `../../scripts/lumi/` and documented in
`../../docs/lumi-g.md`.

This directory contains the result tables and available execution logs:

- `results/strong_scaling.csv` records the four reference-to-target strong
  scaling pairs, including node/GPU counts, wall times, speedups, and
  efficiencies.
- `results/solution_quality.csv` records the nine-molecule distributed
  solution-quality table, including the FCI values needed to recompute both
  absolute and relative errors.
- `raw_evidence/raw_log_evidence.csv` indexes the corresponding archived
  execution records, including their original sizes and SHA-256 hashes.
- `raw_evidence/log_excerpts.txt` retains the exact terminal lines needed to
  identify the published energies and timings in those logs.

`../verify_results.py` recomputes every speedup and efficiency, checks the
eight-GPU-per-node mapping, confirms the 8,000-GPU/60.98% endpoint, and
recomputes all nine energy errors.

## Available logs

The log index lists the source files, their sizes and their SHA-256
checksums. The excerpts keep the lines that report results and timings.
Large caches, environments and container images are stored outside this
repository.

Detailed logs are available for the 1-to-10-node comparison, the
100-to-1,000-node comparison with `maxiter=10`, the 100-node run with
`maxiter=100`, and seven of the nine molecular energies. Scheduler records
also show that the HF, OH- and LiH jobs completed.

For the 8,000-process run with `maxiter=100`, the scheduler output confirms
completion on 1,000 nodes, but the detailed timing log is missing. Detailed
logs are also missing for the 10-to-100-node comparison and the final OH-
and HF energies. The reported values remain in the CSV tables. The log
index identifies whether each available record contains detailed results,
an energy value, or only the scheduler output.
