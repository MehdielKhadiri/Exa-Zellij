# LUMI-G result provenance

The executable, HIP configuration, GPU-selection wrapper, and SLURM template
used for LUMI-G are maintained in `../../scripts/lumi/` and documented in
`../../docs/lumi-g.md`.

This directory contains the numerical coordinates used by the manuscript:

- `results/strong_scaling.csv` records the four reference-to-target strong
  scaling pairs, including node/GPU counts, wall times, speedups, and
  efficiencies.
- `results/solution_quality.csv` records the nine-molecule distributed
  solution-quality table, including the FCI values needed to recompute both
  absolute and relative errors.
- `raw_evidence/raw_log_evidence.csv` indexes the matching files found in the
  recovered LUMI-G backup, including their original sizes and SHA-256 hashes.
- `raw_evidence/log_excerpts.txt` retains the exact terminal lines needed to
  connect those source logs to the published energies and timings.

`../verify_results.py` recomputes every speedup and efficiency, checks the
eight-GPU-per-node mapping, confirms the 8,000-GPU/60.98% endpoint, and
recomputes all nine energy errors.

## Provenance boundary

The recovered backup is approximately 20 GB and contains software
environments, caches, and many exploratory/failed runs, so it is neither
appropriate nor necessary to commit the whole home-directory image.  Instead,
the evidence index names each relevant backup-relative source file, records
its byte size and SHA-256 digest, and preserves its result-bearing lines.

The recovered files exactly support the 1-to-10-node and
100-to-1,000-node (`maxiter=10`) scaling pairs, the 100-node reference of the
`maxiter=100` pair, and six solution-quality energies.  The 8,000-rank
`maxiter=100` scheduler stdout confirms successful execution on 1,000 nodes,
but its detailed run log containing the 1,709.20-second internal timing was not
recovered.  The 10-to-100-node scaling pair and the remaining three quality
rows likewise have no matching source log in this backup.  Those values are a
faithful transcription of the final manuscript and remain labeled
**paper-derived**, not raw-log-derived.
