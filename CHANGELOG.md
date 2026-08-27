# Changelog

## 0.2.0 — 2026-08-27

- Added the Grid'5000 equal-call, knowledge-driven, and robustness experiment
  drivers with resumable OAR staging and worker launchers.
- Added curated run-level/aggregate Grid'5000 results and the complete local
  195-run robustness dataset.
- Added machine-readable LUMI-G scaling and nine-molecule solution-quality
  coordinates plus hash-indexed excerpts from archived execution records.
  Evidence levels distinguish detailed-log, energy-only, and scheduler-only
  provenance.
- Added a dependency-free verifier that reconstructs the central comparison
  counts and checks result checksums and numerical consistency.
- Updated the VQE evaluator for explicit precision, unwrapped-domain, and
  state-metric support required by the reproduced experiments.
- Added citation metadata and expanded platform-specific rerun documentation.

## 0.1.0 — 2026-07-29

- Assembled the configurable FBD and FBO implementations in one focused tree.
- Restored the implemented pybind11 VQE objective path to the configurable
  optimizer driver.
- Added CUDA/HIP build selection, LUMI-G launch material, examples, and
  dependency-light release checks.
