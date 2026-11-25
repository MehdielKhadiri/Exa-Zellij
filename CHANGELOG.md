# Changelog

## 0.3.0 — in development

- Renamed the project and source distribution to Fractal-Based Optimizers
  (`fractal-based-optimizers`). Existing `fqo` imports/CLI, `FQO_*` options,
  and the native executable name are unchanged. VQE remains optional.

- Added an optional CPU solver backend and single-process execution without MPI.
- Added a Python callable/batch/executable interface and editable JSON runs.
- FBD uses its depth, beam and search settings to determine its work; FBO
  takes an evaluation budget.
- Added installable source packaging, backend detection and executable tests.
- Added JUPITER build instructions and a four-process test; platform validation is
  tracked in `docs/validation.md`.
- Fixed external FBD probe scoring to invoke the registered callback instead
  of the built-in objective fallback. Unknown objective names now fail.
- Fixed FBD's MPI exchange of the best solution: it now sends the evaluated
  point and its value, instead of replacing the point with the fractal centre.
- Added generic embedded Python modules and checked batch conversion/shape,
  finite results and propagated evaluation failures.

## 0.2.0 — 2026-08-27

- Added the Grid'5000 equal-call, knowledge-driven, and robustness experiment
  drivers with OAR jobs that can continue unfinished runs.
- Added individual and summary Grid'5000 results and the complete local
  195-run robustness dataset.
- Added machine-readable LUMI-G scaling and nine-molecule solution-quality
  coordinates plus hash-indexed excerpts from archived execution records.
  The records identify whether a result has a detailed execution log, an energy
  value or only a scheduler completion record.
- Added a dependency-free verifier that reconstructs the central comparison
  counts and checks result checksums and numerical consistency.
- Updated the VQE evaluator for explicit precision, unwrapped-domain, and
  state-metric support required by the reproduced experiments.
- Added citation metadata and expanded platform-specific rerun documentation.

## 0.1.0 — 2026-07-29

- Combined the configurable FBD and FBO families in one implementation.
- Restored the implemented pybind11 VQE objective path to the configurable
  optimizer driver.
- Added CUDA/HIP build selection, LUMI-G launch material, examples, and
  release checks that require no GPU or quantum packages.
