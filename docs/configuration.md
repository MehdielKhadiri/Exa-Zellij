# Runtime configuration

At startup, the executable prints the family and components it will use. Options are case-insensitive.

## Common arguments

```text
fractal_optimizer <beam_cap> <dimension> <budget> <expense_factor> <objective>
```

`beam_cap` limits the number of active fractals. `dimension` is ignored for VQE after
the Python evaluator reports the ansatz parameter count. `budget` is the global
objective-evaluation budget for FBO. FBD requires this argument but ignores
its value: decomposition, depth, beam and search components determine its work.
`expense_factor` repeats arithmetic for the included analytic objectives. `objective` can also be supplied with
`--objective`.

Common options:

| Option | Meaning |
|---|---|
| `--optimizer-family fbd\|fbo` | Select the optimizer family |
| `--rho R` | Acquisition-mass coverage used by FBO tree selection |
| `--n-init N` | Override initial-design size |
| `--max-depth H` | Maximum fractal-tree depth |
| `--seed S` | Reproducibility seed |
| `--lower-bound X`, `--upper-bound X` | Override box bounds |
| `--exploit-budget N` | Explicit global exploitation budget |
| `--doe-file PATH` | CSV design used with a file-backed DOE |

## FBD components

FBD is configured by five independent choices:

| Component | Option | Values |
|---|---|---|
| Fractal decomposition | `--fbd-decomposition` | `hypersphere`, `hypercube`, `bisection`, `trisection`, `voronoi`, `simplex` |
| Exploration | `--fbd-exploration` | `center`, `random`, `center-random`, `center-axial`, `corners`, `low-discrepancy` |
| Scoring | `--fbd-scoring` | `best-value`, `mean-value`, `improvement`, `improvement-distance`, `optimistic` |
| Tree search | `--fbd-tree-search` | `beam`, `best-first`, `breadth-first`, `depth-first`, `epsilon-greedy`, `cyclic-best` |
| Exploitation | `--fbd-exploitation` | `best-cell`, `random-local`, `random-restart`, `none` |

The initial design is selected with `--fbd-initial-design`:
`random`, `lhs`, `halton`, or `hammersley`.

Additional controls include `--fbd-points-per-cell`,
`--fbd-initial-points`, `--fbd-beam-width`, `--fbd-epsilon`,
`--fbd-hypercube-active-dims`, `--fbd-voronoi-sites`, and
`--fbd-simplex-children`.

### Branching and beam width

Hypersphere decomposition creates `2*d` children per parent. Defaults
for simplex-lite (`simplex`) and Voronoi-style (`voronoi`) are `d+1` and `2*d+1`.
These two implementations use a center plus positive coordinate offsets or
signed coordinate offsets, respectively; they are not arbitrary geometric
simplex/Voronoi tessellations. Smaller requested child counts restrict the
covered coordinate directions. Counts above those defaults repeat generated
children rather than introducing new directions. Distinct children of one
parent do not guarantee non-overlapping regions across parents.

`--fbd-beam-width` limits the number of active fractals; it does not set branching or guarantee
a full beam at every level. More children, wider beams, more exploration points and longer
exploitation increase work and can change solution quality. Report actual
evaluations as well as runtime when comparing them.

For MPI comparisons, explicitly select `--fbo-doe-distribution distributed`
even with FBD if `--n-init` is intended as a global initial-design size. The
default replicated mode performs the design on every rank and increases work
with rank count. Check the reported evaluation counts when comparing runs with different
numbers of processes.

## FBO components

For FBO, choose the design of experiments, decomposition, surrogate, budget
distribution and exploitation algorithm:

| Component | Option | Values |
|---|---|---|
| DOE | `--fbo-doe` | `sobol`, `file`, `random`, `lhs`, `halton`, `hammersley` |
| DOE execution | `--fbo-doe-distribution` | `replicated`, `distributed` |
| Decomposition | `--fbo-decomposition` | `hypersphere`, `hypercube`, `bisection`, `trisection`, `voronoi`, `simplex` |
| Split rule | `--fbo-split-policy` | `cyclic`, `longest-side`, `random` |
| Surrogate | `--fbo-surrogate` | `pca-gp`, `hsic-gp`, `idw`, `rbf`, `none` |
| GP kernel | `--fbo-gp-kernel` | `rbf`, `exponential`, `matern32`, `matern52`, `rational-quadratic` |
| Exploitation | `--fbo-exploitation` | `surrogate-local`, `random-local`, `random-restart`, `best-cell`, `none` |
| Budget distribution | `--fbo-budget` | `astra`, `fixed-fractions` |

The contraction and exploitation-guidance settings can be selected independently:

- `--fbo-radius-mode fixed|adaptive`
- `--fbo-exploitation-guidance fixed|adaptive`

Fixed-fraction budgeting uses `--doe-frac`, `--explore-frac`, and
`--exploit-frac`. GP and exploitation controls are exposed as
`--fbo-gp-sigma-f`, `--fbo-gp-length-scale`, `--fbo-gp-noise`,
`--fbo-gp-rq-alpha`, `--fbo-exploit-starts`,
`--fbo-exploit-local-iters`, and `--fbo-exploit-radius-scale`.

For a fresh CLI/MPI FBO run, the global cap counts actual evaluations across
all ranks, including each rank's initial-center evaluation and any replicated
DOE calls. The phase fractions set the intended shares of this budget: exploration must fit the
remainder after initialization, and exploitation is limited by the actual
remaining budget. Local-search starts and coordinate proposals count toward
that limit. A run may finish below its cap. If initialization leaves no search
budget, reduce `--n-init`, use distributed DOE, or increase the cap.

## VQE objective

| Option | Meaning |
|---|---|
| `--objective vqe` | Select the external VQE objective |
| `--vqe-mol NAME` | Molecule available to PennyLane qchem |
| `--vqe-basis BASIS` | Basis set, for example `STO-3G` |
| `--vqe-python-dir PATH` | Directory containing `vqe_bridge.py` |

`VQE_BATCH` controls the fixed JAX batch shape. `VQE_OFFLINE=1` prohibits the
online PennyLane data path and loads an existing local qchem HDF5 file.
