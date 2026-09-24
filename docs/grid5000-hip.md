# AMD verification on Grid'5000

Grid'5000's Larochette cluster in Luxembourg provides AMD MI210 GPUs (gfx90a).
This gives another place to validate the HIP backend before deploying on LUMI.
See the [site hardware inventory](https://www.grid5000.fr/w/Luxembourg:Hardware).

From the repository on the Luxembourg frontend:

```sh
chmod +x scripts/grid5000/hip_smoke.oar.sh
oarsub -S ./scripts/grid5000/hip_smoke.oar.sh
```

The script requests one node for at most 20 minutes. It loads the site ROCm
7.1.1, rocSOLVER, CMake, Open MPI and Python modules, then lets `configure.py`
detect HIP. The architecture is explicitly VEGA90A. Module versions may need
updating as the site environment changes.

Compilation runs on the allocated node in `/tmp`. The job then runs two-rank
FBD/FBO analytic tests and the Python callback/component tests. Logs, the native
executable, an exit code and (on success) `PASS.txt` are saved under
`results/grid5000-hip-JOB_ID`. A submitted job or a successful compile alone is
not a completed runtime validation; check `PASS.txt` and the test logs.

The initial job, 334968, exited during setup on 2026-09-21 before building or
running tests. Its stderr reports a missing user `.bashrc`; no `PASS.txt` was
produced. The recipe is not yet a successful AMD validation. See
[validation.md](validation.md) for the measured status.
