# AMD tests on Grid'5000

Grid'5000's Larochette cluster in Luxembourg provides AMD MI210 GPUs (gfx90a).
The job below builds the HIP backend and runs its tests on one node.
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
`results/grid5000-hip-JOB_ID`. Check `PASS.txt` and the logs to confirm that
the tests completed.

Job 334968 stopped during setup on 21 September 2026 because the user's
`.bashrc` was missing. It did not reach the build or tests. The HIP tests
therefore still need to be run; see the [test report](validation.md).
