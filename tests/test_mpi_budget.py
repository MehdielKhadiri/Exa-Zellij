"""Exercise actual global FBO budgets, including MPI initialization and ILS."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--executable", required=True)
    parser.add_argument("--mpiexec", required=True)
    parser.add_argument("--numproc-flag", default="-n")
    args = parser.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_PROC_BIND="false",
               OMPI_MCA_rmaps_base_oversubscribe="1")
    cases = [
        ("center-cost", 100, 25, "best-cell", []),
        ("exploration-cap", 40, 25, "none", ["--exploit-frac", "0"]),
        ("restart-cost", 400, 25, "random-restart-local",
         ["--exploit-frac", "0.8", "--explore-frac", "0.1",
          "--fbo-exploit-starts", "2", "--fbo-exploit-local-iters", "1"]),
        ("impossible-init", 28, 25, "best-cell", []),
    ]
    with tempfile.TemporaryDirectory(prefix="fqo-mpi-budget-") as directory:
        for name, budget, initial, policy, extra in cases:
            output = Path(directory) / (name + ".json")
            command = [args.mpiexec, args.numproc_flag, "4", args.executable,
                       "64", "10", str(budget), "0", "shifted-ellipsoid",
                       "--optimizer-family", "fbo", "--max-depth", "4",
                       "--fbo-decomposition", "hypersphere",
                       "--fbo-surrogate", "no-surrogate", "--fbo-doe", "random",
                       "--fbo-doe-distribution", "distributed", "--n-init", str(initial),
                       "--fbo-budget", "fixed", "--doe-frac", "0.25",
                       "--explore-frac", "0.65", "--exploit-frac", "0.10",
                       "--fbo-exploitation", policy, "--lower-bound", "-5",
                       "--upper-bound", "5", "--seed", "0", "--result-json", str(output)]
            # Replace defaults rather than supplying repeated CLI options.
            for flag, value in zip(extra[::2], extra[1::2]):
                if flag in command:
                    command[command.index(flag) + 1] = value
                else:
                    command += [flag, value]
            run = subprocess.run(command, cwd=directory, env=env,
                                 capture_output=True, text=True, timeout=60)
            if name == "impossible-init":
                assert run.returncode != 0 and "FBO initialization requires" in run.stderr, run.stdout + run.stderr
                assert not output.exists()
                print("PASS", name, "rejected before objective execution")
                continue
            assert run.returncode == 0, run.stdout + run.stderr
            record = json.loads(output.read_text())
            count = record["evaluations"]
            assert initial + 4 < count <= budget, (name, count, budget)
            assert len(record["rank_evaluations"]) == 4
            assert sum(record["rank_evaluations"]) == count
            expected = sum(10 ** (6 * j / 9) *
                           (x - (0.35 + 0.9 * ((37 * j + 11) % 101) / 100)) ** 2
                           for j, x in enumerate(record["x"]))
            assert math.isclose(record["fun"], expected, rel_tol=1e-8, abs_tol=1e-7)
            assert "[DEPTH" in run.stdout
            if name == "restart-cost":
                assert "[exploit] policy = random-restart-local" in run.stdout, run.stdout
            print("PASS", name, count, "<=", budget)


if __name__ == "__main__":
    main()
