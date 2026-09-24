"""Check distributed native point/value pairs, including off-centre FBD probes."""
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
    parser.add_argument("--python", action="store_true")
    args = parser.parse_args()
    objective_dir = Path(__file__).resolve().parent
    environment = dict(os.environ, OMPI_MCA_rmaps_base_oversubscribe="1")
    with tempfile.TemporaryDirectory(prefix="fqo-mpi-") as directory:
        for family in ("fbd", "fbo"):
            for objective in (("shiftedsphere", "external") if args.python else ("shiftedsphere",)):
                output = Path(directory) / f"{family}-{objective}.json"
                command = [args.mpiexec, args.numproc_flag, "2", args.executable,
                           "4", "3", "80", "1", objective,
                           "--optimizer-family", family, "--max-depth", "2",
                           "--fbo-doe", "halton", "--fbo-doe-distribution", "distributed",
                           "--seed", "7", "--result-json", str(output)]
                if family == "fbo":
                    command += ["--n-init", "20"]
                if objective == "external":
                    command += ["--python-module", "embedded_objective", "--python-dir", str(objective_dir)]
                run = subprocess.run(command, cwd=directory, env=environment,
                                     capture_output=True, text=True, timeout=60)
                if run.returncode:
                    raise AssertionError(run.stdout + run.stderr)
                result = json.loads(output.read_text())
                offset, centre = (7, .37) if objective == "external" else (5, 1)
                expected = offset + sum((x-centre)**2 for x in result["x"])
                if not math.isclose(result["fun"], expected, rel_tol=1e-6, abs_tol=1e-6):
                    raise AssertionError((family, objective, result, expected))
                if "[DEPTH" not in run.stdout:
                    raise AssertionError("The test must exercise search after the initial design")
                print(f"PASS {family}/{objective}: fun={result['fun']}")


if __name__ == "__main__":
    main()
