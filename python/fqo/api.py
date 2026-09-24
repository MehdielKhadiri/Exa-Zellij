from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field
import json
import math
import os
from pathlib import Path
import queue
import shutil
import subprocess
import tempfile
import threading
import time
from typing import Callable, Sequence


_CHOICES = {
    "decomposition": {"hypersphere", "hypercube", "bisection", "trisection", "voronoi", "simplex"},
    "split-policy": {"cyclic", "longest-side", "random"},
    "exploitation": {"best-cell", "random-local", "random-restart", "none", "surrogate-local"},
    "doe": {"sobol", "file", "random", "lhs", "halton", "hammersley"},
    "doe-distribution": {"replicated", "distributed"},
    "surrogate": {"pca-gp", "hsic-gp", "idw", "rbf", "none"},
    "gp-kernel": {"rbf", "exponential", "matern32", "matern52", "rational-quadratic"},
    "budget": {"astra", "fixed-fractions"},
    "radius-mode": {"fixed", "adaptive"},
    "exploitation-guidance": {"fixed", "adaptive"},
    "exploration": {"center", "random", "center-random", "center-axial", "corners", "low-discrepancy"},
    "scoring": {"best-value", "mean-value", "improvement", "improvement-distance", "optimistic"},
    "tree-search": {"beam", "best-first", "breadth-first", "depth-first", "epsilon-greedy", "cyclic-best"},
    "initial-design": {"random", "lhs", "halton", "hammersley"},
}
_COMMON = {"decomposition", "split-policy", "exploitation", "hypercube-active-dims", "voronoi-sites", "simplex-children"}
_FBO = _COMMON | {"doe", "doe-distribution", "surrogate", "gp-kernel", "budget", "radius-mode", "exploitation-guidance", "surrogate-length-scale", "idw-power", "gp-sigma-f", "gp-length-scale", "gp-noise", "gp-rq-alpha", "exploit-starts", "exploit-local-iters", "exploit-radius-scale"}
_FBD = _COMMON | {"exploration", "scoring", "tree-search", "initial-design", "points-per-cell", "initial-points", "beam-width", "epsilon"}


@dataclass(frozen=True)
class Config:
    """FBO uses an evaluation budget (default 100); FBD uses its structure.

    Leave budget unset for FBD: depth, beam, geometry and exploration determine
    its evaluation count. Both families report actual user evaluations.
    """
    family: str = "fbo"
    budget: int | None = None
    beam: int = 4
    max_depth: int = 3
    seed: int = 0
    n_init: int | None = None
    components: dict = field(default_factory=dict)

    def arguments(self, dimension: int) -> list[str]:
        if self.family not in {"fbo", "fbd"}:
            raise ValueError("family must be 'fbo' or 'fbd'")
        if self.family == "fbd" and self.budget is not None:
            raise ValueError("FBD has no evaluation budget; leave budget unset and configure depth/beam/exploration")
        if self.family == "fbd" and self.n_init is not None:
            raise ValueError("Configure FBD's initial-points component instead of n_init")
        if self.budget is not None and (isinstance(self.budget, bool) or not isinstance(self.budget,int) or self.budget < 1):
            raise ValueError("FBO budget must be a positive integer")
        for name in ("beam", "max_depth"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        # The legacy executable requires a positional budget token for both
        # families. FBD's search is structural; this token is not a call cap.
        native_budget = self.budget if self.budget is not None else 100
        args = [str(self.beam), str(dimension), str(native_budget), "1", "external",
                "--objective-stdio", "--optimizer-family", self.family,
                "--max-depth", str(self.max_depth), "--seed", str(self.seed),
                "--lower-bound", "0", "--upper-bound", "1"]
        # Explicit, self-contained initial design: no implicit Sobol CSV file.
        components = {"doe": "halton"} if self.family == "fbo" else {}
        components.update({key.replace("_", "-"): value for key, value in self.components.items()})
        if self.family == "fbo":
            minimum = 2 if components.get("surrogate", "pca-gp") in {"pca-gp", "hsic-gp"} else 1
            initial = self.n_init if self.n_init is not None else min(native_budget, max(minimum, native_budget // 4))
            if isinstance(initial, bool) or not isinstance(initial, int) or not minimum <= initial <= native_budget:
                raise ValueError(f"n_init must be an integer between {minimum} and the FBO budget")
            args += ["--n-init", str(initial)]
        allowed = _FBO if self.family == "fbo" else _FBD
        for key, value in components.items():
            if key not in allowed:
                raise ValueError(f"Unknown {self.family} component: {key}")
            if key in _CHOICES:
                if value not in _CHOICES[key]:
                    raise ValueError(f"Invalid {key}: {value!r}; choose from {sorted(_CHOICES[key])}")
            elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            if self.family == "fbd" and key == "exploitation" and value == "surrogate-local":
                raise ValueError("surrogate-local requires the fbo family")
            if key == "doe" and value in {"sobol", "file"}:
                raise ValueError("File-backed DOE currently requires the native CLI; use halton, lhs or random here")
            args += [f"--{self.family}-{key}", str(value)]
        return args


@dataclass(frozen=True)
class Result:
    x: list[float]
    fun: float
    nfev: int
    status: str
    elapsed_seconds: float
    config: Config
    log: str = field(repr=False)

    def to_dict(self) -> dict:
        return asdict(self)


def executable_path(executable: str | os.PathLike | None = None) -> str:
    candidate = executable or os.environ.get("FQO_EXECUTABLE")
    if candidate:
        resolved = shutil.which(str(candidate))
        if resolved:
            return resolved
        raise FileNotFoundError(f"FQO executable not found or not executable: {candidate}")
    bundled = Path(__file__).parent / "bin" / ("fractal_optimizer.exe" if os.name == "nt" else "fractal_optimizer")
    if bundled.is_file():
        return str(bundled)
    found = shutil.which("fractal_optimizer")
    if found:
        return found
    raise FileNotFoundError("Install the compiled package with 'python -m pip install .', or set FQO_EXECUTABLE to your CMake-built executable")


@dataclass(frozen=True)
class CommandObjective:
    """Any-language adapter: JSON {x:[...]} on stdin, JSON number on stdout.

    The command is an argument list, never interpreted by a shell. Diagnostic
    output belongs on stderr. Use a persistent/batched callable for cheap work.
    """
    command: Sequence[str]
    timeout: float = 60.0

    def __call__(self, x: list[float]) -> float:
        if isinstance(self.command, (str, bytes)) or not self.command:
            raise ValueError("command must be a nonempty argument list")
        result = subprocess.run(list(self.command), input=json.dumps({"x": x}),
                                text=True, capture_output=True, check=True,
                                timeout=self.timeout)
        return float(json.loads(result.stdout))


def minimize(objective: Callable, bounds: Sequence[Sequence[float]], *,
             config: Config | None = None, batch_objective: Callable | None = None,
             executable: str | os.PathLike | None = None, timeout: float = 300.0) -> Result:
    """Minimize a scalar real-valued callable using the compiled FBD/FBO core.

    Coordinates are affinely normalized inside the core; the callback and result
    use the supplied physical bounds. Supports different bounds per coordinate.
    A scalar callback receives a list; batch_objective receives a list of rows.
    Every value must be finite. Exceptions propagate after stopping the child.
    timeout bounds waiting on the native engine, not an in-process user callback.
    """
    config = config or Config()
    box = [(float(pair[0]), float(pair[1])) for pair in bounds if len(pair) == 2]
    if len(box) != len(bounds) or not box or any(not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi for lo, hi in box):
        raise ValueError("bounds must contain one finite (lower, upper) pair per dimension")
    if not callable(objective) or (batch_objective is not None and not callable(batch_objective)):
        raise TypeError("objective and optional batch_objective must be callable")
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    args = config.arguments(len(box))
    engine = executable_path(executable)
    logs: deque[str] = deque(maxlen=500)
    evaluations = 0
    best_x: list[float] = []
    best_value = math.inf
    status = "completed"
    start = time.monotonic()

    with tempfile.TemporaryDirectory(prefix="fqo-") as directory:
        result_path = Path(directory) / "result.json"
        process = subprocess.Popen([engine, *args, "--result-json", str(result_path)],
                                   cwd=directory, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1)
        lines: queue.Queue = queue.Queue()
        def read_output():
            try:
                for line in process.stdout:
                    lines.put(line)
            finally:
                lines.put(None)
        reader = threading.Thread(target=read_output, daemon=True)
        reader.start()
        try:
            while True:
                remaining_time = timeout - (time.monotonic() - start)
                if remaining_time <= 0:
                    raise TimeoutError(f"FQO exceeded {timeout} seconds")
                try:
                    line = lines.get(timeout=remaining_time)
                except queue.Empty:
                    raise TimeoutError(f"FQO exceeded {timeout} seconds") from None
                if line is None:
                    break
                if not line.startswith("@FQO_EVAL "):
                    logs.append(line)
                    continue
                rows = json.loads(line[len("@FQO_EVAL "):])["points"]
                if any(len(row) != len(box) or any(not math.isfinite(v) for v in row) for row in rows):
                    raise RuntimeError("Native optimizer requested invalid coordinates")
                physical = [[lo + value * (hi - lo) for value, (lo, hi) in zip(row, box)] for row in rows]
                remaining = ((config.budget if config.budget is not None else 100) - evaluations
                             if config.family == "fbo" else len(physical))
                selected = physical[:remaining]
                values = list(batch_objective(selected)) if batch_objective and selected else [objective(x) for x in selected]
                if len(values) != len(selected):
                    raise ValueError("batch_objective must return exactly one value per row")
                values = [float(v) for v in values]
                if any(not math.isfinite(v) for v in values):
                    raise ValueError("objective returned a non-finite value")
                for x, value in zip(selected, values):
                    evaluations += 1
                    if value < best_value:
                        best_value, best_x = value, x
                if len(selected) < len(physical):
                    status = "evaluation_limit"
                    break
                process.stdin.write(" ".join(format(value, ".17g") for value in values) + "\n")
                process.stdin.flush()
            if status == "completed":
                code = process.wait(timeout=max(0.01, timeout - (time.monotonic() - start)))
                if code:
                    raise RuntimeError(f"Native optimizer failed (exit {code}):\n{''.join(logs)}")
                if not result_path.is_file():
                    raise RuntimeError("Native optimizer did not produce its result file")
                native = json.loads(result_path.read_text())
                if native.get("schema_version") != 1 or not math.isfinite(native["fun"]):
                    raise RuntimeError("Invalid native optimization result")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            process.stdin.close()
            reader.join(timeout=5)
            process.stdout.close()
    if not best_x:
        raise RuntimeError("Optimizer completed without evaluating the objective")
    return Result(best_x, best_value, evaluations, status, time.monotonic() - start, config, "".join(logs))
