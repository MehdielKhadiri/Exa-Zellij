from __future__ import annotations

import ast
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def require(path: str) -> Path:
    target = ROOT / path
    if not target.is_file():
        raise AssertionError(f"missing release file: {path}")
    return target


def test_python_syntax() -> None:
    for relative in ("python/vqe_bridge.py", "python/vqe_cost_reference.py"):
        path = require(relative)
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def test_pauli_reference() -> None:
    payload = json.loads(require("data/lih_sto3g_pauli.json").read_text())
    assert isinstance(payload["n_qubits"], int) and payload["n_qubits"] > 0
    assert payload["num_terms"] == len(payload["terms"])
    assert all("coeff" in term and "pauli" in term for term in payload["terms"])


def test_component_switches_are_present() -> None:
    fbd = require("src/FBDConfig.cpp").read_text(encoding="utf-8")
    fbo = require("src/FBOConfig.cpp").read_text(encoding="utf-8")
    main = require("src/main.cpp").read_text(encoding="utf-8")

    for flag in (
        "--fbd-decomposition",
        "--fbd-exploration",
        "--fbd-scoring",
        "--fbd-tree-search",
        "--fbd-exploitation",
    ):
        assert flag in fbd

    for flag in (
        "--fbo-doe",
        "--fbo-decomposition",
        "--fbo-surrogate",
        "--fbo-gp-kernel",
        "--fbo-exploitation",
        "--fbo-budget",
    ):
        assert flag in fbo

    assert "init_vqe_bridge" in main
    assert "vqe_eval_many_trampoline" in main


def test_no_site_identity_or_credentials() -> None:
    text_extensions = {
        ".cpp", ".hpp", ".py", ".md", ".txt", ".yml", ".yaml",
        ".sh", ".sbatch", ".cmake",
    }
    forbidden = (
        re.compile(r"project_\d+"),
        re.compile(r"/home/[A-Za-z0-9._-]+"),
        re.compile(r"/users/[A-Za-z0-9._-]+"),
        re.compile(r"BEGIN (?:RSA |OPENSSH )?PRIVATE KEY"),
        re.compile(r"(?i)(?:api[_-]?key|token|password)\s*[:=]\s*[\"'][^\"']+"),
    )

    for path in ROOT.rglob("*"):
        relative_parts = path.relative_to(ROOT).parts
        if any(
            part == ".git"
            or part == "__pycache__"
            or part.startswith("build")
            for part in relative_parts
        ):
            continue
        if not path.is_file() or path.suffix.lower() not in text_extensions:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in forbidden:
            if pattern.search(text):
                raise AssertionError(f"sensitive/site-specific text in {path}: {pattern.pattern}")


if __name__ == "__main__":
    test_python_syntax()
    test_pauli_reference()
    test_component_switches_are_present()
    test_no_site_identity_or_credentials()
    print("release asset checks: OK")
