#!/usr/bin/env python3
"""Detect CPU/CUDA/HIP before CMake selects a compiler. No Python packages needed."""
import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess


def installed_kokkos(prefixes):
    for prefix in prefixes:
        root = Path(prefix)
        candidates = [root / "KokkosConfigCommon.cmake"]
        candidates += [root / lib / "cmake/Kokkos/KokkosConfigCommon.cmake" for lib in ("lib", "lib64")]
        for path in candidates:
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8")
            match = re.search(r"SET\(Kokkos_DEVICES\s+([^)]*)\)", text, re.I)
            if not match:
                continue
            devices = set(re.findall(r"[A-Z0-9]+", match.group(1).upper()))
            if "CUDA" in devices:
                return "cuda", root
            if "HIP" in devices:
                return "hip", root
            if devices & {"SYCL", "OPENMPTARGET", "OPENACC"}:
                raise ValueError("This Kokkos backend has no FQO solver adapter yet; use CPU, CUDA or HIP")
            return "cpu", root
    return None, None


def detect_backend(prefixes, which=shutil.which):
    backend, prefix = installed_kokkos(prefixes)
    if backend:
        return backend, prefix, "existing Kokkos configuration"
    cuda = which("nvcc_wrapper") or which("nvcc")
    hip = which("hipcc")
    if cuda and hip:
        raise ValueError("Both CUDA and HIP toolchains are present. Select --backend cuda or --backend hip")
    if cuda:
        return "cuda", None, "CUDA toolchain"
    if hip:
        return "hip", None, "HIP toolchain"
    return "cpu", None, "no CUDA/HIP toolchain found"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "hip"], default="auto")
    parser.add_argument("--prefix", action="append", default=[], help="Dependency/Kokkos install prefix; repeatable")
    parser.add_argument("--arch", help="Kokkos GPU architecture, e.g. HOPPER90 or VEGA90A")
    parser.add_argument("--mpi", action="store_true")
    parser.add_argument("--python", action="store_true", help="Enable embedded Python objective modules")
    parser.add_argument("--cxx", help="Explicit C++/device compiler")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--build", action="store_true", help="Also compile after configuration")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--kernels-tag", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args, extra = parser.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]
    if any(not item.startswith("-D") for item in extra):
        parser.error("Additional options must be CMake -DKEY=VALUE definitions")
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    repo = Path(__file__).resolve().parent
    prefixes = list(args.prefix)
    prefixes += [v for v in re.split(r";" if os.name == "nt" else r"[;:]", os.environ.get("CMAKE_PREFIX_PATH", "")) if v]
    prefixes += [os.environ[v] for v in ("Kokkos_ROOT", "Kokkos_DIR", "EBROOTKOKKOS") if os.environ.get(v)]
    try:
        installed, prefix = installed_kokkos(prefixes)
        if args.backend == "auto":
            backend, prefix, reason = detect_backend(prefixes)
        else:
            backend, reason = args.backend, "explicit selection"
            if installed and installed != backend:
                raise ValueError(f"Selected {backend}, but the supplied Kokkos is {installed}. Use a matching installation or remove that prefix")
        build = (args.build_dir or repo / f"build-{backend}").resolve()
        cmake = shutil.which("cmake")
        if not cmake:
            raise ValueError("CMake is missing. Install CMake 3.20+ (for example: python -m pip install cmake)")
        compiler = args.cxx
        source_override = None
        if backend == "hip" and not compiler:
            compiler = shutil.which("hipcc")
            if not compiler:
                raise ValueError("HIP selected but hipcc is missing; load ROCm or provide --cxx")
        if backend == "cuda" and not compiler:
            compiler = shutil.which("nvcc_wrapper")
            if not compiler and prefix:
                wrapper = prefix / "bin/nvcc_wrapper"
                if wrapper.is_file():
                    compiler = str(wrapper)
            if not compiler:
                if not shutil.which("nvcc"):
                    raise ValueError("CUDA selected but nvcc is missing; load/install the CUDA toolkit")
                if os.name == "nt":
                    raise ValueError("CUDA source builds use nvcc_wrapper under Linux/WSL; native Windows is outside the supported scope")
                source_override = build / "_bootstrap/kokkos"
                if not args.dry_run and not source_override.exists():
                    source_override.parent.mkdir(parents=True, exist_ok=True)
                    subprocess.run(["git", "clone", "--depth", "1", "--branch", "4.5.01",
                                    "https://github.com/kokkos/kokkos.git", str(source_override)], check=True)
                compiler = str(source_override / "bin/nvcc_wrapper")
        command = [cmake, "-S", str(repo), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
                   f"-DFQO_GPU_BACKEND={backend.upper()}",
                   f"-DFQO_ENABLE_MPI={'ON' if args.mpi else 'OFF'}",
                   f"-DFQO_ENABLE_PYTHON={'ON' if args.python else 'OFF'}"]
        if compiler:
            command.append(f"-DCMAKE_CXX_COMPILER={compiler}")
        if prefixes:
            command.append("-DCMAKE_PREFIX_PATH=" + ";".join(prefixes))
        if args.arch:
            if not re.fullmatch(r"[A-Za-z0-9_]+", args.arch):
                raise ValueError("Invalid Kokkos architecture name")
            command.append(f"-DKokkos_ARCH_{args.arch.upper()}=ON")
        if args.python:
            command.append("-DFQO_FETCH_PYBIND11=ON")
        if source_override:
            command.append(f"-DFETCHCONTENT_SOURCE_DIR_KOKKOS={source_override}")
        kernels_tag = args.kernels_tag
        if not kernels_tag and prefix and "4.7." in str(prefix):
            kernels_tag = "4.7.01"
        if kernels_tag:
            command.append(f"-DFQO_KOKKOSKERNELS_TAG={kernels_tag}")
        command += extra
        print(f"Backend: {backend} ({reason}); MPI: {args.mpi}; embedded Python: {args.python}", flush=True)
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, check=True)
            if args.build:
                subprocess.run([cmake, "--build", str(build), "--config", "Release", "--parallel", str(args.jobs)], check=True)
        return 0
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"configure: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
