import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

from . import Config, CommandObjective, __version__, minimize
from .api import executable_path


def main():
    parser = argparse.ArgumentParser(description="Configurable FBD/FBO optimization")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Check the compiled optimizer installation")
    run = commands.add_parser("run", help="Run a JSON configuration")
    run.add_argument("config", type=Path)
    run.add_argument("--output", type=Path, default=Path("result.json"))
    args = parser.parse_args()
    try:
        if args.command == "doctor":
            engine = executable_path()
            subprocess.run([engine, "--version"], check=True)
            print(f"Native executable: {engine}")
            return 0
        data = json.loads(args.config.read_text(encoding="utf-8"))
        unknown = set(data) - {"objective", "bounds", "config", "timeout"}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        objective_spec = data["objective"]
        if isinstance(objective_spec, dict):
            objective = CommandObjective(**objective_spec)
        else:
            module, name = objective_spec.split(":", 1)
            sys.path.insert(0, str(args.config.resolve().parent))
            objective = getattr(importlib.import_module(module), name)
        result = minimize(objective, data["bounds"], config=Config(**data.get("config", {})),
                          timeout=data.get("timeout", 300))
        args.output.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"{result.status}: f={result.fun:.8g}, evaluations={result.nfev}; {args.output}")
        return 0
    except Exception as exc:
        parser.exit(1, f"fqo: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
