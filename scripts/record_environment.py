from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from iot_ids.reporting import collect_environment_metadata


def _package_versions() -> dict[str, str]:
    packages = ["numpy", "pandas", "sklearn", "xgboost", "matplotlib"]
    versions: dict[str, str] = {}
    for package in packages:
        try:
            module = __import__(package)
        except ImportError:
            continue
        versions[package] = getattr(module, "__version__", "unknown")
    return versions


def _gpu_info() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip()
    return output or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Record environment metadata for reproducible experiment reporting.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/environment_manifest.json")
    parser.add_argument("--command", nargs=argparse.REMAINDER, required=True)
    args = parser.parse_args()

    metadata = collect_environment_metadata(
        config_path=args.config,
        command=args.command,
        package_versions=_package_versions(),
        gpu_info=_gpu_info(),
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
