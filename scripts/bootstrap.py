from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Install dependencies and prepare local models.")
    parser.add_argument(
        "--with-torch",
        action="store_true",
        help="Install optional PyTorch dependencies for future training/export work.",
    )
    parser.add_argument(
        "--skip-model-download",
        action="store_true",
        help="Do not download or refresh model files.",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip the pytest smoke verification step.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    python = sys.executable

    run([python, "-m", "pip", "install", "--upgrade", "pip"])
    run([python, "-m", "pip", "install", "-r", "requirements.txt"])
    run([python, "-m", "pip", "install", "-r", "requirements-model.txt"])

    if args.with_torch:
        run([python, "-m", "pip", "install", "-r", "requirements-torch.txt"])

    if not args.skip_model_download:
        run([python, "scripts/download_models.py"])

    if not args.no_verify:
        run([python, "-m", "pytest", "-q"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
