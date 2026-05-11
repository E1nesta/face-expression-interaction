from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODEL_FILES = (
    ROOT / "models" / "face_landmarker.task",
    ROOT / "models" / "emotion.onnx",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the robot expression interaction demo.")
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--source", choices=("camera", "video"), default="camera")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--video-path", default="samples/smoke_no_face.mp4")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--skip-model-check",
        action="store_true",
        help="Run without checking or downloading bundled demo models.",
    )
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def ensure_models() -> None:
    missing = [model for model in MODEL_FILES if not model.exists()]
    if not missing:
        return

    print("missing model files:", flush=True)
    for model in missing:
        print(f"- {model.relative_to(ROOT)}", flush=True)
    run([sys.executable, "scripts/download_models.py"])


def build_main_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "main.py",
        "--config",
        args.config,
        "--source",
        args.source,
    ]

    if args.source == "camera":
        command.extend(["--camera-id", str(args.camera_id)])
    else:
        command.extend(["--video-path", args.video_path])

    if args.max_frames is not None:
        command.extend(["--max-frames", str(args.max_frames)])

    return command


def main() -> int:
    args = parse_args()
    if not args.skip_model_check:
        ensure_models()
    run(build_main_command(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
