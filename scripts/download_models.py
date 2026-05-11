from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
EMOTION_ONNX_URL = (
    "https://github.com/sb-ai-lab/EmotiEffLib/raw/main/models/"
    "affectnet_emotions/onnx/enet_b0_8_best_vgaf.onnx"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download local model assets.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download even when the target file already exists.",
    )
    parser.add_argument(
        "--output",
        default="models/face_landmarker.task",
        help="Target path for the MediaPipe FaceLandmarker model.",
    )
    parser.add_argument(
        "--emotion-output",
        default="models/emotion.onnx",
        help="Target path for the ONNX emotion model.",
    )
    parser.add_argument(
        "--skip-face",
        action="store_true",
        help="Do not download the MediaPipe FaceLandmarker model.",
    )
    parser.add_argument(
        "--skip-emotion",
        action="store_true",
        help="Do not download the ONNX emotion model.",
    )
    return parser.parse_args()


def download_file(url: str, target: Path, force: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not force:
        print(f"exists: {target}")
        return

    print(f"downloading: {url}")
    print(f"target: {target}")
    urlretrieve(url, target)
    print(f"saved: {target} ({target.stat().st_size} bytes)")


def main() -> int:
    args = parse_args()

    if not args.skip_face:
        download_file(FACE_LANDMARKER_URL, Path(args.output), args.force)

    if not args.skip_emotion:
        download_file(EMOTION_ONNX_URL, Path(args.emotion_output), args.force)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
