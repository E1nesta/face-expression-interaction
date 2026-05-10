from __future__ import annotations

import argparse
from pathlib import Path
from urllib.request import urlretrieve


FACE_LANDMARKER_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists() and not args.force:
        print(f"exists: {target}")
        return 0

    print(f"downloading: {FACE_LANDMARKER_URL}")
    print(f"target: {target}")
    urlretrieve(FACE_LANDMARKER_URL, target)
    print(f"saved: {target} ({target.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
