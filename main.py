from __future__ import annotations

import argparse
from pathlib import Path

from app.pipeline import RobotExpressionPipeline
from control.output import ConsoleJsonOutput
from core.config_loader import load_config, merge_cli_overrides
from core.logger import setup_logging
from decision.fsm import ExpressionDecisionFsm
from demo.opencv_viewer import OpenCvViewer
from features.feature_smoother import FeatureSmoother
from input.text_input import AsyncTextInput
from input.video_source import VideoSource
from perception.face_landmark import create_face_landmark_detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robot expression interaction demo")
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--source", choices=("camera", "video"))
    parser.add_argument("--camera-id", type=int)
    parser.add_argument("--video-path")
    parser.add_argument(
        "--max-frames",
        type=int,
        help="Stop after N frames. Useful for smoke tests and camera validation.",
    )
    return parser.parse_args()


def main() -> int:
    setup_logging()
    args = parse_args()
    config = merge_cli_overrides(
        load_config(Path(args.config)),
        {
            "source": args.source,
            "camera_id": args.camera_id,
            "video_path": args.video_path,
        },
    )

    video_source = VideoSource(
        source=config["source"],
        camera_id=int(config["camera_id"]),
        video_path=config.get("video_path") or None,
        frame_width=int(config["frame_width"]),
        frame_height=int(config["frame_height"]),
    )
    text_input = AsyncTextInput()
    detector = create_face_landmark_detector(config["face_detector"])
    viewer = OpenCvViewer(config["demo"]["window_name"])
    output = ConsoleJsonOutput()

    pipeline = RobotExpressionPipeline(
        video_source=video_source,
        text_input=text_input,
        detector=detector,
        viewer=viewer,
        write_command=output.send,
        smoother=FeatureSmoother(config["smoothing"]["ema_alpha"]),
        fsm=ExpressionDecisionFsm(config["decision"]["stable_frames"]),
        face_emotion_thresholds=config["emotion"]["thresholds"],
        overlay_options={
            "draw_landmarks": bool(config["demo"]["draw_landmarks"]),
            "draw_fps": bool(config["demo"]["draw_fps"]),
            "draw_status": bool(config["demo"]["draw_status"]),
        },
        default_duration_ms=int(config["control"]["default_duration_ms"]),
        default_transition_ms=int(config["control"]["default_transition_ms"]),
    )
    pipeline.run(max_frames=args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
