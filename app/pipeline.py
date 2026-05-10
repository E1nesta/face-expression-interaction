from __future__ import annotations

from collections.abc import Callable

from control.protocol import build_control_command
from core.data_structures import ControlCommand
from decision.fsm import ExpressionDecisionFsm
from demo.overlay import draw_overlay
from emotion.face_emotion import classify_face_emotion
from emotion.fusion import fuse_emotions
from emotion.text_emotion import classify_text_emotion
from expression.expression_generator import generate_expression_command
from features.facial_feature_extractor import extract_facial_features
from features.feature_smoother import FeatureSmoother
from perception.head_pose import estimate_head_pose


class RobotExpressionPipeline:
    def __init__(
        self,
        video_source,
        text_input,
        detector,
        viewer,
        write_command: Callable[[ControlCommand], None],
        overlay_func=draw_overlay,
        smoother: FeatureSmoother | None = None,
        fsm: ExpressionDecisionFsm | None = None,
        face_emotion_thresholds: dict[str, float] | None = None,
        overlay_options: dict[str, bool] | None = None,
        default_duration_ms: int = 1500,
        default_transition_ms: int = 400,
    ) -> None:
        self.video_source = video_source
        self.text_input = text_input
        self.detector = detector
        self.viewer = viewer
        self.write_command = write_command
        self.overlay_func = overlay_func
        self.smoother = smoother or FeatureSmoother(ema_alpha=0.5)
        self.fsm = fsm or ExpressionDecisionFsm(stable_frames=1)
        self.face_emotion_thresholds = face_emotion_thresholds
        self.overlay_options = overlay_options or {}
        self.default_duration_ms = default_duration_ms
        self.default_transition_ms = default_transition_ms

    def run(self, max_frames: int | None = None) -> int:
        processed = 0
        previous_timestamp: float | None = None
        self.text_input.start()
        try:
            while max_frames is None or processed < max_frames:
                frame_input = self.video_source.read()
                if frame_input is None:
                    break

                timestamp_ms = int(frame_input.timestamp * 1000)
                face = self.detector.detect(frame_input.frame, timestamp_ms=timestamp_ms)
                head_pose = estimate_head_pose(face)
                features = extract_facial_features(face, head_pose)
                smoothed_features = self.smoother.update(features)

                text_state = self.text_input.latest_state()
                face_emotion = classify_face_emotion(
                    smoothed_features,
                    thresholds=self.face_emotion_thresholds,
                )
                text_emotion = classify_text_emotion(text_state)
                emotion = fuse_emotions(
                    face_detected=face.face_detected,
                    face_emotion=face_emotion,
                    text_emotion=text_emotion,
                    head_pose=head_pose,
                )
                decision = self.fsm.update(emotion)
                expression = generate_expression_command(
                    decision,
                    default_duration_ms=self.default_duration_ms,
                    default_transition_ms=self.default_transition_ms,
                )
                command = build_control_command(
                    decision,
                    expression,
                    timestamp=frame_input.timestamp,
                )
                self.write_command(command)

                fps = _compute_fps(previous_timestamp, frame_input.timestamp)
                previous_timestamp = frame_input.timestamp
                output_frame = self.overlay_func(
                    frame_input.frame,
                    face=face,
                    emotion=emotion,
                    decision=decision,
                    expression=expression,
                    fps=fps,
                    **self.overlay_options,
                )
                processed += 1
                if not self.viewer.show(output_frame):
                    break
        finally:
            self.text_input.stop()
            self.video_source.release()
            self.detector.close()
            self.viewer.close()
        return processed


def _compute_fps(previous_timestamp: float | None, current_timestamp: float) -> float:
    if previous_timestamp is None:
        return 0.0
    delta = current_timestamp - previous_timestamp
    if delta <= 0.0:
        return 0.0
    return 1.0 / delta
