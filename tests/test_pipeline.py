import numpy as np

from app.pipeline import RobotExpressionPipeline
from core.data_structures import FaceLandmarkResult, FrameInput, TextInputState


class FakeSource:
    def __init__(self):
        self.frames = [
            FrameInput(
                frame_id=1,
                timestamp=1.0,
                frame=np.zeros((4, 4, 3), dtype=np.uint8),
                source="test",
            )
        ]
        self.released = False

    def read(self):
        if self.frames:
            return self.frames.pop(0)
        return None

    def release(self):
        self.released = True


class FakeTextInput:
    def __init__(self):
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def latest_state(self):
        return TextInputState("开心", timestamp=1.0)

    def stop(self):
        self.stopped = True


class FakeDetector:
    def __init__(self):
        self.closed = False

    def detect(self, frame, timestamp_ms=0):
        return FaceLandmarkResult(
            face_detected=False,
            image_width=4,
            image_height=4,
            timestamp_ms=timestamp_ms,
        )

    def close(self):
        self.closed = True


class FakeViewer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def show(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        self.closed = True


def test_pipeline_runs_one_frame_and_releases_resources():
    source = FakeSource()
    text_input = FakeTextInput()
    detector = FakeDetector()
    viewer = FakeViewer()
    payloads = []

    pipeline = RobotExpressionPipeline(
        video_source=source,
        text_input=text_input,
        detector=detector,
        viewer=viewer,
        write_command=payloads.append,
        overlay_func=lambda frame, **kwargs: frame,
    )

    processed = pipeline.run(max_frames=1)

    assert processed == 1
    assert text_input.started is True
    assert text_input.stopped is True
    assert source.released is True
    assert detector.closed is True
    assert viewer.closed is True
    assert len(payloads) == 1
    assert payloads[0].robot_state == "idle"
    assert viewer.frames


def test_pipeline_passes_configured_thresholds_and_overlay_options(monkeypatch):
    source = FakeSource()
    text_input = FakeTextInput()
    detector = FakeDetector()
    viewer = FakeViewer()
    thresholds = {"happy_smile": 0.2}
    overlay_calls = []
    classifier_calls = []

    def fake_classifier(features, thresholds=None):
        classifier_calls.append(thresholds)
        return "neutral"

    def fake_overlay(frame, **kwargs):
        overlay_calls.append(kwargs)
        return frame

    monkeypatch.setattr("app.pipeline.classify_face_emotion", fake_classifier)

    pipeline = RobotExpressionPipeline(
        video_source=source,
        text_input=text_input,
        detector=detector,
        viewer=viewer,
        write_command=lambda command: None,
        overlay_func=fake_overlay,
        face_emotion_thresholds=thresholds,
        overlay_options={
            "draw_landmarks": False,
            "draw_fps": True,
            "draw_status": False,
        },
    )

    processed = pipeline.run(max_frames=1)

    assert processed == 1
    assert classifier_calls == [thresholds]
    assert overlay_calls[0]["draw_landmarks"] is False
    assert overlay_calls[0]["draw_fps"] is True
    assert overlay_calls[0]["draw_status"] is False
