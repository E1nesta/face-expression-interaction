import numpy as np

from app.pipeline import RobotExpressionPipeline
from core.data_structures import (
    EmotionClassificationResult,
    FaceLandmarkResult,
    FrameInput,
    TextInputState,
)


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


class FakeDetectedFaceDetector(FakeDetector):
    def detect(self, frame, timestamp_ms=0):
        return FaceLandmarkResult(
            face_detected=True,
            landmarks=[],
            image_width=4,
            image_height=4,
            timestamp_ms=timestamp_ms,
        )


class FakeNeutralTextInput(FakeTextInput):
    def latest_state(self):
        return TextInputState("", timestamp=1.0)


class FakeViewer:
    def __init__(self):
        self.frames = []
        self.closed = False

    def show(self, frame):
        self.frames.append(frame)
        return True

    def close(self):
        self.closed = True


class FakeEmotionClassifier:
    backend_name = "fake"

    def __init__(self, emotion="neutral"):
        self.emotion = emotion
        self.calls = []

    def classify(self, *, frame, face, features):
        self.calls.append((frame, face, features))
        return EmotionClassificationResult(label=self.emotion, confidence=0.9)


class PassthroughStabilizer:
    def update(self, result, *, face_detected):
        return result


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


def test_pipeline_uses_injected_classifier_and_overlay_options():
    source = FakeSource()
    text_input = FakeTextInput()
    detector = FakeDetector()
    viewer = FakeViewer()
    overlay_calls = []
    emotion_classifier = FakeEmotionClassifier()

    def fake_overlay(frame, **kwargs):
        overlay_calls.append(kwargs)
        return frame

    pipeline = RobotExpressionPipeline(
        video_source=source,
        text_input=text_input,
        detector=detector,
        viewer=viewer,
        write_command=lambda command: None,
        overlay_func=fake_overlay,
        emotion_classifier=emotion_classifier,
        overlay_options={
            "draw_landmarks": False,
            "draw_fps": True,
            "draw_status": False,
        },
    )

    processed = pipeline.run(max_frames=1)

    assert processed == 1
    assert len(emotion_classifier.calls) == 1
    assert overlay_calls[0]["draw_landmarks"] is False
    assert overlay_calls[0]["draw_fps"] is True
    assert overlay_calls[0]["draw_status"] is False


def test_pipeline_passes_face_label_mapping_to_fusion():
    source = FakeSource()
    text_input = FakeNeutralTextInput()
    detector = FakeDetectedFaceDetector()
    viewer = FakeViewer()
    payloads = []

    pipeline = RobotExpressionPipeline(
        video_source=source,
        text_input=text_input,
        detector=detector,
        viewer=viewer,
        write_command=payloads.append,
        overlay_func=lambda frame, **kwargs: frame,
        emotion_classifier=FakeEmotionClassifier("happiness"),
        emotion_stabilizer=PassthroughStabilizer(),
        face_label_mapping={"happiness": "happy"},
    )

    processed = pipeline.run(max_frames=1)

    assert processed == 1
    assert payloads[0].robot_state == "happy_reply"
    assert payloads[0].emotion.face_emotion == "happiness"


def test_pipeline_uses_stabilized_emotion_for_fusion():
    source = FakeSource()
    text_input = FakeNeutralTextInput()
    detector = FakeDetectedFaceDetector()
    viewer = FakeViewer()
    payloads = []

    class HoldHappyStabilizer:
        def update(self, result, *, face_detected):
            assert result.label == "sadness"
            return EmotionClassificationResult(label="happiness", confidence=0.8)

    pipeline = RobotExpressionPipeline(
        video_source=source,
        text_input=text_input,
        detector=detector,
        viewer=viewer,
        write_command=payloads.append,
        overlay_func=lambda frame, **kwargs: frame,
        emotion_classifier=FakeEmotionClassifier("sadness"),
        emotion_stabilizer=HoldHappyStabilizer(),
        face_label_mapping={"happiness": "happy", "sadness": "sad"},
    )

    processed = pipeline.run(max_frames=1)

    assert processed == 1
    assert payloads[0].robot_state == "happy_reply"
    assert payloads[0].raw_face_emotion == "sadness"
    assert payloads[0].emotion.face_emotion == "happiness"
