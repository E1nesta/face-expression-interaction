import numpy as np

from demo.opencv_viewer import OpenCvViewer
from demo.overlay import draw_overlay
from core.data_structures import (
    DecisionResult,
    EmotionResult,
    ExpressionCommand,
    FaceLandmarkResult,
)


def test_draw_overlay_returns_frame_and_draws_status(monkeypatch):
    calls = []
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    face = FaceLandmarkResult(
        face_detected=True,
        landmarks=[(0.5, 0.5, 0.0)],
        image_width=10,
        image_height=10,
    )

    monkeypatch.setattr("demo.overlay.cv2.putText", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr("demo.overlay.cv2.circle", lambda *args, **kwargs: calls.append(args))

    result = draw_overlay(
        frame,
        face=face,
        emotion=EmotionResult(user_state="happy"),
        decision=DecisionResult(robot_state="happy_reply", expression="happy_smile"),
        expression=ExpressionCommand(
            expression="happy_smile",
            intensity=0.75,
            duration_ms=1500,
            transition_ms=400,
        ),
        fps=30.0,
    )

    assert result is frame
    assert calls


def test_draw_overlay_respects_draw_flags(monkeypatch):
    calls = []
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    face = FaceLandmarkResult(
        face_detected=True,
        landmarks=[(0.5, 0.5, 0.0)],
        image_width=10,
        image_height=10,
    )

    monkeypatch.setattr("demo.overlay.cv2.putText", lambda *args, **kwargs: calls.append(args))
    monkeypatch.setattr("demo.overlay.cv2.circle", lambda *args, **kwargs: calls.append(args))

    result = draw_overlay(
        frame,
        face=face,
        emotion=EmotionResult(user_state="happy"),
        decision=DecisionResult(robot_state="happy_reply", expression="happy_smile"),
        expression=ExpressionCommand(
            expression="happy_smile",
            intensity=0.75,
            duration_ms=1500,
            transition_ms=400,
        ),
        fps=30.0,
        draw_landmarks=False,
        draw_fps=False,
        draw_status=False,
    )

    assert result is frame
    assert calls == []


def test_viewer_show_returns_false_on_quit(monkeypatch):
    shown = []
    destroyed = []
    viewer = OpenCvViewer(window_name="test")

    monkeypatch.setattr("demo.opencv_viewer.cv2.imshow", lambda name, frame: shown.append(name))
    monkeypatch.setattr("demo.opencv_viewer.cv2.waitKey", lambda delay: ord("q"))
    monkeypatch.setattr("demo.opencv_viewer.cv2.destroyAllWindows", lambda: destroyed.append(True))

    keep_running = viewer.show(np.zeros((2, 2, 3), dtype=np.uint8))
    viewer.close()

    assert keep_running is False
    assert shown == ["test"]
    assert destroyed == [True]
