from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.data_structures import FaceLandmarkResult, FacialFeatures
from emotion.classifier import (
    DEFAULT_FACE_EMOTION_LABELS,
    EmotionClassifierError,
    create_emotion_classifier,
    validate_face_emotion_label,
)
from emotion.onnx_classifier import OnnxEmotionClassifier
from emotion.rule_classifier import RuleEmotionClassifier


def test_default_face_emotion_labels_include_open_source_model_labels():
    assert "happiness" in DEFAULT_FACE_EMOTION_LABELS
    assert "anger" in DEFAULT_FACE_EMOTION_LABELS
    assert "contempt" in DEFAULT_FACE_EMOTION_LABELS
    assert validate_face_emotion_label("Happiness") == "happiness"

    with pytest.raises(EmotionClassifierError, match="Unsupported emotion label"):
        validate_face_emotion_label("no_face")


def test_rule_backend_uses_configured_thresholds():
    classifier = create_emotion_classifier(
        {
            "face_backend": "rule",
            "thresholds": {"happy_smile": 0.2},
        }
    )

    emotion = classifier.classify(
        frame=None,
        face=FaceLandmarkResult(face_detected=True),
        features=FacialFeatures(smile_score=0.3, eye_open=0.7),
    )

    assert isinstance(classifier, RuleEmotionClassifier)
    assert emotion.label == "happy"
    assert emotion.confidence == 0.7


def test_unknown_emotion_backend_raises_clear_error():
    with pytest.raises(EmotionClassifierError, match="Unsupported emotion backend"):
        create_emotion_classifier({"face_backend": "unknown"})


def test_onnx_backend_missing_model_falls_back_to_rule(tmp_path: Path):
    classifier = create_emotion_classifier(
        {
            "face_backend": "onnx",
            "fallback_to_rule": True,
            "thresholds": {"happy_smile": 0.2},
            "onnx": {
                "model_path": str(tmp_path / "missing.onnx"),
                "labels": ["neutral", "happy"],
            },
        }
    )

    assert isinstance(classifier, RuleEmotionClassifier)


def test_onnx_backend_missing_model_without_fallback_raises(tmp_path: Path):
    with pytest.raises(EmotionClassifierError, match="ONNX emotion model not found"):
        create_emotion_classifier(
            {
                "face_backend": "onnx",
                "fallback_to_rule": False,
                "onnx": {
                    "model_path": str(tmp_path / "missing.onnx"),
                    "labels": ["neutral", "happy"],
                },
            }
        )


def test_onnx_classifier_maps_feature_vector_to_open_source_label(
    tmp_path: Path,
    monkeypatch,
):
    model_path = tmp_path / "emotion.onnx"
    model_path.write_bytes(b"placeholder")
    captured_feeds = []

    class FakeSession:
        def __init__(self, path):
            assert path == str(model_path)

        def get_inputs(self):
            return [SimpleNamespace(name="features")]

        def run(self, _output_names, feeds):
            captured_feeds.append(feeds)
            return [np.array([[0.1, 0.9]], dtype=np.float32)]

    fake_ort = SimpleNamespace(InferenceSession=FakeSession)
    monkeypatch.setattr(
        "emotion.onnx_classifier.importlib.import_module",
        lambda name: fake_ort,
    )

    classifier = OnnxEmotionClassifier(
        {
            "model_path": str(model_path),
            "input_type": "features",
            "labels": ["neutral", "anger"],
        }
    )

    emotion = classifier.classify(
        frame=None,
        face=FaceLandmarkResult(face_detected=True),
        features=FacialFeatures(
            mouth_open=0.1,
            smile_score=0.8,
            eye_open=0.7,
            brow_raise=0.2,
            head_yaw=1.0,
            head_pitch=-2.0,
            head_roll=0.5,
        ),
    )

    assert emotion.label == "anger"
    assert emotion.confidence == pytest.approx(0.689974, rel=1e-5)
    assert emotion.scores["neutral"] == pytest.approx(0.310025, rel=1e-5)
    assert emotion.scores["anger"] == pytest.approx(0.689974, rel=1e-5)
    assert captured_feeds[0]["features"].shape == (1, 7)


def test_onnx_classifier_supports_face_image_input(tmp_path: Path, monkeypatch):
    model_path = tmp_path / "emotion.onnx"
    model_path.write_bytes(b"placeholder")
    captured_feeds = []

    class FakeSession:
        def __init__(self, path):
            assert path == str(model_path)

        def get_inputs(self):
            return [SimpleNamespace(name="image")]

        def run(self, _output_names, feeds):
            captured_feeds.append(feeds)
            return [np.array([[0.1, 0.9]], dtype=np.float32)]

    fake_ort = SimpleNamespace(InferenceSession=FakeSession)
    monkeypatch.setattr(
        "emotion.onnx_classifier.importlib.import_module",
        lambda name: fake_ort,
    )
    landmarks = [(0.5, 0.5, 0.0) for _ in range(468)]
    landmarks[0] = (0.25, 0.25, 0.0)
    landmarks[1] = (0.75, 0.75, 0.0)
    face = FaceLandmarkResult(
        face_detected=True,
        landmarks=landmarks,
        image_width=80,
        image_height=60,
    )
    frame = np.zeros((60, 80, 3), dtype=np.uint8)

    classifier = OnnxEmotionClassifier(
        {
            "model_path": str(model_path),
            "input_type": "face_image",
            "image_size": [64, 64],
            "color_mode": "grayscale",
            "labels": ["neutral", "happiness"],
        }
    )

    emotion = classifier.classify(
        frame=frame,
        face=face,
        features=FacialFeatures(),
    )

    assert emotion.label == "happiness"
    assert emotion.confidence == pytest.approx(0.689974, rel=1e-5)
    assert captured_feeds[0]["image"].shape == (1, 1, 64, 64)
