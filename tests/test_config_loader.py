from pathlib import Path

from core.config_loader import load_config, merge_cli_overrides


def test_load_default_config_contains_v1_face_landmarker_settings():
    config = load_config(Path("config/default.json"))

    assert config["source"] == "camera"
    assert config["face_detector"]["backend"] == "mediapipe_face_landmarker"
    assert config["face_detector"]["model_path"] == "models/face_landmarker.task"
    assert config["face_detector"]["running_mode"] == "video"
    assert config["decision"]["stable_frames"] == 5


def test_merge_cli_overrides_only_replaces_provided_values():
    config = load_config(Path("config/default.json"))

    merged = merge_cli_overrides(
        config,
        {
            "source": "video",
            "camera_id": None,
            "video_path": "samples/demo.mp4",
        },
    )

    assert merged["source"] == "video"
    assert merged["camera_id"] == config["camera_id"]
    assert merged["video_path"] == "samples/demo.mp4"

