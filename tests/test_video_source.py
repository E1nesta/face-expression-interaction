from pathlib import Path

import numpy as np
import pytest

from input.video_source import VideoSource, VideoSourceError


class FakeCapture:
    def __init__(self, target, opened=True, frames=None):
        self.target = target
        self.opened = opened
        self.frames = list(frames or [])
        self.released = False
        self.properties = {}

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def set(self, prop, value):
        self.properties[prop] = value
        return True

    def release(self):
        self.released = True


def test_camera_source_reads_frame(monkeypatch):
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    captures = []

    def fake_video_capture(target):
        capture = FakeCapture(target, frames=[frame])
        captures.append(capture)
        return capture

    monkeypatch.setattr("input.video_source.cv2.VideoCapture", fake_video_capture)

    source = VideoSource(
        source="camera",
        camera_id=1,
        frame_width=640,
        frame_height=480,
        clock=lambda: 123.0,
    )
    result = source.read()
    source.release()

    assert captures[0].target == 1
    assert result is not None
    assert result.frame_id == 1
    assert result.timestamp == 123.0
    assert result.source == "camera"
    assert result.frame is frame
    assert captures[0].released is True


def test_video_source_reads_existing_file(monkeypatch, tmp_path):
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"fake")
    frame = np.ones((2, 3, 3), dtype=np.uint8)

    monkeypatch.setattr(
        "input.video_source.cv2.VideoCapture",
        lambda target: FakeCapture(target, frames=[frame]),
    )

    source = VideoSource(
        source="video",
        video_path=video_path,
        clock=lambda: 200.0,
    )

    result = source.read()
    end = source.read()

    assert result is not None
    assert Path(result.source) == video_path
    assert end is None


def test_video_source_rejects_missing_video_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        VideoSource(source="video", video_path=tmp_path / "missing.mp4")


def test_video_source_rejects_unopened_capture(monkeypatch):
    monkeypatch.setattr(
        "input.video_source.cv2.VideoCapture",
        lambda target: FakeCapture(target, opened=False),
    )

    with pytest.raises(VideoSourceError):
        VideoSource(source="camera", camera_id=0)

