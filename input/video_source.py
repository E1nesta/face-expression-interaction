from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

import cv2

from core.data_structures import FrameInput


class VideoSourceError(RuntimeError):
    pass


class VideoSource:
    def __init__(
        self,
        source: str,
        camera_id: int = 0,
        video_path: Path | str | None = None,
        frame_width: int | None = None,
        frame_height: int | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.source = source
        self.camera_id = camera_id
        self.video_path = Path(video_path) if video_path else None
        self.clock = clock
        self.frame_id = 0

        target = self._resolve_target()
        self.capture = cv2.VideoCapture(target)
        if not self.capture.isOpened():
            raise VideoSourceError(f"failed to open {source} source: {target}")

        if source == "camera":
            if frame_width is not None:
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
            if frame_height is not None:
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    def _resolve_target(self) -> int | str:
        if self.source == "camera":
            return self.camera_id

        if self.source == "video":
            if self.video_path is None:
                raise ValueError("video_path is required when source is 'video'")
            if not self.video_path.exists():
                raise FileNotFoundError(f"video file not found: {self.video_path}")
            return str(self.video_path)

        raise ValueError(f"unsupported video source: {self.source}")

    def read(self) -> FrameInput | None:
        ok, frame = self.capture.read()
        if not ok:
            return None

        self.frame_id += 1
        return FrameInput(
            frame_id=self.frame_id,
            timestamp=self.clock(),
            frame=frame,
            source=str(self.video_path) if self.source == "video" else self.source,
        )

    def release(self) -> None:
        self.capture.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()

