from __future__ import annotations

import cv2


class OpenCvViewer:
    def __init__(self, window_name: str = "Robot Expression Demo") -> None:
        self.window_name = window_name

    def show(self, frame) -> bool:
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key not in (ord("q"), 27)

    def close(self) -> None:
        cv2.destroyAllWindows()

