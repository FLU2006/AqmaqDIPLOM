from __future__ import annotations

from typing import List, Tuple
import numpy as np
from insightface.app import FaceAnalysis

BBox = Tuple[int, int, int, int]  # x, y, w, h


class InsightFaceDetector:
    def __init__(self, det_size=(640, 640)):
        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_size=det_size)

    def detect(self, frame_bgr: np.ndarray) -> List[Tuple[BBox, np.ndarray]]:
        faces = self.app.get(frame_bgr)
        results = []

        for face in faces:
            x1, y1, x2, y2 = face.bbox.astype(int).tolist()
            x1 = max(0, x1)
            y1 = max(0, y1)
            w = max(1, x2 - x1)
            h = max(1, y2 - y1)

            crop = frame_bgr[y1:y1 + h, x1:x1 + w].copy()
            if crop.size == 0:
                continue

            results.append(((x1, y1, w, h), crop))

        return results