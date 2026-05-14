from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


BBox = Tuple[int, int, int, int]  # x, y, w, h


def bbox_center(box: BBox) -> Tuple[float, float]:
    x, y, w, h = box
    return x + w / 2.0, y + h / 2.0


def bbox_diag(box: BBox) -> float:
    _, _, w, h = box
    return float((w ** 2 + h ** 2) ** 0.5)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b

    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0
    return inter_area / union


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


@dataclass
class TrackState:
    track_id: int
    bbox: BBox
    first_seen_ts: float
    last_seen_ts: float

    status: str = "pending"  # pending | known | unknown
    person_id: Optional[str] = None
    display_name: Optional[str] = None

    samples: List[np.ndarray] = field(default_factory=list)
    best_score: float = -1.0
    attempts: int = 0

    last_capture_ts: float = 0.0
    last_event_ts: float = 0.0
    representative_emb: Optional[np.ndarray] = None

    matched_in_frame: bool = False
    
    # Коэффициент сглаживания EMA (от 0 до 1). 
    # 0.7 означает, что новая рамка берет 70% от новых координат и 30% от старых.
    smooth_alpha: float = 0.7 

    def update_bbox(self, new_bbox: BBox):
        """Сглаживание перемещения рамки (Exponential Moving Average)."""
        if self.bbox is None:
            self.bbox = new_bbox
            return
        
        nx, ny, nw, nh = new_bbox
        ox, oy, ow, oh = self.bbox
        
        # Плавно меняем координаты
        sx = int(ox * (1 - self.smooth_alpha) + nx * self.smooth_alpha)
        sy = int(oy * (1 - self.smooth_alpha) + ny * self.smooth_alpha)
        sw = int(ow * (1 - self.smooth_alpha) + nw * self.smooth_alpha)
        sh = int(oh * (1 - self.smooth_alpha) + nh * self.smooth_alpha)
        
        self.bbox = (sx, sy, sw, sh)

    def add_embedding(self, emb: np.ndarray):
        emb = emb / (np.linalg.norm(emb) + 1e-9)
        self.samples.append(emb)
        self.attempts += 1

    def mean_embedding(self) -> Optional[np.ndarray]:
        if not self.samples:
            return None
        mean = np.mean(np.stack(self.samples, axis=0), axis=0)
        mean = mean / (np.linalg.norm(mean) + 1e-9)
        return mean


class TrackManager:
    def __init__(
        self,
        track_timeout_sec: float,
        max_center_distance_ratio: float,
        min_iou_for_match: float,
    ):
        self.track_timeout_sec = track_timeout_sec
        self.max_center_distance_ratio = max_center_distance_ratio
        self.min_iou_for_match = min_iou_for_match

        self.tracks: Dict[int, TrackState] = {}
        self._next_track_id = 1

    def begin_frame(self):
        for track in self.tracks.values():
            track.matched_in_frame = False

    def cleanup(self, now_ts: float):
        expired = []
        for track_id, track in self.tracks.items():
            if now_ts - track.last_seen_ts > self.track_timeout_sec:
                expired.append(track_id)

        for track_id in expired:
            del self.tracks[track_id]

    def _can_match(self, track: TrackState, bbox: BBox, now_ts: float) -> bool:
        if track.matched_in_frame:
            return False
        if now_ts - track.last_seen_ts > self.track_timeout_sec:
            return False

        iou = bbox_iou(track.bbox, bbox)
        if iou >= self.min_iou_for_match:
            return True

        cx1, cy1 = bbox_center(track.bbox)
        cx2, cy2 = bbox_center(bbox)
        dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        limit = max(bbox_diag(track.bbox), bbox_diag(bbox)) * self.max_center_distance_ratio
        return dist <= limit

    def assign(self, bbox: BBox, now_ts: float) -> TrackState:
        best_track: Optional[TrackState] = None
        best_cost = float("inf")

        for track in self.tracks.values():
            if not self._can_match(track, bbox, now_ts):
                continue

            iou = bbox_iou(track.bbox, bbox)
            cx1, cy1 = bbox_center(track.bbox)
            cx2, cy2 = bbox_center(bbox)
            dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5

            # Меньше cost = лучше
            cost = dist - iou * 100.0
            if cost < best_cost:
                best_cost = cost
                best_track = track

        if best_track is None:
            track = TrackState(
                track_id=self._next_track_id,
                bbox=bbox,
                first_seen_ts=now_ts,
                last_seen_ts=now_ts,
            )
            track.matched_in_frame = True
            self.tracks[track.track_id] = track
            self._next_track_id += 1
            return track

        # Здесь применяем сглаживание вместо жесткой перезаписи
        best_track.update_bbox(bbox)
        best_track.last_seen_ts = now_ts
        best_track.matched_in_frame = True
        return best_track

    def get(self, track_id: int) -> Optional[TrackState]:
        return self.tracks.get(track_id)