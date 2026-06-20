"""Lightweight face tracking by bounding-box overlap."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass
class _Track:
    track_id: int
    region: Any
    missed: int = 0


class FaceTracker:
    """Assign stable integer IDs to detected faces across nearby frames."""

    def __init__(self, iou_threshold: float = 0.25, max_missed: int = 10) -> None:
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self._next_id = 1
        self._tracks: list[_Track] = []

    def assign(self, regions: Sequence[Any]) -> list[int]:
        if not regions:
            self._age_unmatched(set())
            return []

        assignments: list[int | None] = [None] * len(regions)
        matched_tracks: set[int] = set()

        for region_index, region in enumerate(regions):
            best_track_index = None
            best_iou = 0.0
            for track_index, track in enumerate(self._tracks):
                if track_index in matched_tracks:
                    continue
                overlap = _region_iou(region, track.region)
                if overlap > best_iou:
                    best_iou = overlap
                    best_track_index = track_index

            if best_track_index is not None and best_iou >= self.iou_threshold:
                track = self._tracks[best_track_index]
                track.region = region
                track.missed = 0
                assignments[region_index] = track.track_id
                matched_tracks.add(best_track_index)

        for region_index, region in enumerate(regions):
            if assignments[region_index] is not None:
                continue
            track = _Track(track_id=self._next_id, region=region)
            self._next_id += 1
            self._tracks.append(track)
            assignments[region_index] = track.track_id
            matched_tracks.add(len(self._tracks) - 1)

        self._age_unmatched(matched_tracks)
        return [int(track_id) for track_id in assignments if track_id is not None]

    def reset(self) -> None:
        self._next_id = 1
        self._tracks.clear()

    def _age_unmatched(self, matched_tracks: set[int]) -> None:
        for track_index, track in enumerate(self._tracks):
            if track_index not in matched_tracks:
                track.missed += 1
        self._tracks = [track for track in self._tracks if track.missed <= self.max_missed]


def _region_iou(a: Any, b: Any) -> float:
    ax1, ay1, ax2, ay2 = _region_bounds(a)
    bx1, by1, bx2, by2 = _region_bounds(b)

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _region_bounds(region: Any) -> tuple[int, int, int, int]:
    x = int(getattr(region, "x", 0))
    y = int(getattr(region, "y", 0))
    w = int(getattr(region, "w", 0))
    h = int(getattr(region, "h", 0))
    return x, y, x + w, y + h
