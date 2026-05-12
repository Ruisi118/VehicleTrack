"""Homography utilities for VehicleTrack.

The calibration maps image-plane pixels into a bird's-eye-view road plane
measured in meters. Vehicle speed should be computed from the projected
ground point, not from raw pixel displacement.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


Point = tuple[float, float]
BBox = Sequence[float]


@dataclass(frozen=True)
class HomographyCalibration:
    """A projective transform from image pixels to world meters."""

    source_points: np.ndarray
    destination_points: np.ndarray
    transform: np.ndarray
    metadata: dict

    @classmethod
    def from_points(
        cls,
        source_points: Iterable[Sequence[float]],
        destination_points: Iterable[Sequence[float]],
        metadata: Mapping | None = None,
    ) -> "HomographyCalibration":
        src = _as_four_points(source_points, "source_points")
        dst = _as_four_points(destination_points, "destination_points")
        transform = compute_homography(src, dst)
        return cls(
            source_points=src,
            destination_points=dst,
            transform=transform,
            metadata=dict(metadata or {}),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HomographyCalibration | MultiSegmentCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if "segments" in data:
            return MultiSegmentCalibration.from_dict(data)

        src = _as_four_points(data["source_points"], "source_points")
        dst = _as_four_points(data["destination_points"], "destination_points")

        if "transform" in data:
            transform = np.asarray(data["transform"], dtype=float)
            if transform.shape != (3, 3):
                raise ValueError("transform must be a 3x3 matrix")
        else:
            transform = compute_homography(src, dst)

        metadata = {
            key: value
            for key, value in data.items()
            if key not in {"source_points", "destination_points", "transform"}
        }
        return cls(src, dst, transform, metadata)

    def to_dict(self) -> dict:
        return {
            **self.metadata,
            "source_points": self.source_points.tolist(),
            "destination_points": self.destination_points.tolist(),
            "transform": self.transform.tolist(),
        }

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def pixel_to_world(self, x: float, y: float) -> Point:
        return pixel_to_world(self.transform, x, y)

    def bbox_to_ground_point(self, bbox: BBox) -> Point:
        x, y = bbox_bottom_midpoint(bbox)
        return self.pixel_to_world(x, y)

    def project_bbox(self, bbox: BBox) -> dict:
        x, y = bbox_bottom_midpoint(bbox)
        world_x, world_y = self.pixel_to_world(x, y)
        return {
            "world_x": world_x,
            "world_y": world_y,
            "in_calibration_roi": self.contains_pixel(x, y),
            "calibration_segment": self.metadata.get("segment_name", "single_roi"),
        }

    def contains_pixel(self, x: float, y: float) -> bool:
        return point_in_polygon((x, y), self.source_points)

    def bbox_in_source_roi(self, bbox: BBox) -> bool:
        x, y = bbox_bottom_midpoint(bbox)
        return self.contains_pixel(x, y)


@dataclass(frozen=True)
class MultiSegmentCalibration:
    """A set of local Homography transforms for a curved or extended road area."""

    segments: list[HomographyCalibration]
    metadata: dict

    @classmethod
    def from_dict(cls, data: Mapping) -> "MultiSegmentCalibration":
        segments = []
        for idx, item in enumerate(data.get("segments", []), start=1):
            name = str(item.get("name") or f"segment_{idx}")
            segment_metadata = {
                key: value
                for key, value in item.items()
                if key not in {"source_points", "destination_points", "transform"}
            }
            segment_metadata["segment_name"] = name
            segments.append(
                HomographyCalibration.from_points(
                    item["source_points"],
                    item["destination_points"],
                    metadata=segment_metadata,
                )
            )
        if not segments:
            raise ValueError("multi-segment calibration must contain at least one segment")

        metadata = {
            key: value
            for key, value in data.items()
            if key != "segments"
        }
        return cls(segments=segments, metadata=metadata)

    def to_dict(self) -> dict:
        return {
            **self.metadata,
            "segments": [
                {
                    **segment.metadata,
                    "source_points": segment.source_points.tolist(),
                    "destination_points": segment.destination_points.tolist(),
                    "transform": segment.transform.tolist(),
                }
                for segment in self.segments
            ],
        }

    def save(self, path: str | Path) -> None:
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def select_segment(self, x: float, y: float) -> HomographyCalibration | None:
        for segment in self.segments:
            if segment.contains_pixel(x, y):
                return segment
        return None

    def pixel_to_world(self, x: float, y: float) -> Point:
        segment = self.select_segment(x, y)
        if segment is None:
            return float("nan"), float("nan")
        return segment.pixel_to_world(x, y)

    def bbox_to_ground_point(self, bbox: BBox) -> Point:
        x, y = bbox_bottom_midpoint(bbox)
        return self.pixel_to_world(x, y)

    def project_bbox(self, bbox: BBox) -> dict:
        x, y = bbox_bottom_midpoint(bbox)
        segment = self.select_segment(x, y)
        if segment is None:
            return {
                "world_x": float("nan"),
                "world_y": float("nan"),
                "in_calibration_roi": False,
                "calibration_segment": "",
            }
        world_x, world_y = segment.pixel_to_world(x, y)
        return {
            "world_x": world_x,
            "world_y": world_y,
            "in_calibration_roi": True,
            "calibration_segment": segment.metadata.get("segment_name", ""),
        }

    def contains_pixel(self, x: float, y: float) -> bool:
        return self.select_segment(x, y) is not None

    def bbox_in_source_roi(self, bbox: BBox) -> bool:
        x, y = bbox_bottom_midpoint(bbox)
        return self.contains_pixel(x, y)


def compute_homography(source_points: np.ndarray, destination_points: np.ndarray) -> np.ndarray:
    """Compute a 3x3 homography matrix from four point pairs.

    This uses the standard Direct Linear Transform formulation. It avoids
    requiring OpenCV inside the pure geometry module, which keeps tests light.
    """

    src = _as_four_points(source_points, "source_points")
    dst = _as_four_points(destination_points, "destination_points")

    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y, -u])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y, -v])

    _, _, vh = np.linalg.svd(np.asarray(rows, dtype=float))
    h = vh[-1].reshape(3, 3)
    if np.isclose(h[2, 2], 0.0):
        raise ValueError("Degenerate point configuration; cannot normalize homography")
    return h / h[2, 2]


def pixel_to_world(transform: np.ndarray, x: float, y: float) -> Point:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (3, 3):
        raise ValueError("transform must be a 3x3 matrix")

    projected = matrix @ np.asarray([x, y, 1.0], dtype=float)
    scale = projected[2]
    if np.isclose(scale, 0.0):
        raise ValueError("Point projects to infinity under this homography")

    world = projected[:2] / scale
    return float(world[0]), float(world[1])


def bbox_bottom_midpoint(bbox: BBox) -> Point:
    if len(bbox) != 4:
        raise ValueError("bbox must contain four values: x1, y1, x2, y2")
    x1, _y1, x2, y2 = (float(value) for value in bbox)
    return (x1 + x2) / 2.0, y2


def estimate_speed_kmh(records: Sequence[Mapping[str, float]], window: int = 10) -> float:
    """Estimate speed from timestamped world-coordinate trajectory records."""

    if len(records) < 2:
        return 0.0

    recent = list(records[-window:])
    first = recent[0]
    last = recent[-1]
    dt = float(last["timestamp"]) - float(first["timestamp"])
    if dt <= 0:
        return 0.0

    dx = float(last["world_x"]) - float(first["world_x"])
    dy = float(last["world_y"]) - float(first["world_y"])
    return (float(np.hypot(dx, dy)) / dt) * 3.6


def point_in_polygon(point: Point, polygon: np.ndarray) -> bool:
    """Return whether a 2D point is inside a polygon, including the boundary."""

    x, y = point
    points = np.asarray(polygon, dtype=float)
    inside = False
    count = len(points)
    for idx in range(count):
        x1, y1 = points[idx]
        x2, y2 = points[(idx + 1) % count]

        cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
        if np.isclose(cross, 0.0) and min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
            return True

        if (y1 > y) != (y2 > y):
            x_intersect = ((x2 - x1) * (y - y1) / (y2 - y1)) + x1
            if x <= x_intersect:
                inside = not inside
    return inside


def _as_four_points(points: Iterable[Sequence[float]], name: str) -> np.ndarray:
    arr = np.asarray(list(points), dtype=float)
    if arr.shape != (4, 2):
        raise ValueError(f"{name} must contain exactly four (x, y) points")
    return arr
