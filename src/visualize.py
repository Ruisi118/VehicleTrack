"""Video annotation helpers for VehicleTrack."""

from __future__ import annotations

import cv2
import numpy as np

from tracking import Track


def draw_tracks(frame_bgr, tracks: list[Track]):
    annotated = frame_bgr.copy()
    for track in tracks:
        if track.missed > 0:
            continue
        x1, y1, x2, y2 = (int(value) for value in track.bbox)
        color = track_color(track.track_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        if track.in_calibration_roi and track.speed_kmh > 0:
            speed_label = f"{track.speed_kmh:.1f} km/h"
        else:
            speed_label = "-- km/h"
        label = f"{track.label} ID {track.track_id} {speed_label}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            4,
        )
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        points = [
            (
                int((record["x1"] + record["x2"]) / 2),
                int(record["y2"]),
            )
            for record in track.records[-30:]
        ]
        if len(points) >= 2:
            cv2.polylines(
                annotated,
                [np.asarray(points, dtype=np.int32)],
                isClosed=False,
                color=color,
                thickness=2,
            )
    return annotated


def draw_frame_status(frame_bgr, frame_idx: int, fps: float, active_tracks: int):
    text = f"frame {frame_idx} | fps {fps:.1f} | active tracks {active_tracks}"
    cv2.putText(frame_bgr, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(frame_bgr, text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame_bgr


def track_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id)
    color = rng.integers(low=80, high=255, size=3)
    return int(color[0]), int(color[1]), int(color[2])
