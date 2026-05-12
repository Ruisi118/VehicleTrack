"""Tracking backends for VehicleTrack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import warnings

from detection import Detection
from homography import HomographyCalibration, bbox_bottom_midpoint, estimate_speed_kmh
import numpy as np


BBox = tuple[float, float, float, float]
MAX_REASONABLE_SPEED_KMH = 180.0
CROSS_SEGMENT_WINDOW = 6
MAX_SPEED_DELTA_KMH = 50.0
SPEED_EMA_ALPHA = 0.4


@dataclass
class Track:
    track_id: int
    bbox: BBox
    label: str
    score: float
    created_frame_idx: int
    missed: int = 0
    records: list[dict] = field(default_factory=list)
    speed_kmh: float = 0.0
    in_calibration_roi: bool = False
    calibration_segment: str = ""

    def update(
        self,
        detection: Detection,
        frame_idx: int,
        timestamp: float,
        calibration: HomographyCalibration,
    ) -> None:
        self.bbox = detection.bbox
        self.label = detection.label
        self.score = detection.score
        self.missed = 0
        ground_px_x, ground_px_y = bbox_bottom_midpoint(detection.bbox)
        projection = calibration.project_bbox(detection.bbox)
        world_x = projection["world_x"]
        world_y = projection["world_y"]
        in_calibration_roi = projection["in_calibration_roi"]
        calibration_segment = projection["calibration_segment"]
        self.in_calibration_roi = in_calibration_roi
        self.calibration_segment = calibration_segment
        x1, y1, x2, y2 = detection.bbox
        record = {
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "track_id": self.track_id,
            "label": self.label,
            "score": self.score,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "world_x": world_x,
            "world_y": world_y,
            "in_calibration_roi": in_calibration_roi,
            "calibration_segment": calibration_segment,
        }
        self.records.append(record)
        speed, raw_speed, speed_source = self._estimate_display_speed(
            in_calibration_roi,
            calibration_segment,
        )
        self.speed_kmh = speed
        self.records[-1]["speed_kmh"] = speed
        self.records[-1]["raw_speed_kmh"] = raw_speed
        self.records[-1]["speed_source"] = speed_source

    def _estimate_display_speed(
        self,
        in_calibration_roi: bool,
        calibration_segment: str,
    ) -> tuple[float, float, str]:
        if not in_calibration_roi:
            return 0.0, 0.0, "unavailable"

        same_segment_records = [
            item
            for item in self.records
            if item.get("in_calibration_roi")
            and item.get("calibration_segment") == calibration_segment
        ]
        same_segment_speed = estimate_speed_kmh(same_segment_records)
        if same_segment_speed > 0:
            return self._filter_and_smooth_speed(same_segment_speed, "same_segment")

        cross_segment_records = [
            item
            for item in self.records
            if item.get("in_calibration_roi")
        ][-CROSS_SEGMENT_WINDOW:]
        cross_segment_speed = estimate_speed_kmh(
            cross_segment_records,
            window=CROSS_SEGMENT_WINDOW,
        )
        if cross_segment_speed > 0:
            return self._filter_and_smooth_speed(
                cross_segment_speed,
                "cross_segment_smoothed",
            )

        return 0.0, 0.0, "unavailable"

    def _filter_and_smooth_speed(self, raw_speed: float, source: str) -> tuple[float, float, str]:
        if raw_speed > MAX_REASONABLE_SPEED_KMH:
            return 0.0, raw_speed, "filtered_outlier"

        previous_speed = self._previous_display_speed()
        if (
            source == "cross_segment_smoothed"
            and previous_speed > 0
            and abs(raw_speed - previous_speed) > MAX_SPEED_DELTA_KMH
        ):
            return 0.0, raw_speed, "filtered_outlier"

        if source == "cross_segment_smoothed" and previous_speed > 0:
            smoothed = (SPEED_EMA_ALPHA * raw_speed) + ((1.0 - SPEED_EMA_ALPHA) * previous_speed)
            return smoothed, raw_speed, source

        return raw_speed, raw_speed, source

    def _previous_display_speed(self) -> float:
        for record in reversed(self.records[:-1]):
            speed = float(record.get("speed_kmh", 0.0) or 0.0)
            if speed > 0:
                return speed
        return 0.0


class IoUTracker:
    """Greedy IoU tracker matching the notebook baseline behavior."""

    def __init__(
        self,
        calibration: HomographyCalibration,
        iou_threshold: float = 0.3,
        max_missed: int = 20,
    ) -> None:
        self.calibration = calibration
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed
        self.tracks: dict[int, Track] = {}
        self.archived_records: list[dict] = []
        self.next_track_id = 1

    def update(
        self,
        detections: Iterable[Detection],
        frame_idx: int,
        timestamp: float,
    ) -> list[Track]:
        detections = list(detections)
        for track in self.tracks.values():
            track.missed += 1

        matches, unmatched_detections = associate_tracks(
            self.tracks,
            detections,
            self.iou_threshold,
        )

        for track_id, detection_idx in matches:
            self.tracks[track_id].update(
                detections[detection_idx],
                frame_idx,
                timestamp,
                self.calibration,
            )

        for detection_idx in unmatched_detections:
            detection = detections[detection_idx]
            track = Track(
                track_id=self.next_track_id,
                bbox=detection.bbox,
                label=detection.label,
                score=detection.score,
                created_frame_idx=frame_idx,
            )
            track.update(detection, frame_idx, timestamp, self.calibration)
            self.tracks[self.next_track_id] = track
            self.next_track_id += 1

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed > self.max_missed
        ]
        for track_id in expired:
            self.archived_records.extend(self.tracks[track_id].records)
            del self.tracks[track_id]

        return list(self.tracks.values())

    def all_records(self) -> list[dict]:
        records: list[dict] = list(self.archived_records)
        for track in self.tracks.values():
            records.extend(track.records)
        return records

    @property
    def total_tracks_created(self) -> int:
        return self.next_track_id - 1


class ByteTrackTracker:
    """ByteTrack adapter that emits the same Track records as the IoU baseline."""

    def __init__(
        self,
        calibration: HomographyCalibration,
        max_missed: int = 20,
    ) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            import supervision as sv

            self._sv = sv
            self._tracker = sv.ByteTrack()
        self.calibration = calibration
        self.max_missed = max_missed
        self.tracks: dict[int, Track] = {}
        self.archived_records: list[dict] = []
        self.seen_track_ids: set[int] = set()

    def update(
        self,
        detections: Iterable[Detection],
        frame_idx: int,
        timestamp: float,
    ) -> list[Track]:
        detections = list(detections)
        for track in self.tracks.values():
            track.missed += 1

        tracked_detections = self._tracker.update_with_detections(
            detections_to_supervision(self._sv, detections)
        )
        for idx in range(len(tracked_detections)):
            track_id = int(tracked_detections.tracker_id[idx])
            bbox = tuple(float(value) for value in tracked_detections.xyxy[idx])
            class_id = int(tracked_detections.class_id[idx])
            score = float(tracked_detections.confidence[idx])
            label = str(tracked_detections.data["label"][idx])
            detection = Detection(
                bbox=bbox,
                label=label,
                score=score,
                class_id=class_id,
            )
            if track_id not in self.tracks:
                self.tracks[track_id] = Track(
                    track_id=track_id,
                    bbox=bbox,
                    label=label,
                    score=score,
                    created_frame_idx=frame_idx,
                )
                self.seen_track_ids.add(track_id)
            self.tracks[track_id].update(
                detection,
                frame_idx,
                timestamp,
                self.calibration,
            )

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if track.missed > self.max_missed
        ]
        for track_id in expired:
            self.archived_records.extend(self.tracks[track_id].records)
            del self.tracks[track_id]

        return list(self.tracks.values())

    def all_records(self) -> list[dict]:
        records: list[dict] = list(self.archived_records)
        for track in self.tracks.values():
            records.extend(track.records)
        return records

    @property
    def total_tracks_created(self) -> int:
        return len(self.seen_track_ids)


def detections_to_supervision(sv, detections: list[Detection]):
    if not detections:
        return sv.Detections(
            xyxy=np.empty((0, 4), dtype=float),
            confidence=np.empty((0,), dtype=float),
            class_id=np.empty((0,), dtype=int),
            data={"label": np.empty((0,), dtype=str)},
        )

    return sv.Detections(
        xyxy=np.asarray([detection.bbox for detection in detections], dtype=float),
        confidence=np.asarray([detection.score for detection in detections], dtype=float),
        class_id=np.asarray([detection.class_id for detection in detections], dtype=int),
        data={"label": np.asarray([detection.label for detection in detections])},
    )


def associate_tracks(
    tracks: dict[int, Track],
    detections: list[Detection],
    iou_threshold: float,
) -> tuple[list[tuple[int, int]], list[int]]:
    matches: list[tuple[int, int]] = []
    unmatched_detections: list[int] = []
    unmatched_tracks = set(tracks.keys())

    for detection_idx, detection in enumerate(detections):
        best_track_id = None
        best_iou = 0.0
        for track_id in list(unmatched_tracks):
            score = iou(detection.bbox, tracks[track_id].bbox)
            if score > best_iou:
                best_iou = score
                best_track_id = track_id
        if best_track_id is not None and best_iou >= iou_threshold:
            matches.append((best_track_id, detection_idx))
            unmatched_tracks.remove(best_track_id)
        else:
            unmatched_detections.append(detection_idx)

    return matches, unmatched_detections


def iou(box_a: BBox, box_b: BBox) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union
