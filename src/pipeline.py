"""End-to-end baseline pipeline for VehicleTrack smoke runs."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import time

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))

from detection import FasterRCNNVehicleDetector
from homography import HomographyCalibration
from tracking import ByteTrackTracker, IoUTracker
from visualize import draw_frame_status, draw_tracks


def main() -> int:
    args = parse_args()
    run_pipeline(args)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run VehicleTrack baseline pipeline.")
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument("--calib", required=True, help="Calibration JSON path.")
    parser.add_argument("--out", required=True, help="Output run folder.")
    parser.add_argument("--max-frames", type=int, default=50, help="Max frames to process.")
    parser.add_argument("--start-frame", type=int, help="Frame index to start processing from.")
    parser.add_argument("--start-sec", type=float, help="Timestamp in seconds to start processing from.")
    parser.add_argument("--score-threshold", type=float, default=0.7)
    parser.add_argument(
        "--tracker",
        choices=("iou", "bytetrack"),
        default="iou",
        help="Tracking backend. Use bytetrack for the stronger tracker run.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    parser.add_argument("--max-missed", type=int, default=20)
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cpu.")
    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> None:
    video_path = Path(args.video).expanduser().resolve()
    calibration_path = Path(args.calib).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    calibration = HomographyCalibration.load(calibration_path)
    detector = FasterRCNNVehicleDetector(
        score_threshold=args.score_threshold,
        device=args.device,
    )
    tracker = create_tracker(args, calibration)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    start_frame = resolve_start_frame(args.start_frame, args.start_sec, fps)
    if start_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    writer = cv2.VideoWriter(
        str(output_dir / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open output video writer: {output_dir / 'annotated.mp4'}")

    config = {
        "video": str(video_path),
        "calibration": str(calibration_path),
        "detector": "fasterrcnn_resnet50_fpn_coco",
        "tracker": args.tracker,
        "max_frames": args.max_frames,
        "score_threshold": args.score_threshold,
        "iou_threshold": args.iou_threshold,
        "max_missed": args.max_missed,
        "fps": fps,
        "width": width,
        "height": height,
        "start_frame": start_frame,
        "start_sec": start_frame / fps if fps else 0.0,
        "calibration_type": calibration.metadata.get("calibration_type", "single_roi"),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    start = time.perf_counter()
    frame_idx = start_frame
    processed_frames = 0
    per_frame_counts: list[int] = []

    while processed_frames < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        timestamp = frame_idx / fps if fps else 0.0
        detections = detector.detect(frame)
        tracks = tracker.update(detections, frame_idx=frame_idx, timestamp=timestamp)
        per_frame_counts.append(len(detections))

        annotated = draw_tracks(frame, tracks)
        annotated = draw_frame_status(annotated, frame_idx, fps, active_tracks=len(tracks))
        writer.write(annotated)

        processed_frames += 1
        frame_idx += 1
        print(
            f"frame={frame_idx} detections={len(detections)} active_tracks={len(tracks)}",
            flush=True,
        )

    cap.release()
    writer.release()
    elapsed_sec = time.perf_counter() - start

    records = sorted(
        tracker.all_records(),
        key=lambda record: (record["frame_idx"], record["track_id"]),
    )
    write_trajectories(output_dir / "trajectories.csv", records)
    metrics = build_metrics(
        records=records,
        config=config,
        calibration=calibration,
        processed_frames=processed_frames,
        elapsed_sec=elapsed_sec,
        per_frame_counts=per_frame_counts,
        total_tracks=tracker.total_tracks_created,
    )
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Done. Saved run artifacts to {output_dir}")


def create_tracker(args: argparse.Namespace, calibration: HomographyCalibration):
    if args.tracker == "bytetrack":
        return ByteTrackTracker(
            calibration=calibration,
            max_missed=args.max_missed,
        )
    return IoUTracker(
        calibration=calibration,
        iou_threshold=args.iou_threshold,
        max_missed=args.max_missed,
    )


def write_trajectories(path: Path, records: list[dict]) -> None:
    fieldnames = [
        "frame_idx",
        "timestamp",
        "track_id",
        "label",
        "score",
        "x1",
        "y1",
        "x2",
        "y2",
        "world_x",
        "world_y",
        "in_calibration_roi",
        "calibration_segment",
        "raw_speed_kmh",
        "speed_source",
        "speed_kmh",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})


def build_metrics(
    records: list[dict],
    config: dict,
    calibration: HomographyCalibration,
    processed_frames: int,
    elapsed_sec: float,
    per_frame_counts: list[int],
    total_tracks: int,
) -> dict:
    speeds = [
        float(record["speed_kmh"])
        for record in records
        if record.get("in_calibration_roi") and float(record.get("speed_kmh", 0.0)) > 0
    ]
    roi_rows = [record for record in records if record.get("in_calibration_roi")]
    segment_counts = Counter(
        str(record.get("calibration_segment") or "outside_calibrated_area")
        for record in records
    )
    speed_source_counts = Counter(
        str(record.get("speed_source") or "unavailable")
        for record in records
    )
    label_counts = Counter(str(record.get("label", "unknown")) for record in records)
    unique_tracks = sorted({int(record["track_id"]) for record in records})
    roi_tracks = sorted({int(record["track_id"]) for record in roi_rows})
    frame_indices = [int(record["frame_idx"]) for record in records]
    timestamps = [float(record["timestamp"]) for record in records]
    processing_fps = processed_frames / elapsed_sec if elapsed_sec > 0 else 0.0
    mean_detections = sum(per_frame_counts) / len(per_frame_counts) if per_frame_counts else 0.0

    return {
        "schema_version": "1.0",
        "run": {
            "video": config["video"],
            "detector": config["detector"],
            "tracker": config["tracker"],
            "start_frame": config["start_frame"],
            "end_frame": max(frame_indices) if frame_indices else config["start_frame"],
            "start_sec": min(timestamps) if timestamps else config["start_sec"],
            "end_sec": max(timestamps) if timestamps else config["start_sec"],
            "processed_frames": processed_frames,
            "processed_duration_sec": processed_frames / config["fps"] if config["fps"] else 0.0,
        },
        "processing": {
            "elapsed_sec": elapsed_sec,
            "processing_fps": processing_fps,
            "video_fps": config["fps"],
            "realtime_ratio": processing_fps / config["fps"] if config["fps"] else 0.0,
            "offline_only": True,
        },
        "detections": {
            "mean_per_frame": mean_detections,
            "min_per_frame": min(per_frame_counts) if per_frame_counts else 0,
            "max_per_frame": max(per_frame_counts) if per_frame_counts else 0,
            "total_detection_rows": len(records),
            "label_counts_rows": dict(sorted(label_counts.items())),
        },
        "tracking": {
            "tracker": config["tracker"],
            "total_tracks_created": total_tracks,
            "unique_tracks_in_csv": len(unique_tracks),
            "roi_tracks": len(roi_tracks),
            "trajectory_rows": len(records),
            "manual_id_switch_review_required": True,
        },
        "speed": {
            "roi_only": True,
            "calibration_type": config.get("calibration_type", "single_roi"),
            "point": "bbox_bottom_midpoint",
            "unit": "km/h",
            "count": len(speeds),
            "roi_rows": len(roi_rows),
            "segment_rows": dict(sorted(segment_counts.items())),
            "source_rows": dict(sorted(speed_source_counts.items())),
            "max_reasonable_speed_kmh": 180.0,
            "mean": mean(speeds),
            "median": percentile(speeds, 50),
            "min": min(speeds) if speeds else 0.0,
            "p25": percentile(speeds, 25),
            "p75": percentile(speeds, 75),
            "max": max(speeds) if speeds else 0.0,
            "histogram": histogram(speeds, bins=[0, 50, 70, 90, 110, 130, 160]),
            "calibration_uncertainty": "local Homography ROI; absolute speed depends on lane-width/depth assumptions",
        },
        "calibration": calibration_metrics(calibration, config["calibration"]),
        "artifacts": {
            "annotated_video": "annotated.mp4",
            "trajectories_csv": "trajectories.csv",
            "config_json": "config.json",
            "metrics_json": "metrics.json",
        },
        "legacy_summary": {
            "processed_frames": processed_frames,
            "elapsed_sec": elapsed_sec,
            "processing_fps": processing_fps,
            "total_tracks": total_tracks,
            "trajectory_rows": len(records),
            "calibration_roi_rows": len(roi_rows),
            "calibration_segment_rows": dict(sorted(segment_counts.items())),
            "speed_source_rows": dict(sorted(speed_source_counts.items())),
            "mean_detections_per_frame": mean_detections,
            "mean_speed_kmh": mean(speeds),
            "max_speed_kmh": max(speeds) if speeds else 0.0,
        },
    }


def calibration_metrics(calibration: HomographyCalibration, calibration_path: str) -> dict:
    base = {
        "path": calibration_path,
        "video_name": calibration.metadata.get("video_name"),
        "frame_idx": calibration.metadata.get("calibration_frame_idx"),
        "timestamp": calibration.metadata.get("calibration_timestamp"),
        "point_order": calibration.metadata.get("point_order"),
        "calibration_type": calibration.metadata.get("calibration_type", "single_roi"),
    }
    if hasattr(calibration, "segments"):
        base["segments"] = [
            {
                "name": segment.metadata.get("segment_name"),
                "width_m": segment.metadata.get("width_m"),
                "depth_m": segment.metadata.get("depth_m"),
                "source_points": segment.source_points.tolist(),
                "destination_points": segment.destination_points.tolist(),
            }
            for segment in calibration.segments
        ]
        return base

    return {
        **base,
        "width_m": calibration.metadata.get("width_m"),
        "depth_m": calibration.metadata.get("depth_m"),
        "source_points": calibration.source_points.tolist(),
        "destination_points": calibration.destination_points.tolist(),
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    fraction = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * fraction


def histogram(values: list[float], bins: list[float]) -> list[dict]:
    if len(bins) < 2:
        raise ValueError("histogram requires at least two bin edges")
    buckets = []
    for idx, lower in enumerate(bins[:-1]):
        upper = bins[idx + 1]
        count = sum(1 for value in values if lower <= value < upper)
        buckets.append({"range": f"{lower:g}-{upper:g}", "min": lower, "max": upper, "count": count})
    overflow_min = bins[-1]
    overflow_count = sum(1 for value in values if value >= overflow_min)
    buckets.append({"range": f"{overflow_min:g}+", "min": overflow_min, "max": None, "count": overflow_count})
    return buckets


def resolve_start_frame(start_frame: int | None, start_sec: float | None, fps: float) -> int:
    if start_frame is not None and start_sec is not None:
        raise ValueError("Use either --start-frame or --start-sec, not both")
    if start_frame is not None:
        if start_frame < 0:
            raise ValueError("--start-frame must be non-negative")
        return start_frame
    if start_sec is not None:
        if start_sec < 0:
            raise ValueError("--start-sec must be non-negative")
        return int(round(start_sec * fps))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
