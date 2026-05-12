"""Run a geometry-only calibration A/B diagnostic for VehicleTrack.

This script reuses an existing trajectories.csv file as the detection/tracking
source of truth, then reprojects each bbox bottom-midpoint through one or more
calibration files. It recomputes speed labels with the same display-speed logic
used by the tracker and writes a compact A/B report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from homography import HomographyCalibration, estimate_speed_kmh


MAX_REASONABLE_SPEED_KMH = 180.0
CROSS_SEGMENT_WINDOW = 6
MAX_SPEED_DELTA_KMH = 50.0
SPEED_EMA_ALPHA = 0.4

DEFAULT_TRAJECTORIES = PROJECT_ROOT / "outputs" / "run_bytetrack_smooth_1min" / "trajectories.csv"
DEFAULT_CONTROL = PROJECT_ROOT / "calibration" / "video.multisegment.calibration.json"
DEFAULT_EXPERIMENTAL = PROJECT_ROOT / "calibration" / "video.far-split.experimental.calibration.json"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "calibration_ab"

DIAGNOSTIC_RANGES = {
    "near": {"low": 1.0, "high": 26.0, "distance_m": 25.0},
    "mid": {"low": 28.0, "high": 80.0, "distance_m": 52.0},
    "far": {"low": 90.0, "high": 300.0, "distance_m": 210.0},
}


def main() -> int:
    args = parse_args()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    source_rows = read_trajectories(Path(args.trajectories).expanduser().resolve())
    candidates = [
        ("control", Path(args.control).expanduser().resolve()),
        ("experimental", Path(args.experimental).expanduser().resolve()),
    ]

    all_summary_rows = []
    all_segment_rows = []
    all_diagnostic_rows = []
    recalculated_by_name = {}

    for name, calibration_path in candidates:
        calibration = HomographyCalibration.load(calibration_path)
        recalculated = recalculate_rows(source_rows, calibration)
        recalculated_by_name[name] = recalculated
        metrics = build_metrics(recalculated)
        segment_diagnostics = build_segment_diagnostics(recalculated)

        write_csv(output_dir / f"{name}_reprojected_trajectories.csv", recalculated)
        (output_dir / f"{name}_metrics.json").write_text(
            json.dumps(metrics, indent=2),
            encoding="utf-8",
        )

        all_summary_rows.append(
            {
                "candidate": name,
                "calibration": str(calibration_path),
                "rows": len(recalculated),
                "roi_rows": metrics["roi_rows"],
                "speed_rows": metrics["speed_rows"],
                "mean_speed_kmh": round(metrics["mean_speed_kmh"], 2),
                "median_speed_kmh": round(metrics["median_speed_kmh"], 2),
                "filtered_outlier_rows": metrics["speed_source_rows"].get("filtered_outlier", 0),
                "unavailable_rows": metrics["speed_source_rows"].get("unavailable", 0),
            }
        )
        for segment, count in metrics["segment_rows"].items():
            all_segment_rows.append(
                {
                    "candidate": name,
                    "segment": segment,
                    "rows": count,
                    "filtered_outlier_rows": metrics["filtered_outlier_by_segment"].get(segment, 0),
                    "speed_rows": metrics["speed_rows_by_segment"].get(segment, 0),
                }
            )
        for row in segment_diagnostics:
            all_diagnostic_rows.append({"candidate": name, **row})

    decision = build_acceptance_decision(
        control_metrics=json.loads((output_dir / "control_metrics.json").read_text(encoding="utf-8")),
        experimental_metrics=json.loads((output_dir / "experimental_metrics.json").read_text(encoding="utf-8")),
        diagnostic_rows=all_diagnostic_rows,
    )

    write_csv(output_dir / "summary.csv", all_summary_rows)
    write_csv(output_dir / "segment_rows.csv", all_segment_rows)
    write_csv(output_dir / "segment_diagnostic_ab.csv", all_diagnostic_rows)
    (output_dir / "acceptance_decision.md").write_text(decision, encoding="utf-8")

    print(f"Wrote calibration A/B diagnostics to {output_dir}")
    print(decision)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VehicleTrack calibration A/B without rerunning detection.")
    parser.add_argument("--trajectories", default=str(DEFAULT_TRAJECTORIES))
    parser.add_argument("--control", default=str(DEFAULT_CONTROL))
    parser.add_argument("--experimental", default=str(DEFAULT_EXPERIMENTAL))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser.parse_args()


def read_trajectories(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def recalculate_rows(rows: list[dict], calibration: HomographyCalibration) -> list[dict]:
    track_records: dict[int, list[dict]] = defaultdict(list)
    output_rows = []

    for source in sorted(rows, key=lambda item: (int(item["frame_idx"]), int(item["track_id"]))):
        track_id = int(source["track_id"])
        bbox = (
            float(source["x1"]),
            float(source["y1"]),
            float(source["x2"]),
            float(source["y2"]),
        )
        projection = calibration.project_bbox(bbox)
        record = {
            **source,
            "world_x": projection["world_x"],
            "world_y": projection["world_y"],
            "in_calibration_roi": projection["in_calibration_roi"],
            "calibration_segment": projection["calibration_segment"],
        }
        track_records[track_id].append(record)
        speed, raw_speed, speed_source = estimate_display_speed(track_records[track_id], record)
        record["raw_speed_kmh"] = raw_speed
        record["speed_source"] = speed_source
        record["speed_kmh"] = speed
        output_rows.append(normalize_row(record))

    return output_rows


def estimate_display_speed(records: list[dict], current: dict) -> tuple[float, float, str]:
    if not current["in_calibration_roi"]:
        return 0.0, 0.0, "unavailable"

    calibration_segment = current["calibration_segment"]
    same_segment_records = [
        item
        for item in records
        if item.get("in_calibration_roi") and item.get("calibration_segment") == calibration_segment
    ]
    same_segment_speed = estimate_speed_kmh(same_segment_records)
    if same_segment_speed > 0:
        return filter_and_smooth_speed(records, same_segment_speed, "same_segment")

    cross_segment_records = [
        item
        for item in records
        if item.get("in_calibration_roi")
    ][-CROSS_SEGMENT_WINDOW:]
    cross_segment_speed = estimate_speed_kmh(cross_segment_records, window=CROSS_SEGMENT_WINDOW)
    if cross_segment_speed > 0:
        return filter_and_smooth_speed(records, cross_segment_speed, "cross_segment_smoothed")

    return 0.0, 0.0, "unavailable"


def filter_and_smooth_speed(records: list[dict], raw_speed: float, source: str) -> tuple[float, float, str]:
    if raw_speed > MAX_REASONABLE_SPEED_KMH:
        return 0.0, raw_speed, "filtered_outlier"

    previous_speed = previous_display_speed(records)
    if source == "cross_segment_smoothed" and previous_speed > 0 and abs(raw_speed - previous_speed) > MAX_SPEED_DELTA_KMH:
        return 0.0, raw_speed, "filtered_outlier"

    if source == "cross_segment_smoothed" and previous_speed > 0:
        smoothed = (SPEED_EMA_ALPHA * raw_speed) + ((1.0 - SPEED_EMA_ALPHA) * previous_speed)
        return smoothed, raw_speed, source

    return raw_speed, raw_speed, source


def previous_display_speed(records: list[dict]) -> float:
    for record in reversed(records[:-1]):
        speed = float(record.get("speed_kmh", 0.0) or 0.0)
        if speed > 0:
            return speed
    return 0.0


def normalize_row(record: dict) -> dict:
    output = dict(record)
    output["world_x"] = format_float(output["world_x"])
    output["world_y"] = format_float(output["world_y"])
    output["in_calibration_roi"] = str(bool(output["in_calibration_roi"]))
    output["raw_speed_kmh"] = format_float(output["raw_speed_kmh"])
    output["speed_kmh"] = format_float(output["speed_kmh"])
    return output


def build_metrics(rows: list[dict]) -> dict:
    speeds = [float(row["speed_kmh"]) for row in rows if as_bool(row["in_calibration_roi"]) and float(row["speed_kmh"]) > 0]
    segment_rows = Counter(row["calibration_segment"] or "outside_calibrated_area" for row in rows)
    source_rows = Counter(row["speed_source"] or "unavailable" for row in rows)
    filtered_by_segment = Counter(
        row["calibration_segment"] or "outside_calibrated_area"
        for row in rows
        if row["speed_source"] == "filtered_outlier"
    )
    speed_by_segment = Counter(
        row["calibration_segment"] or "outside_calibrated_area"
        for row in rows
        if float(row["speed_kmh"]) > 0
    )
    return {
        "rows": len(rows),
        "roi_rows": sum(1 for row in rows if as_bool(row["in_calibration_roi"])),
        "speed_rows": len(speeds),
        "mean_speed_kmh": mean(speeds),
        "median_speed_kmh": percentile(speeds, 50),
        "speed_source_rows": dict(sorted(source_rows.items())),
        "segment_rows": dict(sorted(segment_rows.items())),
        "filtered_outlier_by_segment": dict(sorted(filtered_by_segment.items())),
        "speed_rows_by_segment": dict(sorted(speed_by_segment.items())),
    }


def build_segment_diagnostics(rows: list[dict]) -> list[dict]:
    output = []
    for segment_group, params in DIAGNOSTIC_RANGES.items():
        samples = diagnostic_samples(rows, segment_group, params["low"], params["high"])
        rel_errors = [sample["rel_error_pct"] for sample in samples]
        output.append(
            {
                "segment_group": segment_group,
                "sample_count": len(samples),
                "reference_distance_m": params["distance_m"],
                "median_rel_error_pct": round(percentile(rel_errors, 50), 2) if rel_errors else 0.0,
                "mean_rel_error_pct": round(mean(rel_errors), 2) if rel_errors else 0.0,
                "p75_rel_error_pct": round(percentile(rel_errors, 75), 2) if rel_errors else 0.0,
                "max_rel_error_pct": round(max(rel_errors), 2) if rel_errors else 0.0,
            }
        )
    return output


def diagnostic_samples(rows: list[dict], segment_group: str, low_y: float, high_y: float) -> list[dict]:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if as_bool(row["in_calibration_roi"]) and segment_matches(row["calibration_segment"], segment_group):
            by_track[int(row["track_id"])].append(row)

    samples = []
    for track_id, track_rows in by_track.items():
        ordered = sorted(track_rows, key=lambda item: int(item["frame_idx"]))
        high_cross = crossing_frame(ordered, high_y)
        low_cross = crossing_frame(ordered, low_y)
        if high_cross is None or low_cross is None or low_cross <= high_cross:
            continue
        elapsed = (low_cross - high_cross) / 25.0
        if elapsed <= 0:
            continue
        system_speeds = [
            float(row["speed_kmh"])
            for row in ordered
            if high_cross <= float(row["frame_idx"]) <= low_cross and float(row["speed_kmh"]) > 0
        ]
        if not system_speeds:
            continue
        reference_speed = ((high_y - low_y) / elapsed) * 3.6
        system_speed = percentile(system_speeds, 50)
        if reference_speed <= 0:
            continue
        samples.append(
            {
                "track_id": track_id,
                "reference_kmh": reference_speed,
                "system_kmh": system_speed,
                "rel_error_pct": abs(system_speed - reference_speed) / reference_speed * 100.0,
            }
        )
    return samples


def crossing_frame(rows: list[dict], target_y: float) -> float | None:
    for previous, current in zip(rows, rows[1:], strict=False):
        y0 = float(previous["world_y"])
        y1 = float(current["world_y"])
        if (y0 - target_y) == 0:
            return float(previous["frame_idx"])
        if (y0 - target_y) * (y1 - target_y) <= 0 and y0 != y1:
            fraction = (target_y - y0) / (y1 - y0)
            return float(previous["frame_idx"]) + fraction * (float(current["frame_idx"]) - float(previous["frame_idx"]))
    return None


def segment_matches(segment: str, group: str) -> bool:
    if group == "far":
        return segment.startswith("far_")
    return segment.startswith(f"{group}_")


def build_acceptance_decision(control_metrics: dict, experimental_metrics: dict, diagnostic_rows: list[dict]) -> str:
    control_far = find_diagnostic(diagnostic_rows, "control", "far")
    experimental_far = find_diagnostic(diagnostic_rows, "experimental", "far")
    control_near = find_diagnostic(diagnostic_rows, "control", "near")
    experimental_near = find_diagnostic(diagnostic_rows, "experimental", "near")
    control_mid = find_diagnostic(diagnostic_rows, "control", "mid")
    experimental_mid = find_diagnostic(diagnostic_rows, "experimental", "mid")

    control_far_filtered = far_filtered_rows(control_metrics)
    experimental_far_filtered = far_filtered_rows(experimental_metrics)
    far_error_improved = experimental_far["median_rel_error_pct"] <= 38.0
    far_filtered_improved = experimental_far_filtered <= 1163
    near_ok = experimental_near["median_rel_error_pct"] <= max(15.0, control_near["median_rel_error_pct"] + 5.0)
    mid_ok = experimental_mid["median_rel_error_pct"] <= max(25.0, control_mid["median_rel_error_pct"] + 8.0)
    speed_rows_ok = experimental_metrics["speed_rows"] >= control_metrics["speed_rows"] * 0.85
    accepted = (far_error_improved or far_filtered_improved) and near_ok and mid_ok and speed_rows_ok

    status = "PASS" if accepted else "DO NOT ADOPT"
    return "\n".join(
        [
            "# Far Calibration A/B Acceptance Decision",
            "",
            f"Decision: **{status}**",
            "",
            "## Criteria",
            "",
            f"- Far diagnostic median relative error: control {control_far['median_rel_error_pct']}%, experimental {experimental_far['median_rel_error_pct']}% (target <= 38%).",
            f"- Far filtered outlier rows: control {control_far_filtered}, experimental {experimental_far_filtered} (target <= 1163).",
            f"- Near median relative error: control {control_near['median_rel_error_pct']}%, experimental {experimental_near['median_rel_error_pct']}%.",
            f"- Mid median relative error: control {control_mid['median_rel_error_pct']}%, experimental {experimental_mid['median_rel_error_pct']}%.",
            f"- Speed rows: control {control_metrics['speed_rows']}, experimental {experimental_metrics['speed_rows']}.",
            "",
            "## Interpretation",
            "",
            (
                "Experimental far split passes the pre-set A/B thresholds. It is reasonable to run a full 1-minute ByteTrack pipeline with this calibration before adopting it."
                if accepted
                else "Experimental far split does not clear the pre-set adoption thresholds. Keep the current calibration as the documented best tradeoff unless a second experimental point set is tested."
            ),
            "",
        ]
    )


def find_diagnostic(rows: list[dict], candidate: str, segment_group: str) -> dict:
    for row in rows:
        if row["candidate"] == candidate and row["segment_group"] == segment_group:
            return row
    raise KeyError(f"Missing diagnostic row for {candidate}/{segment_group}")


def far_filtered_rows(metrics: dict) -> int:
    return sum(
        count
        for segment, count in metrics["filtered_outlier_by_segment"].items()
        if segment.startswith("far_")
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


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


def format_float(value: object) -> str:
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
