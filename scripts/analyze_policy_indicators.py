"""Generate policy-oriented analysis artifacts for VehicleTrack.

The input is the best current trajectory run. The output tables are intended
for the final report's data-analysis and urban-policy sections, not for model
training or enforcement-grade measurement.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from pathlib import Path
import statistics


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAJECTORIES = PROJECT_ROOT / "outputs" / "run_bytetrack_smooth_1min" / "trajectories.csv"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "policy_analysis"
KPH_THRESHOLD_65_MPH = 105.0
DEFAULT_COUNTING_LINE_Y_M = 27.0
RELIABLE_SPEED_SOURCES = {"same_segment", "cross_segment_smoothed"}
RELIABLE_POLICY_SEGMENTS = {"near_two_lane", "mid_two_lane"}


def main() -> int:
    args = parse_args()
    trajectories_path = Path(args.trajectories).expanduser().resolve()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(trajectories_path)
    flow_events = build_flow_events(rows, args.counting_line_y_m)
    overspeed_events = build_overspeed_events(rows, args.speed_threshold_kmh)
    policy_rows = build_policy_indicators(
        rows,
        args.speed_threshold_kmh,
        args.counting_line_y_m,
        flow_events,
        overspeed_events,
    )
    write_csv(output_dir / "policy_indicators.csv", policy_rows)
    write_csv(output_dir / "speed_distribution.csv", build_speed_distribution(rows, args.speed_threshold_kmh))
    write_csv(output_dir / "class_mix.csv", build_class_mix(rows))
    write_csv(output_dir / "density_time_profile.csv", build_density_time_profile(rows))
    write_csv(output_dir / "spatial_speed_bins.csv", build_spatial_speed_bins(rows))
    write_csv(output_dir / "speed_source_by_segment.csv", build_speed_source_by_segment(rows))
    write_csv(output_dir / "track_speed_summary.csv", build_track_speed_summary(rows))
    write_csv(output_dir / "flow_events.csv", flow_events)
    write_csv(output_dir / "flow_summary.csv", build_flow_summary(flow_events, rows, args.counting_line_y_m))
    write_csv(output_dir / "overspeed_events.csv", overspeed_events)
    write_csv(output_dir / "overspeed_summary.csv", build_overspeed_summary(overspeed_events, rows, args.speed_threshold_kmh))

    print(f"Wrote policy analysis artifacts to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate VehicleTrack policy analysis tables.")
    parser.add_argument("--trajectories", default=str(DEFAULT_TRAJECTORIES))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--speed-threshold-kmh", type=float, default=KPH_THRESHOLD_65_MPH)
    parser.add_argument("--counting-line-y-m", type=float, default=DEFAULT_COUNTING_LINE_Y_M)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["frame_idx"] = int(row["frame_idx"])
        row["timestamp"] = float(row["timestamp"])
        row["track_id"] = int(row["track_id"])
        row["score"] = to_float(row["score"])
        row["world_x"] = to_float(row["world_x"])
        row["world_y"] = to_float(row["world_y"])
        row["raw_speed_kmh"] = to_float(row["raw_speed_kmh"])
        row["speed_kmh"] = to_float(row["speed_kmh"])
        row["in_calibration_roi"] = row["in_calibration_roi"].lower() == "true"
    return rows


def build_policy_indicators(
    rows: list[dict],
    threshold_kmh: float,
    counting_line_y_m: float,
    flow_events: list[dict],
    overspeed_events: list[dict],
) -> list[dict]:
    duration_sec = max(row["timestamp"] for row in rows) - min(row["timestamp"] for row in rows)
    duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0
    unique_tracks = {row["track_id"] for row in rows}
    reliable_rows = reliable_policy_speed_rows(rows)
    reliable_speeds = [row["speed_kmh"] for row in reliable_rows]
    all_displayed_speeds = [row["speed_kmh"] for row in rows if row["speed_kmh"] > 0]
    track_summary = build_track_speed_summary(rows)
    reliable_track_speeds = [
        float(row["median_reliable_near_mid_speed_kmh"])
        for row in track_summary
        if row["median_reliable_near_mid_speed_kmh"]
    ]
    high_speed_tracks = [speed for speed in reliable_track_speeds if speed > threshold_kmh]
    source_counts = Counter(row["speed_source"] or "unavailable" for row in rows)
    segment_counts = Counter(row["calibration_segment"] or "outside_calibrated_area" for row in rows)
    duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0

    return [
        indicator("trajectory_records", len(rows), "rows", "Derived tabular dataset size from the 60-second video."),
        indicator("unique_track_ids", len(unique_tracks), "tracks", "ByteTrack vehicle candidates in the derived dataset."),
        indicator("mean_active_tracks_per_second", mean_active_tracks_per_second(rows), "tracks/sec", "Presence / occupancy proxy in the camera view; not a flow-rate crossing count."),
        indicator("counting_line_y_m", counting_line_y_m, "m", "Virtual counting line used for flow events."),
        indicator("counting_line_crossings", len(flow_events), "vehicles", "Unique track crossings of the virtual counting line."),
        indicator("counting_line_flow_per_min", safe_divide(len(flow_events), duration_min), "vehicles/min", "Pilot-window flow rate at the counting line."),
        indicator("overspeed_event_tracks", len(overspeed_events), "tracks", "Unique tracks with reliable near/mid speed above the screening threshold."),
        indicator("reliable_policy_speed_rows", len(reliable_rows), "rows", "Rows with displayed speed from near/mid segments only."),
        indicator("reliable_policy_speed_share", safe_pct(len(reliable_rows), len(rows)), "% of rows", "Conservative share usable for policy-facing speed analysis."),
        indicator("excluded_from_policy_speed_share", 100.0 - safe_pct(len(reliable_rows), len(rows)), "% of rows", "Rows excluded from policy-grade speed claims because of confidence zoning, missing trajectory history, or implausible projection."),
        indicator("median_reliable_policy_speed_kmh", median(reliable_speeds), "km/h", "Median near/mid displayed speed; far speeds are excluded from this policy indicator."),
        indicator("p25_reliable_policy_speed_kmh", percentile(reliable_speeds, 25), "km/h", "Lower quartile of reliable near/mid displayed speed."),
        indicator("p75_reliable_policy_speed_kmh", percentile(reliable_speeds, 75), "km/h", "Upper quartile of reliable near/mid displayed speed."),
        indicator("row_share_above_105_kmh_near_mid", safe_pct(sum(1 for row in reliable_rows if row["speed_kmh"] > threshold_kmh), len(reliable_rows)), "% of reliable near/mid speed rows", "Hypothetical 65 mph screening threshold; not a posted-speed claim."),
        indicator("track_share_above_105_kmh_near_mid", safe_pct(len(high_speed_tracks), len(reliable_track_speeds)), "% of tracks with near/mid median speed", "Track-level high-speed screening under the same hypothetical threshold."),
        indicator("all_displayed_median_speed_kmh", median(all_displayed_speeds), "km/h", "All displayed speeds, including far segment; useful for context but less policy-safe."),
        indicator("filtered_outlier_rows", source_counts.get("filtered_outlier", 0), "rows", "Rows where projected speed exceeded the plausibility cap and was hidden as -- km/h."),
        indicator("unavailable_speed_rows", source_counts.get("unavailable", 0), "rows", "Rows without enough calibrated trajectory history for speed display."),
        indicator("far_segment_row_share", safe_pct(segment_counts.get("far_two_lane", 0), len(rows)), "% of rows", "Coverage share in the least reliable speed zone."),
        indicator("near_mid_segment_row_share", safe_pct(segment_counts.get("near_two_lane", 0) + segment_counts.get("mid_two_lane", 0), len(rows)), "% of rows", "Coverage share in the more policy-interpretable zones."),
    ]


def indicator(name: str, value: float | int | str, unit: str, interpretation: str) -> dict:
    return {
        "indicator": name,
        "value": format_value(value),
        "unit": unit,
        "interpretation": interpretation,
    }


def build_speed_distribution(rows: list[dict], threshold_kmh: float) -> list[dict]:
    reliable_rows = reliable_policy_speed_rows(rows)
    bins = [
        ("0-50", 0.0, 50.0),
        ("50-70", 50.0, 70.0),
        ("70-90", 70.0, 90.0),
        ("90-105", 90.0, threshold_kmh),
        ("105-120", threshold_kmh, 120.0),
        ("120-140", 120.0, 140.0),
        ("140+", 140.0, None),
    ]
    output = []
    for label, lower, upper in bins:
        count = sum(1 for row in reliable_rows if lower <= row["speed_kmh"] and (upper is None or row["speed_kmh"] < upper))
        output.append(
            {
                "speed_bin_kmh": label,
                "rows": count,
                "share_of_reliable_near_mid_speed_rows_pct": format_value(safe_pct(count, len(reliable_rows))),
            }
        )
    return output


def build_class_mix(rows: list[dict]) -> list[dict]:
    row_counts = Counter(row["label"] for row in rows)
    track_labels = dominant_track_labels(rows)
    track_counts = Counter(track_labels.values())
    labels = sorted(set(row_counts) | set(track_counts))
    return [
        {
            "label": label,
            "detection_rows": row_counts.get(label, 0),
            "detection_row_share_pct": format_value(safe_pct(row_counts.get(label, 0), len(rows))),
            "dominant_track_count": track_counts.get(label, 0),
            "dominant_track_share_pct": format_value(safe_pct(track_counts.get(label, 0), len(track_labels))),
        }
        for label in labels
    ]


def build_density_time_profile(rows: list[dict]) -> list[dict]:
    by_second: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_second[int(row["timestamp"])].append(row)

    output = []
    for second in range(0, int(max(row["timestamp"] for row in rows)) + 1):
        second_rows = by_second.get(second, [])
        active_tracks = {row["track_id"] for row in second_rows}
        reliable_speeds = [row["speed_kmh"] for row in second_rows if is_reliable_policy_speed_row(row)]
        output.append(
            {
                "second": second,
                "detection_rows": len(second_rows),
                "active_tracks": len(active_tracks),
                "reliable_near_mid_speed_rows": len(reliable_speeds),
                "median_reliable_near_mid_speed_kmh": format_value(median(reliable_speeds)),
            }
        )
    return output


def mean_active_tracks_per_second(rows: list[dict]) -> float:
    by_second: dict[int, set[int]] = defaultdict(set)
    for row in rows:
        by_second[int(row["timestamp"])].add(row["track_id"])
    if not by_second:
        return 0.0
    return sum(len(track_ids) for track_ids in by_second.values()) / len(by_second)


def build_spatial_speed_bins(rows: list[dict]) -> list[dict]:
    bins = [
        ("near_0_27m", 0.0, 27.0),
        ("mid_27_45m", 27.0, 45.0),
        ("mid_45_63m", 45.0, 63.0),
        ("mid_63_81m", 63.0, 81.0),
        ("far_81_150m", 81.0, 150.0),
        ("far_150_220m", 150.0, 220.0),
        ("far_220_300m", 220.0, 300.0),
        ("far_300_369m", 300.0, 369.0),
    ]
    output = []
    for label, low, high in bins:
        bin_rows = [
            row
            for row in rows
            if row["in_calibration_roi"] and low <= row["world_y"] < high
        ]
        reliable_speeds = [
            row["speed_kmh"]
            for row in bin_rows
            if row["speed_kmh"] > 0 and row["speed_source"] in RELIABLE_SPEED_SOURCES
        ]
        source_counts = Counter(row["speed_source"] or "unavailable" for row in bin_rows)
        output.append(
            {
                "world_y_bin": label,
                "rows": len(bin_rows),
                "speed_rows": len(reliable_speeds),
                "median_speed_kmh": format_value(median(reliable_speeds)),
                "p25_speed_kmh": format_value(percentile(reliable_speeds, 25)),
                "p75_speed_kmh": format_value(percentile(reliable_speeds, 75)),
                "filtered_outlier_rows": source_counts.get("filtered_outlier", 0),
                "unavailable_rows": source_counts.get("unavailable", 0),
                "confidence_note": "higher" if high <= 81.0 else "lower_far_field",
            }
        )
    return output


def build_speed_source_by_segment(rows: list[dict]) -> list[dict]:
    segment_counts = Counter(row["calibration_segment"] or "outside_calibrated_area" for row in rows)
    grouped: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        segment = row["calibration_segment"] or "outside_calibrated_area"
        grouped[segment][row["speed_source"] or "unavailable"] += 1

    output = []
    for segment in sorted(segment_counts):
        total = segment_counts[segment]
        for source in ["same_segment", "cross_segment_smoothed", "filtered_outlier", "unavailable"]:
            count = grouped[segment].get(source, 0)
            output.append(
                {
                    "segment": segment,
                    "speed_source": source,
                    "rows": count,
                    "share_of_segment_rows_pct": format_value(safe_pct(count, total)),
                }
            )
    return output


def build_track_speed_summary(rows: list[dict]) -> list[dict]:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[row["track_id"]].append(row)

    output = []
    for track_id in sorted(by_track):
        track_rows = by_track[track_id]
        labels = Counter(row["label"] for row in track_rows)
        reliable_rows = [row for row in track_rows if is_reliable_policy_speed_row(row)]
        reliable_speeds = [row["speed_kmh"] for row in reliable_rows]
        output.append(
            {
                "track_id": track_id,
                "dominant_label": labels.most_common(1)[0][0],
                "rows": len(track_rows),
                "duration_sec": format_value(max(row["timestamp"] for row in track_rows) - min(row["timestamp"] for row in track_rows)),
                "reliable_near_mid_speed_rows": len(reliable_rows),
                "median_reliable_near_mid_speed_kmh": format_value(median(reliable_speeds)),
                "max_reliable_near_mid_speed_kmh": format_value(max(reliable_speeds) if reliable_speeds else ""),
            }
        )
    return output


def build_flow_events(rows: list[dict], counting_line_y_m: float) -> list[dict]:
    by_track = rows_by_track(rows)
    events = []
    for track_id, track_rows in sorted(by_track.items()):
        ordered = [
            row
            for row in sorted(track_rows, key=lambda item: item["frame_idx"])
            if row["in_calibration_roi"]
            and row["world_y"] == row["world_y"]
            and row["calibration_segment"] in RELIABLE_POLICY_SEGMENTS
        ]
        for previous, current in zip(ordered, ordered[1:], strict=False):
            previous_delta = previous["world_y"] - counting_line_y_m
            current_delta = current["world_y"] - counting_line_y_m
            if previous_delta == 0:
                event = crossing_event(track_id, previous, current, counting_line_y_m, 0.0)
                events.append(event)
                break
            if previous_delta * current_delta <= 0 and previous["world_y"] != current["world_y"]:
                fraction = (counting_line_y_m - previous["world_y"]) / (current["world_y"] - previous["world_y"])
                event = crossing_event(track_id, previous, current, counting_line_y_m, fraction)
                events.append(event)
                break
    return events


def crossing_event(
    track_id: int,
    previous: dict,
    current: dict,
    counting_line_y_m: float,
    fraction: float,
) -> dict:
    crossing_frame = previous["frame_idx"] + fraction * (current["frame_idx"] - previous["frame_idx"])
    crossing_time = previous["timestamp"] + fraction * (current["timestamp"] - previous["timestamp"])
    direction = (
        "toward_camera"
        if current["world_y"] < previous["world_y"]
        else "away_from_camera"
    )
    return {
        "track_id": track_id,
        "label": dominant_label([previous, current]),
        "counting_line_y_m": format_value(counting_line_y_m),
        "crossing_frame": format_value(crossing_frame),
        "crossing_time_sec": format_value(crossing_time),
        "direction": direction,
        "from_segment": previous["calibration_segment"],
        "to_segment": current["calibration_segment"],
        "previous_world_y": format_value(previous["world_y"]),
        "current_world_y": format_value(current["world_y"]),
    }


def build_flow_summary(flow_events: list[dict], rows: list[dict], counting_line_y_m: float) -> list[dict]:
    duration_sec = max(row["timestamp"] for row in rows) - min(row["timestamp"] for row in rows)
    duration_min = duration_sec / 60.0 if duration_sec > 0 else 0.0
    label_counts = Counter(row["label"] for row in flow_events)
    direction_counts = Counter(row["direction"] for row in flow_events)
    summary = [
        {
            "metric": "counting_line_y_m",
            "value": format_value(counting_line_y_m),
            "unit": "m",
            "interpretation": "Virtual line location in projected road coordinates.",
        },
        {
            "metric": "total_crossings",
            "value": len(flow_events),
            "unit": "vehicles",
            "interpretation": "Unique track crossings of the counting line.",
        },
        {
            "metric": "flow_per_min",
            "value": format_value(safe_divide(len(flow_events), duration_min)),
            "unit": "vehicles/min",
            "interpretation": "Pilot-window flow rate from the 60-second clip.",
        },
    ]
    for label in sorted(label_counts):
        summary.append(
            {
                "metric": f"{label}_crossings",
                "value": label_counts[label],
                "unit": "vehicles",
                "interpretation": f"Counting-line crossings for dominant class {label}.",
            }
        )
    for direction in sorted(direction_counts):
        summary.append(
            {
                "metric": f"{direction}_crossings",
                "value": direction_counts[direction],
                "unit": "vehicles",
                "interpretation": f"Counting-line crossings moving {direction}.",
            }
        )
    return summary


def build_overspeed_events(rows: list[dict], threshold_kmh: float) -> list[dict]:
    by_track = rows_by_track(rows)
    events = []
    for track_id, track_rows in sorted(by_track.items()):
        reliable_rows = [
            row
            for row in sorted(track_rows, key=lambda item: item["frame_idx"])
            if is_reliable_policy_speed_row(row)
        ]
        reliable_speeds = [row["speed_kmh"] for row in reliable_rows]
        track_median_speed = median(reliable_speeds)
        if not track_median_speed or float(track_median_speed) <= threshold_kmh:
            continue
        above_rows = [row for row in reliable_rows if row["speed_kmh"] > threshold_kmh]
        speeds = [row["speed_kmh"] for row in above_rows]
        first = above_rows[0]
        last = above_rows[-1]
        events.append(
            {
                "track_id": track_id,
                "label": dominant_label(reliable_rows),
                "threshold_kmh": format_value(threshold_kmh),
                "event_start_sec": format_value(first["timestamp"]),
                "event_end_sec": format_value(last["timestamp"]),
                "event_duration_sec": format_value(last["timestamp"] - first["timestamp"]),
                "rows_above_threshold": len(above_rows),
                "reliable_near_mid_speed_rows": len(reliable_rows),
                "track_median_reliable_speed_kmh": format_value(track_median_speed),
                "median_above_threshold_kmh": format_value(median(speeds)),
                "max_reliable_speed_kmh": format_value(max(reliable_speeds)),
                "dominant_segment": dominant_segment(above_rows),
            }
        )
    return events


def build_overspeed_summary(overspeed_events: list[dict], rows: list[dict], threshold_kmh: float) -> list[dict]:
    reliable_tracks = {
        row["track_id"]
        for row in rows
        if is_reliable_policy_speed_row(row)
    }
    label_counts = Counter(row["label"] for row in overspeed_events)
    summary = [
        {
            "metric": "threshold_kmh",
            "value": format_value(threshold_kmh),
            "unit": "km/h",
            "interpretation": "Hypothetical screening threshold, not a posted-limit claim.",
        },
        {
            "metric": "overspeed_event_tracks",
            "value": len(overspeed_events),
            "unit": "tracks",
            "interpretation": "Unique tracks with reliable near/mid speed above the threshold.",
        },
        {
            "metric": "reliable_speed_tracks",
            "value": len(reliable_tracks),
            "unit": "tracks",
            "interpretation": "Tracks with at least one reliable near/mid speed row.",
        },
        {
            "metric": "overspeed_track_share",
            "value": format_value(safe_pct(len(overspeed_events), len(reliable_tracks))),
            "unit": "% of reliable-speed tracks",
            "interpretation": "Track-level screening share above threshold.",
        },
    ]
    for label in sorted(label_counts):
        summary.append(
            {
                "metric": f"{label}_overspeed_tracks",
                "value": label_counts[label],
                "unit": "tracks",
                "interpretation": f"Overspeed screening events for dominant class {label}.",
            }
        )
    return summary


def rows_by_track(rows: list[dict]) -> dict[int, list[dict]]:
    by_track: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_track[row["track_id"]].append(row)
    return by_track


def dominant_label(rows: list[dict]) -> str:
    labels = Counter(row["label"] for row in rows)
    return labels.most_common(1)[0][0] if labels else ""


def dominant_segment(rows: list[dict]) -> str:
    segments = Counter(row["calibration_segment"] for row in rows)
    return segments.most_common(1)[0][0] if segments else ""


def reliable_policy_speed_rows(rows: list[dict]) -> list[dict]:
    return [row for row in rows if is_reliable_policy_speed_row(row)]


def is_reliable_policy_speed_row(row: dict) -> bool:
    return (
        row["speed_kmh"] > 0
        and row["speed_source"] in RELIABLE_SPEED_SOURCES
        and row["calibration_segment"] in RELIABLE_POLICY_SEGMENTS
    )


def dominant_track_labels(rows: list[dict]) -> dict[int, str]:
    grouped: dict[int, Counter] = defaultdict(Counter)
    for row in rows:
        grouped[row["track_id"]][row["label"]] += 1
    return {
        track_id: counts.most_common(1)[0][0]
        for track_id, counts in grouped.items()
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: str) -> float:
    try:
        return float(value)
    except ValueError:
        return float("nan")


def median(values: list[float]) -> float | str:
    clean = [value for value in values if value == value]
    return statistics.median(clean) if clean else ""


def percentile(values: list[float], pct: float) -> float | str:
    clean = sorted(value for value in values if value == value)
    if not clean:
        return ""
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * (pct / 100.0)
    low = int(rank)
    high = min(low + 1, len(clean) - 1)
    fraction = rank - low
    return clean[low] + (clean[high] - clean[low]) * fraction


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def safe_pct(numerator: float, denominator: float) -> float:
    return safe_divide(numerator, denominator) * 100.0


def format_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.2f}"


if __name__ == "__main__":
    raise SystemExit(main())
