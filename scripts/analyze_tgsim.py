"""Fetch and profile a small FHWA TGSIM Foggy Bottom trajectory sample.

This script intentionally does not download the full 350 MB CSV. It reads the
official U.S. DOT Socrata API, saves a bounded sample, and writes lightweight
schema/profile artifacts that can support the report's dataset narrative.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
from urllib.parse import urlencode
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "tgsim"
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "tgsim_analysis"
DATASET_ID = "brzy-6zfh"
DATASET_NAME = "TGSIM Foggy Bottom Trajectories"
LANDING_PAGE = "https://data.transportation.gov/d/brzy-6zfh"
COLUMNS_URL = f"https://data.transportation.gov/api/views/{DATASET_ID}/columns.json"
RESOURCE_URL = f"https://data.transportation.gov/resource/{DATASET_ID}.json"
DOWNLOAD_URL = f"https://data.transportation.gov/api/views/{DATASET_ID}/rows.csv?accessType=DOWNLOAD"
TYPE_LABELS = {
    "0": "pedestrian",
    "1": "bicycle",
    "2": "scooter",
    "3": "passenger_car",
    "4": "automated_vehicle",
    "5": "motorcycle",
    "6": "bus",
    "7": "truck",
}


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    out_dir = Path(args.out).expanduser().resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    columns = fetch_json(COLUMNS_URL)
    rows = fetch_sample(args.sample_rows)
    typed_rows = [coerce_row(row) for row in rows]

    sample_csv_path = data_dir / "tgsim_foggy_bottom_sample.csv"
    write_csv(sample_csv_path, rows)
    write_csv(out_dir / "tgsim_resources.csv", build_resource_rows(columns))
    write_csv(out_dir / "tgsim_schema_preview.csv", build_schema_preview(columns, rows))
    write_csv(out_dir / "tgsim_sample_profile.csv", build_sample_profile(columns, typed_rows, args.sample_rows))
    write_csv(out_dir / "tgsim_type_composition.csv", build_type_composition(typed_rows))
    write_csv(out_dir / "tgsim_lane_occupancy.csv", build_lane_occupancy(typed_rows))
    write_csv(out_dir / "tgsim_speed_summary_by_type.csv", build_speed_summary_by_type(typed_rows))
    write_csv(out_dir / "tgsim_vehicletrack_dataset_comparison.csv", build_dataset_comparison(columns, rows))

    print(f"Wrote TGSIM sample to {sample_csv_path}")
    print(f"Wrote TGSIM analysis artifacts to {out_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a small TGSIM Foggy Bottom sample.")
    parser.add_argument("--sample-rows", type=int, default=50_000)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser.parse_args()


def fetch_json(url: str) -> list[dict]:
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_sample(limit: int) -> list[dict]:
    params = urlencode({"$limit": limit, "$order": "time, id"})
    return fetch_json(f"{RESOURCE_URL}?{params}")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def coerce_row(row: dict) -> dict:
    return {
        "id": str(row.get("id", "")),
        "time": to_float(row.get("time")),
        "xloc_kf": to_float(row.get("xloc_kf")),
        "yloc_kf": to_float(row.get("yloc_kf")),
        "lane_kf": normalize_code(row.get("lane_kf")),
        "speed_kf_x": to_float(row.get("speed_kf_x")),
        "speed_kf_y": to_float(row.get("speed_kf_y")),
        "acceleration_kf_x": to_float(row.get("acceleration_kf_x")),
        "acceleration_kf_y": to_float(row.get("acceleration_kf_y")),
        "length_smoothed": to_float(row.get("length_smoothed")),
        "width_smoothed": to_float(row.get("width_smoothed")),
        "type_most_common": normalize_code(row.get("type_most_common")),
    }


def build_resource_rows(columns: list[dict]) -> list[dict]:
    total_rows = official_total_rows(columns)
    return [
        {
            "dataset": DATASET_NAME,
            "source": "U.S. DOT / FHWA, ITS DataHub via data.transportation.gov",
            "dataset_id": DATASET_ID,
            "landing_page": LANDING_PAGE,
            "csv_download_url": DOWNLOAD_URL,
            "columns_url": COLUMNS_URL,
            "approx_total_rows": total_rows,
            "license": "U.S. Public Domain",
            "project_use": "Primary public trajectory analysis dataset; physical-coordinate anchor for VehicleTrack report.",
        }
    ]


def build_schema_preview(columns: list[dict], rows: list[dict]) -> list[dict]:
    sample_values_by_col = defaultdict(list)
    for row in rows[:20]:
        for key, value in row.items():
            if value not in ("", None) and len(sample_values_by_col[key]) < 5:
                sample_values_by_col[key].append(value)

    output = []
    for column in columns:
        field = column["fieldName"]
        cached = column.get("cachedContents", {})
        output.append(
            {
                "column_name": column["name"],
                "api_field_name": field,
                "data_type": column["dataTypeName"],
                "description": column.get("description", ""),
                "role_guess": role_guess(field, column.get("description", "")),
                "non_null_count": cached.get("non_null", ""),
                "null_count": cached.get("null", ""),
                "smallest": cached.get("smallest", ""),
                "largest": cached.get("largest", ""),
                "sample_values": "; ".join(sample_values_by_col.get(field, [])),
            }
        )
    return output


def build_sample_profile(columns: list[dict], rows: list[dict], requested_rows: int) -> list[dict]:
    times = [row["time"] for row in rows if row["time"] is not None]
    speeds = [speed_mps(row) for row in rows if speed_mps(row) is not None]
    speed_kmh = [speed * 3.6 for speed in speeds]
    ids = {row["id"] for row in rows if row["id"]}
    lanes = {row["lane_kf"] for row in rows if row["lane_kf"]}
    type_counts = Counter(type_label(row["type_most_common"]) for row in rows)
    vehicle_rows = [row for row in rows if type_label(row["type_most_common"]) in {"passenger_car", "automated_vehicle", "motorcycle", "bus", "truck"}]

    return [
        profile("dataset", DATASET_NAME, "", "FHWA public trajectory dataset used as the physically grounded analysis layer."),
        profile("sample_requested_rows", requested_rows, "rows", "Configured API sample limit."),
        profile("sample_returned_rows", len(rows), "rows", "Rows returned by the Socrata API sample query."),
        profile("official_total_rows", official_total_rows(columns), "rows", "Cached row count in the official column metadata."),
        profile("columns", len(columns), "columns", "Available fields in the trajectory CSV."),
        profile("unique_road_users_in_sample", len(ids), "ids", "Unique road-user IDs observed in the sample."),
        profile("time_min", min(times) if times else "", "s", "Start time of sampled rows relative to run start."),
        profile("time_max", max(times) if times else "", "s", "End time of sampled rows relative to run start."),
        profile("sample_duration", (max(times) - min(times)) if len(times) > 1 else "", "s", "Time span covered by this bounded sample."),
        profile("sample_frequency", "0.1", "s", "Official trajectory interval described by TGSIM metadata."),
        profile("physical_position_fields", "xloc_kf, yloc_kf", "meters", "Global reference-image coordinates after Kalman filtering."),
        profile("speed_fields", "speed_kf_x, speed_kf_y", "m/s", "Velocity components; scalar speed is derived as sqrt(x^2 + y^2)."),
        profile("acceleration_fields", "acceleration_kf_x, acceleration_kf_y", "m/s^2", "Acceleration components available for future safety analysis."),
        profile("road_user_type_field", "type_most_common", "coded class", "0 pedestrian, 1 bicycle, 2 scooter, 3 passenger car, 4 automated vehicle, 5 motorcycle, 6 bus, 7 truck."),
        profile("lane_or_region_field", "lane_kf", "region id", f"{len(lanes)} unique lane/region IDs observed in sample."),
        profile("sample_vehicle_row_share", safe_pct(len(vehicle_rows), len(rows)), "%", "Rows for motor vehicles only: passenger car, AV, motorcycle, bus, truck."),
        profile("sample_median_speed_kmh", median(speed_kmh), "km/h", "Median scalar speed from official m/s velocity components."),
        profile("sample_p75_speed_kmh", percentile(speed_kmh, 75), "km/h", "Upper quartile scalar speed from official m/s velocity components."),
        profile("largest_sample_class", most_common(type_counts), "", "Most common road-user type in the sample."),
    ]


def build_type_composition(rows: list[dict]) -> list[dict]:
    row_counts = Counter(type_label(row["type_most_common"]) for row in rows)
    ids_by_type: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        ids_by_type[type_label(row["type_most_common"])].add(row["id"])

    output = []
    for label in sorted(row_counts):
        output.append(
            {
                "road_user_type": label,
                "sample_rows": row_counts[label],
                "sample_row_share_pct": fmt(safe_pct(row_counts[label], len(rows))),
                "unique_ids_in_sample": len(ids_by_type[label]),
            }
        )
    return output


def build_lane_occupancy(rows: list[dict]) -> list[dict]:
    lane_counts = Counter(row["lane_kf"] or "unknown" for row in rows)
    output = []
    for lane, count in lane_counts.most_common():
        output.append(
            {
                "lane_or_region_id": lane,
                "sample_rows": count,
                "sample_row_share_pct": fmt(safe_pct(count, len(rows))),
            }
        )
    return output


def build_speed_summary_by_type(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        speed = speed_mps(row)
        if speed is not None:
            grouped[type_label(row["type_most_common"])].append(speed * 3.6)

    output = []
    for label in sorted(grouped):
        speeds = grouped[label]
        output.append(
            {
                "road_user_type": label,
                "speed_rows": len(speeds),
                "median_speed_kmh": fmt(median(speeds)),
                "mean_speed_kmh": fmt(statistics.fmean(speeds) if speeds else None),
                "p25_speed_kmh": fmt(percentile(speeds, 25)),
                "p75_speed_kmh": fmt(percentile(speeds, 75)),
            }
        )
    return output


def build_dataset_comparison(columns: list[dict], sample_rows: list[dict]) -> list[dict]:
    vehicletrack_path = PROJECT_ROOT / "outputs" / "run_bytetrack_smooth_1min" / "trajectories.csv"
    vehicletrack_rows = 0
    vehicletrack_columns = 0
    vehicletrack_tracks = 0
    if vehicletrack_path.exists():
        with vehicletrack_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            track_ids = set()
            for row in reader:
                vehicletrack_rows += 1
                track_ids.add(row.get("track_id", ""))
            vehicletrack_columns = len(reader.fieldnames or [])
            vehicletrack_tracks = len(track_ids)

    return [
        {
            "dataset_layer": "TGSIM Foggy Bottom",
            "role_in_project": "Primary public physically grounded trajectory dataset",
            "source": "FHWA / U.S. DOT TGSIM",
            "input_modality": "Twelve stationary 4K infrastructure cameras processed into public trajectory CSV",
            "rows_available_or_generated": official_total_rows(columns),
            "columns": len(columns),
            "sample_rows_used_here": len(sample_rows),
            "time_resolution": "0.1 seconds",
            "physical_distance_basis": "Official x/y positions in meters, converted from reference-image pixels with published conversion factor",
            "speed_basis": "Official speed components in m/s; scalar speed derived from vector magnitude",
            "policy_use": "Grounded urban trajectory analysis: mode mix, speed, occupancy, lane/region structure, AV/non-AV comparisons",
            "does_validate_cv_pipeline": "No; it validates/anchors traffic analysis, not our detector/tracker without raw-video ground truth.",
        },
        {
            "dataset_layer": "VehicleTrack 60s pilot",
            "role_in_project": "Project-generated video-to-trajectory prototype",
            "source": "Self-recorded fixed-camera highway video",
            "input_modality": "Raw video processed by Faster R-CNN, ByteTrack, and multi-segment Homography",
            "rows_available_or_generated": vehicletrack_rows,
            "columns": vehicletrack_columns,
            "sample_rows_used_here": vehicletrack_rows,
            "time_resolution": "25 fps video; per-frame trajectory rows",
            "physical_distance_basis": "Estimated from road-plane Homography and lane-marking assumptions",
            "speed_basis": "Estimated from bbox bottom-midpoint displacement over time",
            "policy_use": "Demonstrates how local camera footage can be converted into structured traffic indicators",
            "does_validate_cv_pipeline": f"Partial prototype evidence only; {vehicletrack_tracks} generated track IDs require external MOT ground truth for formal validation.",
        },
    ]


def profile(indicator: str, value: object, unit: str, interpretation: str) -> dict:
    return {
        "indicator": indicator,
        "value": fmt(value),
        "unit": unit,
        "interpretation": interpretation,
    }


def speed_mps(row: dict) -> float | None:
    sx = row.get("speed_kf_x")
    sy = row.get("speed_kf_y")
    if sx is None or sy is None:
        return None
    return math.hypot(sx, sy)


def role_guess(field: str, description: str) -> str:
    text = f"{field} {description}".lower()
    if field == "id":
        return "road_user_id"
    if "time" in text:
        return "timestamp"
    if "x-coordinate" in text or field.startswith("xloc"):
        return "physical_x_position_m"
    if "y-coordinate" in text or field.startswith("yloc"):
        return "physical_y_position_m"
    if "lane" in text:
        return "lane_or_region_id"
    if "speed" in text:
        return "velocity_component_mps"
    if "acceleration" in text:
        return "acceleration_component_mps2"
    if "width" in text:
        return "road_user_width_m"
    if "length" in text:
        return "road_user_length_m"
    if "type" in text:
        return "road_user_type"
    return "other"


def official_total_rows(columns: list[dict]) -> str:
    if not columns:
        return ""
    return columns[0].get("cachedContents", {}).get("count", "")


def type_label(value: str) -> str:
    return TYPE_LABELS.get(str(value), f"unknown_{value}")


def normalize_code(value: object) -> str:
    if value in ("", None):
        return ""
    try:
        number = float(str(value))
    except ValueError:
        return str(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def to_float(value: object) -> float | None:
    if value in ("", None):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def safe_pct(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator * 100.0


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], percentile_value: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    position = (len(sorted_values) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[int(position)]
    fraction = position - lower
    return sorted_values[lower] * (1 - fraction) + sorted_values[upper] * fraction


def most_common(counter: Counter) -> str:
    if not counter:
        return ""
    label, count = counter.most_common(1)[0]
    return f"{label} ({count} sample rows)"


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.2f}"
    if value is None:
        return ""
    return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
