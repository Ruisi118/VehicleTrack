"""Check whether public TGSIM resources can validate the CV pipeline.

The core question is narrower than "is TGSIM useful?": do the public resources
include raw video plus frame-level labels that can be aligned to VehicleTrack's
detector/tracker outputs? This script records the resource evidence and writes
an explicit feasibility decision.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "outputs" / "tgsim_pipeline_validation"
DATASETS = [
    {
        "short_name": "foggy_bottom",
        "dataset_name": "TGSIM Foggy Bottom Trajectories",
        "view_id": "brzy-6zfh",
        "landing_page": "https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-foggy-bottom-trajectories",
        "setting": "Urban neighborhood, Washington, D.C.",
    },
    {
        "short_name": "i395",
        "dataset_name": "TGSIM I-395 Trajectories",
        "view_id": "97n2-kuqi",
        "landing_page": "https://catalog.data.gov/dataset/third-generation-simulation-data-tgsim-i-395-trajectories",
        "setting": "Urban expressway, Washington, D.C.",
    },
]
VIDEO_KEYWORDS = ("video", "mp4", "mov", "avi", "mkv", "frame", "image sequence")
ANNOTATION_KEYWORDS = ("annotation", "label", "ground truth", "bbox", "bounding box", "mot")
SUPPORTING_KEYWORDS = ("reference", "ref_image", "boundaries", "dictionary", "regions", "lanes")


def main() -> int:
    args = parse_args()
    output_dir = Path(args.out).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    resource_rows = []
    feasibility_rows = []
    for dataset in DATASETS:
        metadata = fetch_json(f"https://data.transportation.gov/api/views/{dataset['view_id']}")
        attachments = metadata.get("metadata", {}).get("attachments", [])
        resource_rows.extend(build_resource_rows(dataset, metadata, attachments))
        feasibility_rows.append(build_feasibility_row(dataset, metadata, attachments))

    write_csv(output_dir / "tgsim_public_resource_inventory.csv", resource_rows)
    write_csv(output_dir / "tgsim_pipeline_validation_feasibility.csv", feasibility_rows)
    print(f"Wrote TGSIM pipeline-validation feasibility artifacts to {output_dir}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check TGSIM public resources for raw-video pipeline validation feasibility.")
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    return parser.parse_args()


def fetch_json(url: str) -> dict:
    with urlopen(url, timeout=60) as response:
        return json.loads(response.read().decode("utf-8"))


def build_resource_rows(dataset: dict, metadata: dict, attachments: list[dict]) -> list[dict]:
    rows = [
        {
            "dataset": dataset["dataset_name"],
            "resource_name": "trajectory_table_csv",
            "resource_type": "processed_trajectory_csv",
            "public_evidence": metadata.get("name", ""),
            "contains_raw_video": "no",
            "contains_frame_level_detection_labels": "no",
            "pipeline_validation_use": "traffic_analysis_anchor_only",
            "notes": "Socrata table contains processed trajectories with physical position, speed, acceleration, lane/region, and type fields.",
        }
    ]
    for attachment in attachments:
        filename = attachment.get("filename") or attachment.get("name") or ""
        lower = filename.lower()
        rows.append(
            {
                "dataset": dataset["dataset_name"],
                "resource_name": filename,
                "resource_type": classify_resource(lower),
                "public_evidence": filename,
                "contains_raw_video": "yes" if is_video_resource(lower) else "no",
                "contains_frame_level_detection_labels": "yes" if is_frame_label_resource(lower) else "no",
                "pipeline_validation_use": pipeline_use(lower),
                "notes": resource_notes(lower),
            }
        )
    return rows


def build_feasibility_row(dataset: dict, metadata: dict, attachments: list[dict]) -> dict:
    attachment_names = [attachment.get("filename") or attachment.get("name") or "" for attachment in attachments]
    has_video = any(is_video_resource(name.lower()) for name in attachment_names)
    has_frame_labels = any(is_frame_label_resource(name.lower()) for name in attachment_names)
    has_reference_support = any(is_supporting_resource(name.lower()) for name in attachment_names)
    can_validate = has_video and has_frame_labels
    if can_validate:
        decision = "potentially_feasible"
        interpretation = "Public resources appear to include both raw video and frame-level labels; a small validation run can be scoped."
    else:
        decision = "not_feasible_from_public_resources"
        interpretation = (
            "Public resources expose processed trajectory CSV and calibration/reference support, "
            "but not raw video plus frame-level detection/tracking labels. They can anchor traffic analysis, "
            "not directly validate VehicleTrack detector/tracker accuracy."
        )

    return {
        "dataset": dataset["dataset_name"],
        "view_id": dataset["view_id"],
        "setting": dataset["setting"],
        "landing_page": dataset["landing_page"],
        "has_processed_trajectory_csv": "yes",
        "has_reference_or_boundary_support": "yes" if has_reference_support else "unknown",
        "has_public_raw_video": "yes" if has_video else "no",
        "has_public_frame_level_labels": "yes" if has_frame_labels else "no",
        "can_directly_validate_vehicletrack_cv_pipeline": "yes" if can_validate else "no",
        "decision": decision,
        "recommended_project_use": "physical_trajectory_analysis_anchor",
        "interpretation": interpretation,
    }


def classify_resource(name: str) -> str:
    if is_video_resource(name):
        return "raw_video_or_video_container"
    if is_frame_label_resource(name):
        return "frame_level_annotation_candidate"
    if is_supporting_resource(name):
        return "reference_or_region_support"
    return "other_attachment"


def is_video_resource(name: str) -> bool:
    return any(keyword in name for keyword in VIDEO_KEYWORDS)


def is_frame_label_resource(name: str) -> bool:
    return any(keyword in name for keyword in ANNOTATION_KEYWORDS) and any(keyword in name for keyword in ("bbox", "bounding", "mot", "frame", "label"))


def is_supporting_resource(name: str) -> bool:
    return any(keyword in name for keyword in SUPPORTING_KEYWORDS)


def pipeline_use(name: str) -> str:
    if is_video_resource(name):
        return "candidate_video_input"
    if is_frame_label_resource(name):
        return "candidate_ground_truth"
    if is_supporting_resource(name):
        return "calibration_context_only"
    return "supporting_metadata"


def resource_notes(name: str) -> str:
    if is_video_resource(name):
        return "Would need frame timestamps and matching ground truth before VehicleTrack validation."
    if is_frame_label_resource(name):
        return "Attachment name suggests annotation, but not necessarily detection/tracking labels."
    if "region" in name or "lane" in name:
        return "Maps lane/region IDs to reference image; useful for analysis, not detector validation."
    if "reference" in name or "boundaries" in name:
        return "Supports metric coordinate interpretation and lane/region mapping."
    if "dictionary" in name:
        return "Explains fields and units."
    return ""


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
