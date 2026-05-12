"""Interactive 4-point Homography calibration tool for VehicleTrack.

Usage:
    python project/calibrate.py project/data/highway_self.mp4
    python project/calibrate.py project/data/highway_self.mp4 --time-sec 16.9 --save-frame project/outputs/calibration_frame.jpg
    python project/calibrate.py project/data/highway_self.mp4 \
        --source-points "100,600;500,600;430,300;180,300" --width-m 3.66 --depth-m 12.19

The tool opens a calibration frame, lets the user click four road-plane
points, asks for the real-world width/depth in meters, and saves a calibration
JSON file under project/calibration/ by default.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from homography import HomographyCalibration


WINDOW_NAME = "VehicleTrack calibration - click 4 road-plane points"


def main() -> int:
    args = parse_args()
    video_path = Path(args.video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    frame, fps, frame_count, frame_idx, timestamp = read_calibration_frame(
        video_path,
        frame_index=args.frame_index,
        time_sec=args.time_sec,
    )
    if args.save_frame:
        save_frame(frame, Path(args.save_frame).expanduser().resolve())
        return 0

    source_points = parse_source_points(args.source_points) if args.source_points else collect_points(frame)
    width_m = args.width_m or prompt_positive_float(
        "Real-world width between point 1 and point 2, in meters "
        "(example lane width: 3.66): "
    )
    depth_m = args.depth_m or prompt_positive_float(
        "Real-world depth between near edge and far edge, in meters "
        "(example US lane-marking cycle: 12.19): "
    )

    destination_points = np.asarray(
        [
            [0.0, 0.0],
            [width_m, 0.0],
            [width_m, depth_m],
            [0.0, depth_m],
        ],
        dtype=float,
    )

    output_path = resolve_output_path(args.output, video_path)
    calibration = HomographyCalibration.from_points(
        source_points,
        destination_points,
        metadata={
            "video_path": str(video_path),
            "video_name": video_path.name,
            "fps": fps,
            "frame_count": frame_count,
            "calibration_frame_idx": frame_idx,
            "calibration_timestamp": timestamp,
            "width_m": width_m,
            "depth_m": depth_m,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "point_order": "near-left, near-right, far-right, far-left",
        },
    )
    calibration.save(output_path)
    print(f"Saved calibration to {output_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a VehicleTrack calibration JSON.")
    parser.add_argument("video", help="Path to the video to calibrate.")
    parser.add_argument(
        "--output",
        "-o",
        help="Output calibration JSON path. Defaults to project/calibration/<video>.calibration.json.",
    )
    parser.add_argument(
        "--save-frame",
        help="Save the selected calibration frame to this image path and exit without calibration.",
    )
    parser.add_argument("--frame-index", type=int, help="Frame index to use for calibration.")
    parser.add_argument("--time-sec", type=float, help="Timestamp in seconds to use for calibration.")
    parser.add_argument(
        "--source-points",
        help=(
            "Non-interactive source points as 'x1,y1;x2,y2;x3,y3;x4,y4' "
            "in near-left, near-right, far-right, far-left order."
        ),
    )
    parser.add_argument("--width-m", type=positive_float, help="Real-world width in meters.")
    parser.add_argument("--depth-m", type=positive_float, help="Real-world depth in meters.")
    return parser.parse_args()


def read_calibration_frame(
    video_path: Path,
    frame_index: int | None = None,
    time_sec: float | None = None,
) -> tuple[np.ndarray, float, int, int, float]:
    if frame_index is not None and time_sec is not None:
        raise ValueError("Use either --frame-index or --time-sec, not both")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if frame_index is not None:
        if frame_index < 0:
            raise ValueError("--frame-index must be non-negative")
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    elif time_sec is not None:
        if time_sec < 0:
            raise ValueError("--time-sec must be non-negative")
        cap.set(cv2.CAP_PROP_POS_MSEC, time_sec * 1000)

    ok, frame = cap.read()
    actual_frame_idx = max(int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 1) - 1, 0)
    actual_timestamp = actual_frame_idx / fps if fps else 0.0
    cap.release()

    if not ok or frame is None:
        raise RuntimeError(f"Could not read calibration frame from: {video_path}")
    return frame, fps, frame_count, actual_frame_idx, actual_timestamp


def collect_points(frame: np.ndarray) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    display = frame.copy()

    def redraw() -> None:
        display[:] = frame
        for idx, point in enumerate(points, start=1):
            x, y = int(point[0]), int(point[1])
            cv2.circle(display, (x, y), 6, (0, 255, 255), -1)
            cv2.putText(
                display,
                str(idx),
                (x + 8, y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )
        if len(points) == 4:
            poly = np.asarray(points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(display, [poly], isClosed=True, color=(0, 255, 0), thickness=2)

        instruction = "Click: near-left, near-right, far-right, far-left | Enter=save | r=reset | Esc=quit"
        cv2.putText(display, instruction, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
        cv2.putText(display, instruction, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((float(x), float(y)))
            redraw()

    redraw()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, on_mouse)

    while True:
        cv2.imshow(WINDOW_NAME, display)
        key = cv2.waitKey(20) & 0xFF

        if key in {13, 10}:  # Enter / Return
            if len(points) == 4:
                break
            print(f"Need exactly 4 points; currently have {len(points)}.")
        elif key == ord("r"):
            points.clear()
            redraw()
        elif key == 27:  # Escape
            cv2.destroyWindow(WINDOW_NAME)
            raise KeyboardInterrupt("Calibration cancelled by user.")

    cv2.destroyWindow(WINDOW_NAME)
    return points


def prompt_positive_float(prompt: str) -> float:
    while True:
        raw = input(prompt).strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number.")
            continue
        if value > 0:
            return value
        print("Please enter a positive value.")


def positive_float(raw: str) -> float:
    value = float(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def parse_source_points(raw: str) -> list[tuple[float, float]]:
    points = []
    for item in raw.split(";"):
        try:
            x_raw, y_raw = item.split(",", maxsplit=1)
            points.append((float(x_raw.strip()), float(y_raw.strip())))
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "source points must look like 'x1,y1;x2,y2;x3,y3;x4,y4'"
            ) from exc
    if len(points) != 4:
        raise argparse.ArgumentTypeError("source points must contain exactly four points")
    return points


def save_frame(frame: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), frame)
    if not ok:
        raise RuntimeError(f"Could not save frame to {output_path}")
    print(f"Saved first frame to {output_path}")


def resolve_output_path(output: str | None, video_path: Path) -> Path:
    if output:
        return Path(output).expanduser().resolve()
    return PROJECT_ROOT / "calibration" / f"{video_path.stem}.calibration.json"


if __name__ == "__main__":
    raise SystemExit(main())
