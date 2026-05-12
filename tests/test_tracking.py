from pathlib import Path
import os
import sys
import unittest

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"),
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from detection import Detection
from homography import HomographyCalibration
from tracking import ByteTrackTracker, IoUTracker, Track


def test_calibration() -> HomographyCalibration:
    return HomographyCalibration.from_points(
        source_points=[(0, 0), (100, 0), (100, 100), (0, 100)],
        destination_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
    )


class TrackingBackendsTest(unittest.TestCase):
    def test_iou_tracker_reports_total_tracks_created(self):
        tracker = IoUTracker(calibration=test_calibration())
        tracks = tracker.update(
            [
                Detection(
                    bbox=(10, 10, 20, 20),
                    label="car",
                    score=0.9,
                    class_id=3,
                )
            ],
            frame_idx=0,
            timestamp=0.0,
        )

        self.assertEqual(len(tracks), 1)
        self.assertEqual(tracker.total_tracks_created, 1)

    def test_bytetrack_tracker_reuses_track_id_for_continuous_detection(self):
        tracker = ByteTrackTracker(calibration=test_calibration())
        first_tracks = tracker.update(
            [
                Detection(
                    bbox=(10, 10, 20, 20),
                    label="car",
                    score=0.9,
                    class_id=3,
                )
            ],
            frame_idx=0,
            timestamp=0.0,
        )
        second_tracks = tracker.update(
            [
                Detection(
                    bbox=(11, 10, 21, 20),
                    label="car",
                    score=0.9,
                    class_id=3,
                )
            ],
            frame_idx=1,
            timestamp=0.04,
        )

        self.assertEqual(len(first_tracks), 1)
        self.assertEqual(len(second_tracks), 1)
        self.assertEqual(first_tracks[0].track_id, second_tracks[0].track_id)
        self.assertEqual(tracker.total_tracks_created, 1)
        self.assertEqual(len(tracker.all_records()), 2)

    def test_track_uses_cross_segment_smoothing_after_boundary_crossing(self):
        calibration = HomographyCalibration.load(
            PROJECT_ROOT / "calibration" / "video.multisegment.calibration.json"
        )
        track = Track(
            track_id=1,
            bbox=(700, 560, 760, 600),
            label="car",
            score=0.9,
            created_frame_idx=0,
        )

        first = Detection(
            bbox=(700, 560, 760, 600),
            label="car",
            score=0.9,
            class_id=3,
        )
        second = Detection(
            bbox=(700, 460, 760, 500),
            label="car",
            score=0.9,
            class_id=3,
        )
        third = Detection(
            bbox=(700, 420, 760, 460),
            label="car",
            score=0.9,
            class_id=3,
        )

        track.update(first, frame_idx=0, timestamp=0.0, calibration=calibration)
        track.update(second, frame_idx=1, timestamp=1.0, calibration=calibration)
        track.update(third, frame_idx=2, timestamp=2.0, calibration=calibration)

        records = track.records
        self.assertEqual(records[-1]["calibration_segment"], "mid_two_lane")
        self.assertEqual(records[-1]["speed_source"], "cross_segment_smoothed")
        self.assertIn("raw_speed_kmh", records[-1])


if __name__ == "__main__":
    unittest.main()
