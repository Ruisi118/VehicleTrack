from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from homography import (
    HomographyCalibration,
    bbox_bottom_midpoint,
    estimate_speed_kmh,
)


class HomographyCalibrationTest(unittest.TestCase):
    def test_maps_source_rectangle_to_destination_rectangle(self):
        calibration = HomographyCalibration.from_points(
            source_points=[
                (100, 200),
                (300, 200),
                (300, 600),
                (100, 600),
            ],
            destination_points=[
                (0.0, 0.0),
                (4.0, 0.0),
                (4.0, 20.0),
                (0.0, 20.0),
            ],
        )

        np.testing.assert_allclose(calibration.pixel_to_world(100, 200), (0.0, 0.0), atol=1e-8)
        np.testing.assert_allclose(calibration.pixel_to_world(300, 600), (4.0, 20.0), atol=1e-8)
        np.testing.assert_allclose(calibration.pixel_to_world(200, 400), (2.0, 10.0), atol=1e-8)

    def test_bbox_bottom_midpoint_uses_ground_contact_proxy(self):
        self.assertEqual(bbox_bottom_midpoint((10, 20, 30, 80)), (20.0, 80.0))

    def test_contains_pixel_checks_calibration_roi(self):
        calibration = HomographyCalibration.from_points(
            source_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            destination_points=[(0, 0), (2, 0), (2, 2), (0, 2)],
        )

        self.assertTrue(calibration.contains_pixel(5, 5))
        self.assertTrue(calibration.contains_pixel(0, 5))
        self.assertFalse(calibration.contains_pixel(12, 5))

    def test_estimate_speed_uses_timestamps_not_record_count(self):
        records = [
            {"frame_idx": 0, "timestamp": 0.0, "world_x": 0.0, "world_y": 0.0},
            {"frame_idx": 3, "timestamp": 0.3, "world_x": 0.0, "world_y": 3.0},
        ]

        self.assertAlmostEqual(estimate_speed_kmh(records), 36.0)

    def test_round_trip_save_and_load(self):
        calibration = HomographyCalibration.from_points(
            source_points=[(0, 0), (10, 0), (10, 10), (0, 10)],
            destination_points=[(0, 0), (2, 0), (2, 2), (0, 2)],
            metadata={"video_name": "sample.mp4"},
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibration.json"
            calibration.save(path)
            loaded = HomographyCalibration.load(path)

        self.assertEqual(loaded.metadata["video_name"], "sample.mp4")
        np.testing.assert_allclose(loaded.pixel_to_world(5, 5), (1.0, 1.0), atol=1e-8)

    def test_loads_multi_segment_calibration_and_selects_segment(self):
        data = {
            "calibration_type": "multi_segment",
            "segments": [
                {
                    "name": "near",
                    "source_points": [(0, 0), (10, 0), (10, 10), (0, 10)],
                    "destination_points": [(0, 0), (2, 0), (2, 10), (0, 10)],
                },
                {
                    "name": "far",
                    "source_points": [(0, 10), (10, 10), (10, 20), (0, 20)],
                    "destination_points": [(0, 10), (2, 10), (2, 20), (0, 20)],
                },
            ],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.json"
            path.write_text(__import__("json").dumps(data), encoding="utf-8")
            calibration = HomographyCalibration.load(path)

        near = calibration.project_bbox((2, 2, 4, 4))
        far = calibration.project_bbox((2, 12, 4, 14))
        outside = calibration.project_bbox((20, 20, 30, 30))

        self.assertEqual(near["calibration_segment"], "near")
        self.assertEqual(far["calibration_segment"], "far")
        self.assertTrue(near["in_calibration_roi"])
        self.assertFalse(outside["in_calibration_roi"])
        np.testing.assert_allclose((near["world_x"], near["world_y"]), (0.6, 4.0), atol=1e-8)
        np.testing.assert_allclose((far["world_x"], far["world_y"]), (0.6, 14.0), atol=1e-8)


if __name__ == "__main__":
    unittest.main()
