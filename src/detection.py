"""Vehicle detection for the VehicleTrack baseline pipeline."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Sequence

import cv2
import torch
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms.functional import to_tensor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))


COCO_VEHICLE_LABELS = {
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    label: str
    score: float
    class_id: int


class FasterRCNNVehicleDetector:
    """COCO-pretrained Faster R-CNN filtered to vehicle classes."""

    def __init__(
        self,
        score_threshold: float = 0.7,
        device: str | None = None,
    ) -> None:
        self.score_threshold = score_threshold
        self.device = torch.device(device or self._default_device())
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights).to(self.device)
        self.model.eval()

    def detect(self, frame_bgr) -> list[Detection]:
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        tensor = to_tensor(image_rgb).to(self.device)

        with torch.no_grad():
            output = self.model([tensor])[0]

        boxes = output["boxes"].detach().cpu().numpy()
        labels = output["labels"].detach().cpu().numpy()
        scores = output["scores"].detach().cpu().numpy()
        return filter_vehicle_detections(boxes, labels, scores, self.score_threshold)

    @staticmethod
    def _default_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


def filter_vehicle_detections(
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Sequence[float],
    score_threshold: float,
) -> list[Detection]:
    detections: list[Detection] = []
    for box, class_id, score in zip(boxes, labels, scores, strict=True):
        class_id_int = int(class_id)
        score_float = float(score)
        if score_float < score_threshold:
            continue
        if class_id_int not in COCO_VEHICLE_LABELS:
            continue
        x1, y1, x2, y2 = (float(value) for value in box)
        detections.append(
            Detection(
                bbox=(x1, y1, x2, y2),
                label=COCO_VEHICLE_LABELS[class_id_int],
                score=score_float,
                class_id=class_id_int,
            )
        )
    return detections
