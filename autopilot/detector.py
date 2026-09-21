from dataclasses import dataclass

import torch
from PIL import Image
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_320_fpn,
)

from autopilot.cnn_model import default_device


@dataclass
class Detection:
    label: str
    score: float
    box: tuple


class ObjectDetector:
    def __init__(self, device=None, score_threshold=0.5):
        self.device = device or default_device()
        self.score_threshold = score_threshold
        weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.COCO_V1
        self.model = fasterrcnn_mobilenet_v3_large_320_fpn(
            weights=weights, box_score_thresh=score_threshold
        ).to(self.device).eval()
        self.categories = weights.meta["categories"]
        self.transform = weights.transforms()

    @torch.no_grad()
    def detect(self, image_rgb):
        tensor = self.transform(Image.fromarray(image_rgb)).to(self.device)
        output = self.model([tensor])[0]

        detections = []
        for box, label_idx, score in zip(output["boxes"], output["labels"], output["scores"]):
            if score < self.score_threshold:
                continue
            detections.append(
                Detection(
                    label=self.categories[label_idx],
                    score=float(score),
                    box=tuple(box.tolist()),
                )
            )
        return detections
