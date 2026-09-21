from dataclasses import dataclass

from autopilot.config import (
    BRAKE_RISK_THRESHOLD,
    COCO_TO_RISK_CLASS,
    RISK_WEIGHTS,
    SEGMENT_DIRECTIONS,
)


@dataclass
class SegmentAssessment:
    label: str
    risk: float
    cause: str


@dataclass
class NavigationDecision:
    direction: str
    segments: list
    detections: list


def segment_index_for_box(box, image_width, num_segments):
    x1, _, x2, _ = box
    center_x = (x1 + x2) / 2
    index = int(center_x / image_width * num_segments)
    return min(max(index, 0), num_segments - 1)


def decide_direction(
    image_shape,
    detections,
    risk_weights=None,
    class_map=None,
    brake_threshold=BRAKE_RISK_THRESHOLD,
):
    risk_weights = risk_weights if risk_weights is not None else RISK_WEIGHTS
    class_map = class_map if class_map is not None else COCO_TO_RISK_CLASS
    height, width = image_shape[0], image_shape[1]
    num_segments = len(SEGMENT_DIRECTIONS)

    segment_risks = [0.0] * num_segments
    segment_causes = [""] * num_segments

    for detection in detections:
        category = class_map.get(detection.label)
        if category is None:
            continue

        x1, y1, x2, y2 = detection.box
        proximity = min(1.0, max(0.0, (y2 - y1)) / height)
        risk = risk_weights.get(category, 0.0) * detection.score * (0.5 + 0.5 * proximity)

        index = segment_index_for_box(detection.box, width, num_segments)
        if risk > segment_risks[index]:
            segment_risks[index] = risk
            segment_causes[index] = f"{detection.label} ({detection.score:.2f})"

    assessments = [
        SegmentAssessment(label=SEGMENT_DIRECTIONS[i], risk=segment_risks[i], cause=segment_causes[i])
        for i in range(num_segments)
    ]

    safest = min(assessments, key=lambda a: a.risk)
    direction = "Brake" if safest.risk > brake_threshold else safest.label

    return NavigationDecision(direction=direction, segments=assessments, detections=detections)
