from autopilot.detector import Detection
from autopilot.navigation import decide_direction, segment_index_for_box


def test_segment_index_for_box_left_center_right():
    assert segment_index_for_box((0, 0, 10, 10), image_width=90, num_segments=3) == 0
    assert segment_index_for_box((40, 0, 50, 10), image_width=90, num_segments=3) == 1
    assert segment_index_for_box((80, 0, 89, 10), image_width=90, num_segments=3) == 2


def test_decide_direction_prefers_segment_without_risky_detections():
    detections = [
        Detection(label="person", score=0.9, box=(0, 0, 20, 80)),
        Detection(label="car", score=0.8, box=(40, 0, 60, 40)),
    ]

    decision = decide_direction(
        (100, 90, 3),
        detections,
        risk_weights={"human": 0.15, "vehicle": 0.10},
        class_map={"person": "human", "car": "vehicle"},
    )

    assert decision.direction == "Go Right"


def test_decide_direction_ignores_unmapped_labels():
    detections = [Detection(label="traffic light", score=0.99, box=(40, 0, 50, 10))]

    decision = decide_direction(
        (100, 90, 3),
        detections,
        risk_weights={"human": 0.5},
        class_map={"person": "human"},
    )

    assert decision.direction != "Brake"
    assert all(segment.risk == 0.0 for segment in decision.segments)


def test_decide_direction_brakes_when_every_segment_is_risky():
    detections = [
        Detection(label="person", score=1.0, box=(x, 0, x + 30, 100)) for x in (0, 30, 60)
    ]

    decision = decide_direction(
        (100, 90, 3),
        detections,
        risk_weights={"human": 0.5},
        class_map={"person": "human"},
        brake_threshold=0.12,
    )

    assert decision.direction == "Brake"
