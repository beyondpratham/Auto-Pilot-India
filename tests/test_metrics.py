from autopilot.metrics import compute_metrics


def test_compute_metrics_perfect_predictions():
    y_true = [0, 1, 2, 1]
    y_pred = [0, 1, 2, 1]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0


def test_compute_metrics_partial_predictions():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]

    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
