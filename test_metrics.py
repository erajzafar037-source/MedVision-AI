import pytest

from metrics import calculate_binary_metrics


def test_binary_metrics_match_expected_confusion_matrix():
    result = calculate_binary_metrics(
        labels=[0, 0, 1, 1],
        probabilities=[0.10, 0.80, 0.70, 0.60],
        threshold=0.50,
    )

    assert result["sample_count"] == 4
    assert result["class_counts"] == {"no_finding": 2, "pneumonia": 2}
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["sensitivity"] == pytest.approx(1.0)
    assert result["specificity"] == pytest.approx(0.5)
    assert result["precision"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    "labels, probabilities, threshold, message",
    [
        ([], [], 0.5, "empty"),
        ([0, 1], [0.2], 0.5, "same length"),
        ([0, 2], [0.2, 0.8], 0.5, "binary"),
        ([0, 1], [0.2, float("nan")], 0.5, "finite"),
        ([0, 1], [0.2, 1.2], 0.5, "between 0 and 1"),
        ([0, 1], [0.2, 0.8], 1.5, "threshold"),
        ([0, 0], [0.2, 0.3], 0.5, "both classes"),
    ],
)
def test_binary_metrics_reject_invalid_inputs(
    labels, probabilities, threshold, message
):
    with pytest.raises(ValueError, match=message):
        calculate_binary_metrics(labels, probabilities, threshold)
