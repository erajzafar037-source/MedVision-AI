from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_binary_metrics(
    labels,
    probabilities,
    threshold: float = 0.5,
) -> dict:
    """Calculate binary-classification metrics at a fixed decision threshold."""
    y_true = np.asarray(labels, dtype=int).reshape(-1)
    y_score = np.asarray(probabilities, dtype=float).reshape(-1)

    if y_true.size == 0:
        raise ValueError("Cannot evaluate an empty set of predictions.")
    if y_true.size != y_score.size:
        raise ValueError("Labels and probabilities must have the same length.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("Labels must contain only binary values 0 or 1.")
    if not np.isfinite(y_score).all():
        raise ValueError("Probabilities must all be finite numbers.")
    if ((y_score < 0) | (y_score > 1)).any():
        raise ValueError("Probabilities must be between 0 and 1.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Decision threshold must be between 0 and 1.")
    if np.unique(y_true).size != 2:
        raise ValueError(
            "ROC-AUC evaluation requires both classes in the evaluation set."
        )

    predictions = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )

    return {
        "threshold": float(threshold),
        "sample_count": int(y_true.size),
        "class_counts": {
            "no_finding": int((y_true == 0).sum()),
            "pneumonia": int((y_true == 1).sum()),
        },
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "sensitivity": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(
            true_negative / max(true_negative + false_positive, 1)
        ),
        "precision": float(
            precision_score(y_true, predictions, zero_division=0)
        ),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "confusion_matrix": matrix.tolist(),
        "confusion_matrix_labels": ["No Finding", "Pneumonia"],
    }
