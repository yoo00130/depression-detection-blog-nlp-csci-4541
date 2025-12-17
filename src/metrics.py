"""
Evaluation metrics for binary depression detection.

Metrics:
  - Accuracy
  - Precision
  - Recall
  - Macro-F1
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def compute_metrics(y_true, y_pred):
    """
    Compute standard classification metrics.

    Args:
        y_true (list or array): Gold labels (0 or 1)
        y_pred (list or array): Predicted labels (0 or 1)

    Returns:
        dict: Dictionary with accuracy, precision, recall, macro_f1
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }

    return metrics
