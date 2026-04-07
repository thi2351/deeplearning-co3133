from __future__ import annotations

from typing import Dict, Sequence

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int]) -> Dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }



def full_classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
) -> str:
    return classification_report(y_true, y_pred, target_names=list(class_names), digits=4)



def build_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int):
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
