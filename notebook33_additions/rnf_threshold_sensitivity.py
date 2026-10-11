"""Post-hoc RNF threshold sensitivity analysis for multiclass probabilities.

This module does not train or modify a model. It applies an RNF-only decision
rule to probabilities produced by an already-fitted classifier.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_recall_fscore_support


def analyze_rnf_thresholds(
    y_true: Iterable[int],
    y_score: np.ndarray,
    thresholds: Iterable[float] = (0.30, 0.50, 0.70, 0.90),
    rnf_class_id: int = 5,
    normal_class_id: int = 0,
    output_csv: Optional[str] = None,
) -> Tuple[pd.DataFrame, dict]:
    """Compare argmax with RNF-only thresholds and return metrics and predictions.

    The threshold is applied only to windows that argmax assigns to RNF. If an
    RNF score is below a tested threshold, that window is reassigned to the
    highest-scoring non-RNF class. Predictions for all other windows are kept.

    ``output_csv`` is optional. CSV values retain full numeric precision; only
    the printed table is rounded to two decimal places.
    """
    y_true_array = np.asarray(y_true).reshape(-1)
    scores = np.asarray(y_score, dtype=float)
    threshold_values = tuple(float(value) for value in thresholds)

    if scores.ndim != 2:
        raise ValueError("y_score must be a 2D array of class probabilities.")
    if len(y_true_array) != scores.shape[0]:
        raise ValueError("y_true and y_score must contain the same number of rows.")
    if not np.isfinite(scores).all():
        raise ValueError("y_score contains non-finite values.")
    if not 0 <= rnf_class_id < scores.shape[1]:
        raise ValueError("rnf_class_id is outside the probability columns.")
    if not 0 <= normal_class_id < scores.shape[1]:
        raise ValueError("normal_class_id is outside the probability columns.")
    if any(value < 0.0 or value > 1.0 for value in threshold_values):
        raise ValueError("All thresholds must be between 0 and 1.")

    baseline = np.argmax(scores, axis=1)
    non_rnf_scores = scores.copy()
    non_rnf_scores[:, rnf_class_id] = -np.inf
    alternative = np.argmax(non_rnf_scores, axis=1)
    predictions = {"Original argmax": baseline.copy()}

    def summarize(rule_name: str, predicted: np.ndarray) -> dict:
        true_rnf = y_true_array == rnf_class_id
        predicted_rnf = predicted == rnf_class_id
        tp = int(np.sum(true_rnf & predicted_rnf))
        fp = int(np.sum(~true_rnf & predicted_rnf))
        fn = int(np.sum(true_rnf & ~predicted_rnf))
        tn = int(np.sum(~true_rnf & ~predicted_rnf))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_array,
            predicted,
            labels=[rnf_class_id],
            average=None,
            zero_division=0,
        )
        return {
            "Decision rule": rule_name,
            "RNF precision": float(precision[0]),
            "RNF recall": float(recall[0]),
            "RNF F1": float(f1[0]),
            "RNF TP": tp,
            "RNF FP": fp,
            "RNF FN": fn,
            "RNF FPR": fp / (fp + tn) if (fp + tn) else np.nan,
            "RNF alarms": tp + fp,
            "Macro F1": float(
                f1_score(
                    y_true_array,
                    predicted,
                    labels=np.arange(scores.shape[1]),
                    average="macro",
                    zero_division=0,
                )
            ),
            "NF false alarms": int(
                np.sum((y_true_array == normal_class_id) & (predicted != normal_class_id))
            ),
        }

    rows = [summarize("Original argmax", baseline)]
    for threshold in threshold_values:
        adjusted = baseline.copy()
        rejected = (baseline == rnf_class_id) & (scores[:, rnf_class_id] < threshold)
        adjusted[rejected] = alternative[rejected]
        rule_name = f"RNF threshold = {threshold:.2f}"
        predictions[rule_name] = adjusted
        rows.append(summarize(rule_name, adjusted))

    results = pd.DataFrame(rows)
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)

    float_columns = results.select_dtypes(include=["float"]).columns
    formatters = {column: (lambda value: f"{value:.2f}") for column in float_columns}
    print("RNF DECISION THRESHOLD SENSITIVITY ANALYSIS")
    print(results.to_string(index=False, formatters=formatters))
    print(
        "This is a test-set sensitivity analysis. It does not establish an "
        "optimal deployment threshold."
    )
    return results, predictions

