from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import os
import csv

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)


def evaluate_classification(
    model,
    X_test,
    y_test,
    include_cm: bool = False,
    include_roc_auc: bool = False,
    average: str = "binary",
    return_dict: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a classification model with multiple metrics.

    Parameters
    ----------
    model : fitted model
    X_test : features
    y_test : true labels
    average : str
        'binary', 'macro', 'micro', 'weighted'
    return_dict : bool
        If True, returns metrics as dict
    verbose : bool
        If True, prints report

    Returns
    -------
    dict with metrics
    """

    y_pred = model.predict(X_test)

    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average=average, zero_division=0),
        "recall": recall_score(y_test, y_pred, average=average, zero_division=0),
        "f1": f1_score(y_test, y_pred, average=average, zero_division=0),
    }

    if include_roc_auc:
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                results["roc_auc"] = roc_auc_score(y_test, y_proba)
            except Exception:
                results["roc_auc"] = None

    if include_cm:
        results["confusion_matrix"] = confusion_matrix(y_test, y_pred)

    if verbose:
        print("\n=== Classification Report ===")
        print(classification_report(y_test, y_pred))
        if include_cm:
            ("Confusion Matrix:\n", results["confusion_matrix"])

    return results if return_dict else None


def save_evaluation_metrics(file_to_csv: str, model_name: str, results: dict) -> None:
    file_exists = os.path.exists(file_to_csv)

    with open(file_to_csv, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["model_name"] + list(results.keys()))

        writer.writerow([model_name] + list(results.values()))