"""Supplementary gradient boosting diagnostics."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
import shap


def _prepare_features(data: pd.DataFrame, feature_cols: Sequence[str]) -> pd.DataFrame:
    X = pd.get_dummies(data[list(feature_cols)], dummy_na=True)
    return X


def gradient_boosting_diagnostics(
    data: pd.DataFrame,
    outcome: str,
    feature_cols: Sequence[str],
    classification: bool = False,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Dict[str, object]:
    X = _prepare_features(data, feature_cols)
    y = data[outcome]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y if classification else None
    )
    if classification:
        model = GradientBoostingClassifier(random_state=random_state)
    else:
        model = GradientBoostingRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    if classification:
        proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, proba)),
        }
    else:
        metrics = {"rmse": float(np.sqrt(mean_squared_error(y_test, y_pred)))}
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    interaction_values = explainer.shap_interaction_values(X_test)
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    shap_summary = (
        pd.DataFrame({"feature": X_test.columns, "mean_abs_shap": mean_abs_shap})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return {
        "model": model,
        "metrics": metrics,
        "shap_values": shap_values,
        "interaction_values": interaction_values,
        "shap_summary": shap_summary,
    }


__all__ = ["gradient_boosting_diagnostics"]
