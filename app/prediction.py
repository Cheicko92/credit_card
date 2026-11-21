"""Prédiction avec contrôle d'entrée."""

import numpy as np
import pandas as pd

from app.preprocessing import FEATURE_COLUMNS, DataValidationError, validate_data

DEFAULT_THRESHOLD = 0.5


def predict_proba_fraud(model, X):
    """Probabilité de fraude (colonne 1) pour chaque ligne de X."""
    X = _check_input(X)
    proba = model.predict_proba(X[FEATURE_COLUMNS])[:, 1]
    if not np.isfinite(proba).all() or (proba < 0).any() or (proba > 1).any():
        raise RuntimeError("Probabilités invalides produites par le modèle")
    return proba


def predict_fraud(model, X, threshold=DEFAULT_THRESHOLD):
    """Retourne (classes 0/1, probabilités) selon le seuil donné."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Le seuil doit appartenir à [0, 1]")
    proba = predict_proba_fraud(model, X)
    return (proba >= threshold).astype(int), proba


def _check_input(X):
    """Rejette schéma invalide / infinis ; tolère les NaN (imputés par le pipeline)."""
    if not isinstance(X, pd.DataFrame):
        raise DataValidationError("Entrée attendue : pandas.DataFrame")
    validate_data(X, require_target=False, allow_missing=True)
    return X
