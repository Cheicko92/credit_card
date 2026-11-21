"""Construction, entraînement, choix du seuil et sauvegarde du pipeline."""

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from app.preprocessing import FEATURE_COLUMNS, RANDOM_SEED, DataValidationError


def build_pipeline(kind="logreg", seed=RANDOM_SEED, **params):
    """Pipeline = imputation médiane -> standardisation -> classifieur."""
    if kind == "logreg":
        clf = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
            **params,
        )
    elif kind == "tree":
        defaults = {"max_depth": 6, "min_samples_leaf": 20}
        defaults.update(params)
        clf = DecisionTreeClassifier(class_weight="balanced", random_state=seed, **defaults)
    else:
        raise ValueError(f"Modèle inconnu : {kind}")
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def train_model(X, y, kind="logreg", seed=RANDOM_SEED, **params):
    """Ajuste le pipeline sur les SEULES données fournies (entraînement)."""
    if len(X) != len(y):
        raise DataValidationError("X et y n'ont pas la même longueur")
    if len(X) == 0:
        raise DataValidationError("Entraînement impossible sur données vides")
    if len(set(np.asarray(y).tolist())) < 2:
        raise DataValidationError("La cible doit contenir les deux classes")
    model = build_pipeline(kind, seed=seed, **params)
    model.fit(X[FEATURE_COLUMNS], y)
    return model


def select_threshold(y_val, proba_val, min_recall=0.80, max_fpr=0.01):
    """Seuil choisi sur la VALIDATION uniquement.

    Contraintes métier : rappel >= min_recall ET taux de faux positifs <= max_fpr.
    Parmi les seuils admissibles, on retient la meilleure précision.
    Si aucun seuil n'est admissible, repli documenté : seuil de F1 maximal.
    """
    y_val = np.asarray(y_val)
    precision, recall, thresholds = precision_recall_curve(y_val, proba_val)
    precision, recall = precision[:-1], recall[:-1]
    n_pos = max(int(y_val.sum()), 1)
    n_neg = max(int((y_val == 0).sum()), 1)
    tp = recall * n_pos
    fp = np.where(precision > 0, tp / np.clip(precision, 1e-12, None) - tp, n_neg)
    fpr = fp / n_neg
    ok = (recall >= min_recall) & (fpr <= max_fpr)
    if ok.any():
        idx = int(np.argmax(np.where(ok, precision, -1.0)))
    else:
        f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
        idx = int(np.argmax(f1))
    return float(thresholds[idx])


def save_pipeline(model, threshold, path):
    """Sauvegarde le modèle et son seuil dans un même artefact."""
    joblib.dump({"model": model, "threshold": float(threshold)}, path)


def load_pipeline(path):
    """Recharge l'artefact ; erreurs explicites si absent ou mal formé."""
    try:
        artefact = joblib.load(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Pipeline introuvable : {path}") from exc
    if not isinstance(artefact, dict) or {"model", "threshold"} - artefact.keys():
        raise DataValidationError("Artefact de pipeline invalide")
    return artefact["model"], artefact["threshold"]
