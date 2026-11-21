"""Chargement, validation et préparation des données.

Contrat de données
------------------
- Colonnes requises : Time, V1..V28, Amount (+ Class pour les données étiquetées).
- Aucune colonne dupliquée, aucune colonne indispensable manquante.
- Toutes les variables sont numériques ; Class (si présente) vaut 0 ou 1.
- Politique : une donnée invalide est REJETÉE (DataValidationError).
  Seule exception documentée : les NaN des variables explicatives sont imputés
  par la médiane apprise sur l'entraînement (SimpleImputer dans le Pipeline).
  Les valeurs infinies, elles, sont toujours rejetées.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"
RANDOM_SEED = 42


class DataValidationError(ValueError):
    """Levée lorsque les données ne respectent pas le contrat."""


def load_data(path):
    """Charge un CSV ; erreurs explicites si le fichier est absent ou vide."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Fichier vide : {path}") from exc
    return df


def validate_data(df, require_target=True, allow_missing=False):
    """Contrôle le schéma, les types et les valeurs. Retourne df si valide."""
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Entrée attendue : pandas.DataFrame")
    if df.empty:
        raise DataValidationError("Jeu de données vide")

    duplicated = df.columns[df.columns.duplicated()].tolist()
    if duplicated:
        raise DataValidationError(f"Colonnes dupliquées : {duplicated}")

    required = FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise DataValidationError(f"Colonnes manquantes : {missing}")

    cols = required
    non_numeric = [c for c in cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise DataValidationError(f"Colonnes non numériques : {non_numeric}")

    values = df[cols].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise DataValidationError("Valeurs infinies détectées")
    if not allow_missing and np.isnan(df[FEATURE_COLUMNS].to_numpy(dtype=float)).any():
        raise DataValidationError("Valeurs manquantes dans les variables")

    if require_target:
        if df[TARGET_COLUMN].isna().any():
            raise DataValidationError("Valeurs manquantes dans la cible")
        if not set(df[TARGET_COLUMN].unique()) <= {0, 1}:
            raise DataValidationError("La cible doit contenir uniquement 0 et 1")

    if (df["Amount"] < 0).any():
        raise DataValidationError("Montants négatifs interdits")
    return df


def preprocess_data(df, require_target=True, allow_missing=False):
    """Valide puis sépare X (colonnes dans l'ordre du contrat) et y.

    N'apprend AUCUNE statistique : la mise à l'échelle et l'imputation sont
    dans le Pipeline, ajustées sur l'entraînement uniquement (pas de fuite).
    """
    validate_data(df, require_target=require_target, allow_missing=allow_missing)
    X = df[FEATURE_COLUMNS].astype(float).copy()
    y = df[TARGET_COLUMN].astype(int).copy() if require_target else None
    return X, y


def split_data(df, seed=RANDOM_SEED, val_size=0.2, test_size=0.2):
    """Découpage stratifié train / validation / test, AVANT toute transformation."""
    X, y = preprocess_data(df)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def make_synthetic_data(n=20000, fraud_rate=0.0017, seed=RANDOM_SEED):
    """Jeu synthétique au schéma du jeu Kaggle (utilisé si le CSV est absent).

    Les fraudes sont décalées sur quelques composantes, avec recouvrement
    partiel pour que le problème reste difficile. Ce n'est PAS un substitut
    des données réelles.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(int(round(n * fraud_rate)), 10)
    y = np.zeros(n, dtype=int)
    y[rng.choice(n, n_fraud, replace=False)] = 1

    V = rng.normal(0.0, 1.0, size=(n, 28))
    shifts = {0: -2.2, 2: -2.0, 3: 1.8, 9: -1.9, 11: -2.1, 13: -2.3, 16: -1.6}
    for j, s in shifts.items():
        V[y == 1, j] += s * rng.uniform(0.5, 1.1, size=n_fraud)
    amount = rng.lognormal(mean=3.2, sigma=1.3, size=n)
    amount[y == 1] = rng.lognormal(mean=3.6, sigma=1.5, size=n_fraud)
    time = np.sort(rng.uniform(0, 172792, size=n))

    df = pd.DataFrame(V, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", time)
    df["Amount"] = np.round(amount, 2)
    df["Class"] = y
    return df
