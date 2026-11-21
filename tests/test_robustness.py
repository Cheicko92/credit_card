"""Robustesse, sensibilité et sécurité opérationnelle (Activités 5 et 8)."""

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from app.prediction import predict_fraud, predict_proba_fraud
from app.preprocessing import FEATURE_COLUMNS, DataValidationError, validate_data

V_COLS = [f"V{i}" for i in range(1, 29)]
ROOT = Path(__file__).resolve().parents[1]


# --- Entrées imparfaites : rejeter / traiter / alerter -----------------------------
def test_nan_in_features_is_treated_by_imputation(model, X_test):
    """Scénario « traiter » : NaN -> imputation médiane documentée, probabilités valides."""
    X = X_test.copy()
    X.iloc[:20, 5] = np.nan
    p = predict_proba_fraud(model, X)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_all_nan_row_still_gets_valid_probability(model, X_test):
    X = X_test.iloc[:3].copy()
    X.loc[:, FEATURE_COLUMNS] = np.nan
    p = predict_proba_fraud(model, X)
    assert np.isfinite(p).all()


@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_infinite_values_are_rejected(model, X_test, bad):
    X = X_test.copy()
    X.iloc[0, 1] = bad
    with pytest.raises(DataValidationError):
        predict_proba_fraud(model, X)


def test_non_numeric_values_are_rejected(model, X_test):
    X = X_test.copy().astype(object)
    X.iloc[0, 1] = "abc"
    with pytest.raises(DataValidationError):
        predict_proba_fraud(model, X)


def test_invalid_schema_is_rejected(model, X_test):
    with pytest.raises(DataValidationError):
        predict_proba_fraud(model, X_test.drop(columns=["Amount"]))
    with pytest.raises(DataValidationError):
        predict_proba_fraud(model, pd.concat([X_test, X_test[["V1"]]], axis=1))


def test_extreme_amount_gives_valid_probabilities(model, X_test):
    """Contrat de sortie respecté même pour un montant extrême."""
    X = X_test.copy()
    X["Amount"] = 1e9
    p = predict_proba_fraud(model, X)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


@pytest.mark.xfail(
    strict=True,
    reason="ANOMALIE A-03 (ouverte) : un Amount hors plage sature la régression logistique "
    "(avalanche de fausses alertes OU masquage total des fraudes selon le signe du coefficient). "
    "Garde-fou de plausibilité (plafonnement + alerte) à implémenter.",
)
def test_extreme_amount_should_not_change_alert_volume_massively(model, threshold, X_test):
    base_alerts = predict_fraud(model, X_test, threshold)[0].sum()
    extreme_alerts = predict_fraud(model, X_test.assign(Amount=1e7), threshold)[0].sum()
    assert abs(extreme_alerts - base_alerts) <= max(5, 0.5 * base_alerts)


# --- Sensibilité (mesurée, pas d'invariance universelle imposée) -------------------
def test_small_noise_on_v_columns_has_small_effect(model, threshold, X_test):
    rng = np.random.default_rng(0)
    Xp = X_test.copy()
    Xp[V_COLS] = Xp[V_COLS] + rng.normal(0, 0.05, size=(len(Xp), 28))
    p0 = predict_proba_fraud(model, X_test)
    p1 = predict_proba_fraud(model, Xp)
    c0 = predict_fraud(model, X_test, threshold)[0]
    c1 = predict_fraud(model, Xp, threshold)[0]
    assert np.mean(np.abs(p1 - p0)) < 0.02
    assert np.mean(c0 != c1) < 0.02


def test_sensitivity_grows_with_noise_amplitude(model, X_test):
    rng = np.random.default_rng(1)
    base = predict_proba_fraud(model, X_test)
    deltas = []
    for sigma in (0.05, 0.3, 1.0):
        Xp = X_test.copy()
        Xp[V_COLS] = Xp[V_COLS] + rng.normal(0, sigma, size=(len(Xp), 28))
        deltas.append(np.mean(np.abs(predict_proba_fraud(model, Xp) - base)))
    assert deltas[0] < deltas[1] < deltas[2]


def test_distribution_shift_is_measurable_and_detectable(splits):
    """Un décalage de V1 doit être visible via un test de dérive (écart de moyenne standardisé)."""
    X_tr, _, X_te, *_ = splits
    shifted = X_te.assign(V1=X_te["V1"] + 1.0)
    z_same = abs(X_te["V1"].mean() - X_tr["V1"].mean()) / X_tr["V1"].std()
    z_shift = abs(shifted["V1"].mean() - X_tr["V1"].mean()) / X_tr["V1"].std()
    assert z_shift > 0.5 > z_same


@settings(
    max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(
    amount=st.floats(min_value=0, max_value=1e12, allow_nan=False, allow_infinity=False),
    v1=st.floats(min_value=-50, max_value=50, allow_nan=False, allow_infinity=False),
    time=st.floats(min_value=0, max_value=1e7, allow_nan=False, allow_infinity=False),
)
def test_probability_contract_holds_for_generated_inputs(model, X_test, amount, v1, time):
    """Propriété : pour toute entrée valide, 0 <= p <= 1 et p fini."""
    row = X_test.iloc[[0]].copy()
    row["Amount"], row["V1"], row["Time"] = amount, v1, time
    p = predict_proba_fraud(model, row)
    assert np.isfinite(p).all() and 0.0 <= p[0] <= 1.0


@settings(max_examples=30, deadline=None)
@given(st.lists(st.sampled_from(FEATURE_COLUMNS), min_size=1, max_size=5, unique=True))
def test_any_missing_required_column_is_rejected(cols):
    df = pd.DataFrame([[0.0] * 30], columns=FEATURE_COLUMNS).drop(columns=cols)
    with pytest.raises(DataValidationError):
        validate_data(df, require_target=False)


# --- Sécurité opérationnelle -------------------------------------------------------
def test_no_sensitive_values_in_logs(model, X_test, caplog):
    """Aucune donnée d'entrée ne doit apparaître dans les journaux lors d'une prédiction."""
    with caplog.at_level(logging.DEBUG):
        predict_fraud(model, X_test.iloc[:5])
        try:
            predict_proba_fraud(model, X_test.iloc[:2].assign(V1=np.inf))
        except DataValidationError:
            pass
    log_text = caplog.text + " ".join(str(r.getMessage()) for r in caplog.records)
    assert str(X_test.iloc[0]["Amount"]) not in log_text


def test_error_messages_do_not_leak_row_values(model, X_test):
    secret = 123456.789
    X = X_test.iloc[:2].assign(V1=np.inf, Amount=secret)
    with pytest.raises(DataValidationError) as exc:
        predict_proba_fraud(model, X)
    assert str(secret) not in str(exc.value)


SECRET_PATTERNS = [
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----",
    r"(?i)(api[_-]?key|secret|password|token)\s*[:=]\s*['\"][^'\"\s]{8,}['\"]",
    r"ghp_[A-Za-z0-9]{36}",
]


def test_no_secrets_in_repository():
    offenders = []
    for path in ROOT.rglob("*"):
        if (
            not path.is_file()
            or path.suffix not in {".py", ".yml", ".yaml", ".md", ".txt", ".toml", ".json", ".cfg"}
            or any(p in path.parts for p in (".git", ".venv", "__pycache__", ".hypothesis"))
            or path.name == "test_robustness.py"  # contient les motifs de recherche
        ):
            continue
        text = path.read_text(errors="ignore")
        for pat in SECRET_PATTERNS:
            if re.search(pat, text):
                offenders.append((str(path.relative_to(ROOT)), pat))
    assert not offenders, f"Secrets potentiels : {offenders}"


def test_gitignore_excludes_data_and_secrets():
    gi = (ROOT / ".gitignore").read_text()
    assert "data/creditcard.csv" in gi and ".env" in gi
