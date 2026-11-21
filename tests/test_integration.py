"""T05, T06 : pipeline de bout en bout et prévention des fuites de données."""

import numpy as np
import pandas as pd
import pytest

from app.evaluation import evaluate_model
from app.prediction import predict_fraud, predict_proba_fraud
from app.preprocessing import (
    FEATURE_COLUMNS,
    DataValidationError,
    load_data,
    preprocess_data,
    split_data,
)
from app.training import load_pipeline, save_pipeline, select_threshold, train_model

TOL = 1e-12


def test_end_to_end_pipeline(tmp_path, small_df):
    """CSV -> validation -> split -> entraînement -> seuil (val) -> sauvegarde -> rechargement
    -> prédiction test -> métriques."""
    csv = tmp_path / "data.csv"
    small_df.to_csv(csv, index=False)

    df = load_data(csv)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(df, seed=7)
    model = train_model(X_tr, y_tr, seed=7)
    thr = select_threshold(y_val, predict_proba_fraud(model, X_val))

    path = tmp_path / "pipeline.joblib"
    save_pipeline(model, thr, path)
    model2, thr2 = load_pipeline(path)

    pred, proba = predict_fraud(model2, X_te, thr2)
    metrics = evaluate_model(y_te, pred, proba)

    assert thr2 == thr
    assert pred.shape == (len(X_te),)
    assert metrics["tp"] + metrics["fn"] == int(y_te.sum())
    assert metrics["tn"] + metrics["fp"] == int((y_te == 0).sum())
    assert metrics["roc_auc"] > 0.9


def test_predictions_identical_before_and_after_reload(tmp_path, model, threshold, X_test):
    path = tmp_path / "p.joblib"
    save_pipeline(model, threshold, path)
    model2, thr2 = load_pipeline(path)
    p1 = predict_proba_fraud(model, X_test)
    p2 = predict_proba_fraud(model2, X_test)
    assert np.max(np.abs(p1 - p2)) <= TOL
    assert np.array_equal(
        predict_fraud(model, X_test, threshold)[0], predict_fraud(model2, X_test, thr2)[0]
    )


def test_dimensions_are_coherent_through_the_pipeline(model, X_test):
    n_features = len(FEATURE_COLUMNS)
    assert X_test.shape[1] == n_features
    assert model.named_steps["imputer"].transform(X_test).shape == (len(X_test), n_features)
    assert model.named_steps["scaler"].n_features_in_ == n_features
    assert predict_proba_fraud(model, X_test).shape == (len(X_test),)


def test_no_non_finite_values_after_preprocessing_and_scaling(model, X_test):
    X, _ = preprocess_data(X_test.assign(Class=0))
    Z = model[:-1].transform(X)
    assert np.isfinite(Z).all()


# --- Prévention des fuites ---------------------------------------------------------
def test_scaler_is_fitted_on_train_only(splits, model):
    X_tr, _, X_te, *_ = splits
    scaler = model.named_steps["scaler"]
    assert np.allclose(scaler.mean_, X_tr.to_numpy().mean(axis=0))
    assert not np.allclose(scaler.mean_, X_te.to_numpy().mean(axis=0))


def test_imputer_statistics_come_from_train_only(splits, model):
    X_tr, _, X_te, *_ = splits
    stats = model.named_steps["imputer"].statistics_
    assert np.allclose(stats, X_tr.median().to_numpy())
    assert not np.allclose(stats, X_te.median().to_numpy())


def test_predicting_does_not_refit_the_scaler(model, X_test):
    before = model.named_steps["scaler"].mean_.copy()
    predict_proba_fraud(model, X_test)
    predict_proba_fraud(model, X_test.iloc[:5])
    assert np.array_equal(before, model.named_steps["scaler"].mean_)


def test_train_val_test_have_no_common_rows(splits):
    X_tr, X_val, X_te, *_ = splits
    assert not set(X_tr.index) & set(X_te.index)
    assert not set(X_val.index) & set(X_te.index)
    assert not set(X_tr.index) & set(X_val.index)


def test_target_never_present_in_features(splits):
    for X in splits[:3]:
        assert "Class" not in X.columns


def test_model_trained_on_test_would_be_detected_as_leak(splits):
    """Contrôle négatif : un modèle ajusté sur le test mémorise son contenu (surapprentissage
    visible). La différence avec le modèle propre prouve que le test détecte bien une fuite."""
    X_tr, _, X_te, y_tr, _, y_te = splits
    clean = train_model(X_tr, y_tr, kind="tree", max_depth=None, min_samples_leaf=1)
    leaky = train_model(X_te, y_te, kind="tree", max_depth=None, min_samples_leaf=1)
    pc = evaluate_model(y_te, predict_fraud(clean, X_te, 0.5)[0], predict_proba_fraud(clean, X_te))
    pl = evaluate_model(y_te, predict_fraud(leaky, X_te, 0.5)[0], predict_proba_fraud(leaky, X_te))
    assert pl["pr_auc"] > 0.99 > pc["pr_auc"]


# --- Reproductibilité (T06) --------------------------------------------------------
def test_full_run_is_reproducible(small_df):
    def run():
        X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(small_df, seed=21)
        m = train_model(X_tr, y_tr, seed=21)
        thr = select_threshold(y_val, predict_proba_fraud(m, X_val))
        pred, proba = predict_fraud(m, X_te, thr)
        return thr, evaluate_model(y_te, pred, proba)

    (t1, m1), (t2, m2) = run(), run()
    assert t1 == t2
    assert m1 == m2


# --- Gestion explicite des erreurs -------------------------------------------------
def test_load_pipeline_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_pipeline(tmp_path / "absent.joblib")


def test_load_pipeline_wrong_content(tmp_path):
    import joblib

    p = tmp_path / "bad.joblib"
    joblib.dump({"foo": 1}, p)
    with pytest.raises(DataValidationError, match="invalide"):
        load_pipeline(p)


def test_pipeline_rejects_corrupted_batch_at_prediction(model, X_test):
    bad = X_test.copy()
    bad.iloc[0, 3] = np.inf
    with pytest.raises(DataValidationError):
        predict_proba_fraud(model, bad)


def test_loading_csv_with_missing_columns_fails_cleanly(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"Time": [1.0], "Amount": [2.0]}).to_csv(p, index=False)
    with pytest.raises(DataValidationError, match="manquantes"):
        preprocess_data(load_data(p))
