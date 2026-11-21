"""T04 : prédictions, seuil et métriques (tests unitaires et fonctionnels ML)."""

import numpy as np
import pandas as pd
import pytest

from app.evaluation import evaluate_model
from app.prediction import predict_fraud, predict_proba_fraud
from app.preprocessing import DataValidationError
from app.training import build_pipeline, select_threshold, train_model


# --- Contrat de sortie (T04) -------------------------------------------------------
def test_probabilities_are_valid(model, X_test):
    p = predict_proba_fraud(model, X_test)
    assert p.shape == (len(X_test),)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_predictions_are_binary_and_match_threshold(model, X_test):
    pred, proba = predict_fraud(model, X_test, threshold=0.7)
    assert set(np.unique(pred)) <= {0, 1}
    assert np.array_equal(pred, (proba >= 0.7).astype(int))


def test_threshold_is_monotonic(model, X_test):
    """Augmenter le seuil ne peut que réduire le nombre d'alertes."""
    counts = [predict_fraud(model, X_test, t)[0].sum() for t in (0.1, 0.5, 0.9)]
    assert counts[0] >= counts[1] >= counts[2]


def test_extreme_thresholds(model, X_test):
    assert predict_fraud(model, X_test, 0.0)[0].all()  # seuil 0 : tout est signalé
    proba = predict_proba_fraud(model, X_test)
    assert predict_fraud(model, X_test, 1.0)[0].sum() == (proba >= 1.0).sum()


@pytest.mark.parametrize("bad", [-0.1, 1.1])
def test_invalid_threshold_rejected(model, X_test, bad):
    with pytest.raises(ValueError):
        predict_fraud(model, X_test, bad)


def test_single_row_prediction(model, X_test):
    pred, proba = predict_fraud(model, X_test.iloc[[0]])
    assert pred.shape == (1,) and 0 <= proba[0] <= 1


def test_prediction_does_not_depend_on_row_order(model, X_test):
    p = predict_proba_fraud(model, X_test)
    rev = predict_proba_fraud(model, X_test.iloc[::-1])
    assert np.allclose(p, rev[::-1])


def test_prediction_on_invalid_input_is_rejected(model, X_test):
    with pytest.raises(DataValidationError):
        predict_proba_fraud(model, X_test.drop(columns=["V3"]))
    with pytest.raises(DataValidationError):
        predict_proba_fraud(model, X_test.iloc[0:0])
    with pytest.raises(DataValidationError):
        predict_proba_fraud(model, np.zeros((3, 30)))


def test_extra_columns_are_ignored(model, X_test):
    """Politique documentée : colonnes supplémentaires ignorées (pas de fuite vers le modèle)."""
    extra = X_test.assign(Class=0, secret_id="x")
    assert np.allclose(predict_proba_fraud(model, extra), predict_proba_fraud(model, X_test))


# --- Entraînement ------------------------------------------------------------------
def test_train_rejects_single_class(splits):
    X_tr, _, _, y_tr, *_ = splits
    with pytest.raises(DataValidationError, match="deux classes"):
        train_model(X_tr, y_tr * 0)


def test_train_rejects_length_mismatch(splits):
    X_tr, _, _, y_tr, *_ = splits
    with pytest.raises(DataValidationError):
        train_model(X_tr, y_tr.iloc[:-1])


def test_train_rejects_empty(splits):
    X_tr, _, _, y_tr, *_ = splits
    with pytest.raises(DataValidationError):
        train_model(X_tr.iloc[0:0], y_tr.iloc[0:0])


def test_unknown_model_kind():
    with pytest.raises(ValueError):
        build_pipeline("svm")


def test_training_is_deterministic(splits):
    X_tr, _, X_te, y_tr, *_ = splits
    a = predict_proba_fraud(train_model(X_tr, y_tr, seed=5), X_te)
    b = predict_proba_fraud(train_model(X_tr, y_tr, seed=5), X_te)
    assert np.array_equal(a, b)


def test_tree_pipeline_trains(splits):
    X_tr, _, X_te, y_tr, *_ = splits
    p = predict_proba_fraud(train_model(X_tr, y_tr, kind="tree"), X_te)
    assert np.isfinite(p).all()


def test_model_beats_random_on_independent_data(model, X_test, y_test):
    """Garde-fou fonctionnel ML : AUC bien au-dessus du hasard sur le test."""
    m = evaluate_model(y_test, predict_fraud(model, X_test)[0], predict_proba_fraud(model, X_test))
    assert m["roc_auc"] > 0.9


# --- Métriques : valeurs calculées à la main ---------------------------------------
def test_evaluate_model_known_confusion_matrix():
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 1, 0, 1, 0, 0, 0, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.7, 0.3, 0.6, 0.2, 0.1, 0.1, 0.1, 0.1])
    m = evaluate_model(y_true, y_pred, y_proba)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (3, 1, 1, 5)
    assert m["precision"] == pytest.approx(3 / 4)
    assert m["recall"] == pytest.approx(3 / 4)
    assert m["f1"] == pytest.approx(0.75)
    assert m["fpr"] == pytest.approx(1 / 6)
    assert m["fnr"] == pytest.approx(1 / 4)
    assert m["accuracy"] == pytest.approx(0.8)
    # 23 paires (positif, négatif) bien ordonnées sur 24 (le 0,3 est battu par le 0,6)
    assert m["roc_auc"] == pytest.approx(23 / 24)


def test_fpr_and_recall_are_complements():
    y = np.array([1, 0, 1, 0, 0, 1])
    m = evaluate_model(y, np.array([1, 0, 0, 1, 0, 1]), np.array([0.9, 0.1, 0.4, 0.6, 0.2, 0.8]))
    assert m["recall"] + m["fnr"] == pytest.approx(1.0)


def test_accuracy_is_misleading_on_imbalanced_data():
    """Un modèle « jamais de fraude » a une accuracy de 99,8 % et un rappel nul."""
    y = np.zeros(10000, dtype=int)
    y[:20] = 1
    m = evaluate_model(y, np.zeros(10000, dtype=int), np.zeros(10000))
    assert m["accuracy"] == pytest.approx(0.998)
    assert m["recall"] == 0.0 and m["precision"] == 0.0 and m["fnr"] == 1.0


def test_evaluate_model_perfect_classifier():
    y = np.array([0, 0, 1, 1])
    m = evaluate_model(y, y, y.astype(float))
    assert m["f1"] == 1.0 and m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0 and m["fpr"] == 0.0


def test_evaluate_model_single_class_has_nan_auc_not_crash():
    m = evaluate_model(np.array([0, 0, 0]), np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]))
    assert np.isnan(m["roc_auc"]) and m["fp"] == 0


def test_evaluate_model_length_mismatch():
    with pytest.raises(ValueError):
        evaluate_model([0, 1], [0], [0.1, 0.9])


# --- Choix du seuil (sur validation) -----------------------------------------------
def test_select_threshold_respects_constraints():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(100), np.zeros(9900)].astype(int)
    p = np.r_[rng.beta(8, 2, 100), rng.beta(1, 12, 9900)]
    thr = select_threshold(y, p, min_recall=0.8, max_fpr=0.01)
    pred = (p >= thr).astype(int)
    m = evaluate_model(y, pred, p)
    assert m["recall"] >= 0.8 and m["fpr"] <= 0.01


def test_select_threshold_fallback_when_unreachable():
    y = np.array([1, 0, 1, 0, 0, 0] * 20)
    p = np.tile([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], 20)  # aucun pouvoir discriminant
    thr = select_threshold(y, p, min_recall=0.99, max_fpr=0.0)
    assert 0.0 <= thr <= 1.0


def test_select_threshold_uses_only_given_data_types():
    pd_y = pd.Series([0, 0, 0, 1, 1, 0, 1, 0])
    thr = select_threshold(pd_y, [0.1, 0.2, 0.3, 0.9, 0.8, 0.2, 0.7, 0.1], 0.6, 0.5)
    assert 0.0 <= thr <= 1.0
