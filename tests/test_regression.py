"""Non-régression : comparaison à une baseline FIGÉE (tests/baseline.json).

La baseline n'est JAMAIS recalculée par les tests. Elle ne change que par une action
volontaire et revue : `python scripts/run_qualification.py --update-baseline`.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from app.data_source import get_dataset_path
from app.evaluation import evaluate_model
from app.prediction import predict_fraud, predict_proba_fraud
from app.preprocessing import RANDOM_SEED, load_data, make_synthetic_data, split_data
from app.training import train_model

BASELINE_PATH = Path(__file__).parent / "baseline.json"
TOL = {"recall": 0.05, "precision": 0.05, "pr_auc": 0.03, "fpr": 0.002}


@pytest.fixture(scope="module")
def baseline():
    assert BASELINE_PATH.exists(), "baseline.json absente : la référence doit être versionnée"
    return json.loads(BASELINE_PATH.read_text())


@pytest.fixture(scope="module")
def validation_split(baseline):
    n = baseline["dataset"]["n_total"]
    if "synthétique" in baseline["dataset"]["source"]:
        df = make_synthetic_data(n=n)
    else:
        csv = get_dataset_path()
        if csv is None:
            pytest.skip("baseline établie sur le jeu réel : data/creditcard.csv requis")
        df = load_data(csv)
        assert len(df) == n, "le jeu local ne correspond pas à celui de la baseline"
    X_tr, X_val, _, y_tr, y_val, _ = split_data(df, seed=RANDOM_SEED)
    return X_tr, X_val, y_tr, y_val


def _val_metrics(model, thr, X_val, y_val):
    pred, proba = predict_fraud(model, X_val, thr)
    return evaluate_model(y_val, pred, proba)


def test_baseline_records_required_traceability_fields(baseline):
    for key in ("env", "dataset", "model", "threshold", "validation_metrics"):
        assert key in baseline
    for key in ("code_version", "python", "sklearn", "numpy", "pandas", "seed"):
        assert key in baseline["env"]
    for key in ("source", "n_total", "n_fraud", "split"):
        assert key in baseline["dataset"]


def test_no_regression_against_baseline(baseline, validation_split):
    X_tr, X_val, y_tr, y_val = validation_split
    kind = "logreg" if baseline["model"].startswith("Régression") else "tree"
    model = train_model(X_tr, y_tr, kind=kind, seed=baseline["env"]["seed"])
    new = _val_metrics(model, baseline["threshold"], X_val, y_val)
    ref = baseline["validation_metrics"]

    assert new["recall"] >= ref["recall"] - TOL["recall"]
    assert new["precision"] >= ref["precision"] - TOL["precision"]
    assert new["pr_auc"] >= ref["pr_auc"] - TOL["pr_auc"]
    assert new["fpr"] <= ref["fpr"] + TOL["fpr"]


def test_results_match_baseline_exactly_with_same_seed_and_env(baseline, validation_split):
    """Reproductibilité : mêmes données, graine et versions => métriques quasi identiques."""
    X_tr, X_val, y_tr, y_val = validation_split
    model = train_model(X_tr, y_tr, kind="logreg", seed=baseline["env"]["seed"])
    new = _val_metrics(model, baseline["threshold"], X_val, y_val)
    ref = baseline["validation_metrics"]
    for k in ("tp", "fp", "fn", "tn"):
        assert abs(new[k] - ref[k]) <= 1  # tolérance d'1 cas pour variations de BLAS/versions
    assert new["roc_auc"] == pytest.approx(ref["roc_auc"], abs=1e-3)


def test_degraded_model_is_caught_by_the_control(baseline, validation_split):
    """Échec volontaire : un modèle dégradé (arbre de profondeur 1, labels bruités)
    DOIT violer les tolérances — preuve que le contrôle détecte une régression."""
    X_tr, X_val, y_tr, y_val = validation_split
    rng = np.random.default_rng(0)
    y_noisy = y_tr.copy()
    flip = rng.choice(len(y_noisy), size=len(y_noisy) // 5, replace=False)
    y_noisy.iloc[flip] = 1 - y_noisy.iloc[flip]
    bad = train_model(X_tr, y_noisy, kind="tree", max_depth=1)
    new = _val_metrics(bad, baseline["threshold"], X_val, y_val)
    ref = baseline["validation_metrics"]
    degraded = (
        new["recall"] < ref["recall"] - TOL["recall"]
        or new["precision"] < ref["precision"] - TOL["precision"]
        or new["pr_auc"] < ref["pr_auc"] - TOL["pr_auc"]
        or new["fpr"] > ref["fpr"] + TOL["fpr"]
    )
    assert degraded


def test_threshold_is_part_of_the_reference(baseline):
    assert 0.0 <= baseline["threshold"] <= 1.0


def test_baseline_meets_business_requirements_on_validation(baseline):
    """La baseline elle-même doit respecter les exigences fixées avant l'observation du test."""
    ref = baseline["validation_metrics"]
    assert ref["recall"] >= baseline["min_recall_requirement"]
    assert ref["fpr"] <= baseline["max_fpr_requirement"]


def test_probabilities_stable_across_two_trainings(validation_split, baseline):
    X_tr, X_val, y_tr, _ = validation_split
    seed = baseline["env"]["seed"]
    a = predict_proba_fraud(train_model(X_tr, y_tr, seed=seed), X_val)
    b = predict_proba_fraud(train_model(X_tr, y_tr, seed=seed), X_val)
    assert np.allclose(a, b, atol=1e-12)
