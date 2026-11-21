import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.prediction import predict_proba_fraud  # noqa: E402
from app.preprocessing import make_synthetic_data, split_data  # noqa: E402
from app.training import select_threshold, train_model  # noqa: E402


@pytest.fixture(scope="session")
def small_df():
    """Jeu synthétique rapide, avec assez de fraudes pour stratifier (2 %)."""
    return make_synthetic_data(n=6000, fraud_rate=0.02, seed=7)


@pytest.fixture(scope="session")
def splits(small_df):
    return split_data(small_df, seed=7)


@pytest.fixture(scope="session")
def trained(splits):
    X_tr, X_val, X_te, y_tr, y_val, y_te = splits
    model = train_model(X_tr, y_tr, kind="logreg", seed=7)
    thr = select_threshold(y_val, predict_proba_fraud(model, X_val))
    return model, thr


@pytest.fixture()
def model(trained):
    return trained[0]


@pytest.fixture()
def threshold(trained):
    return trained[1]


@pytest.fixture()
def X_test(splits):
    return splits[2]


@pytest.fixture()
def y_test(splits):
    return splits[5]
