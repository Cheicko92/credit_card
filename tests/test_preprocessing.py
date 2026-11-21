"""T01-T03 : contrat de données, prétraitement, découpage (tests unitaires)."""

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from app.preprocessing import (
    FEATURE_COLUMNS,
    DataValidationError,
    load_data,
    make_synthetic_data,
    preprocess_data,
    split_data,
    validate_data,
)


@pytest.fixture()
def df():
    return make_synthetic_data(n=500, fraud_rate=0.05, seed=1)


# --- Contrat de schéma (T01, T02) -------------------------------------------------
def test_valid_data_is_accepted(df):
    assert validate_data(df) is df


@pytest.mark.parametrize("col", ["Time", "V1", "V28", "Amount", "Class"])
def test_missing_required_column_is_rejected(df, col):
    with pytest.raises(DataValidationError, match="manquantes"):
        validate_data(df.drop(columns=[col]))


def test_missing_target_is_fine_when_not_required(df):
    X, y = preprocess_data(df.drop(columns=["Class"]), require_target=False)
    assert y is None and list(X.columns) == FEATURE_COLUMNS


def test_duplicated_column_is_rejected(df):
    dup = pd.concat([df, df[["V1"]]], axis=1)
    with pytest.raises(DataValidationError, match="dupliqu"):
        validate_data(dup)


def test_empty_dataframe_is_rejected(df):
    with pytest.raises(DataValidationError, match="vide"):
        validate_data(df.iloc[0:0])


def test_non_dataframe_is_rejected():
    with pytest.raises(DataValidationError):
        validate_data([[1, 2, 3]])


def test_wrong_type_is_rejected(df):
    bad = df.assign(Amount=df["Amount"].astype(str))
    with pytest.raises(DataValidationError, match="non numériques"):
        validate_data(bad)


@pytest.mark.parametrize("bad_value", [2, -1, 0.5])
def test_target_other_than_0_1_is_rejected(df, bad_value):
    df = df.copy()
    df["Class"] = df["Class"].astype(float)
    df.loc[df.index[0], "Class"] = bad_value
    with pytest.raises(DataValidationError, match="0 et 1"):
        validate_data(df)


def test_missing_values_rejected_by_default_but_tolerated_on_request(df):
    df = df.copy()
    df.loc[df.index[3], "V5"] = np.nan
    with pytest.raises(DataValidationError, match="manquantes"):
        validate_data(df)
    assert validate_data(df, allow_missing=True) is df


@pytest.mark.parametrize("inf", [np.inf, -np.inf])
def test_infinite_values_always_rejected(df, inf):
    df = df.copy()
    df.loc[df.index[0], "V2"] = inf
    with pytest.raises(DataValidationError, match="infinies"):
        validate_data(df)
    with pytest.raises(DataValidationError, match="infinies"):
        validate_data(df, allow_missing=True)


def test_negative_amount_is_rejected(df):
    df = df.copy()
    df.loc[df.index[0], "Amount"] = -1.0
    with pytest.raises(DataValidationError, match="négatifs"):
        validate_data(df)


# --- Valeurs limites -------------------------------------------------------------
@pytest.mark.parametrize("amount", [0.0, 1e-9, 1e9])
def test_boundary_amounts_are_accepted_and_finite(df, amount):
    df = df.copy()
    df.loc[df.index[0], "Amount"] = amount
    X, _ = preprocess_data(df)
    assert np.isfinite(X.to_numpy()).all()


def test_single_observation_is_processed(df):
    X, y = preprocess_data(df.iloc[[0]])
    assert X.shape == (1, len(FEATURE_COLUMNS)) and len(y) == 1


# --- Prétraitement (T03) ---------------------------------------------------------
def test_preprocess_output_is_finite_and_ordered(df):
    shuffled = df[df.columns[::-1]]  # colonnes dans le désordre
    X, y = preprocess_data(shuffled)
    assert list(X.columns) == FEATURE_COLUMNS
    assert np.isfinite(X.to_numpy()).all()
    assert X.shape == (len(df), 30) and y.shape == (len(df),)
    assert set(y.unique()) <= {0, 1}


def test_preprocess_does_not_modify_input(df):
    before = df.copy(deep=True)
    preprocess_data(df)
    pd.testing.assert_frame_equal(df, before)


def test_standard_scaler_example():
    data = pd.DataFrame({"Amount": [100.0, 200.0, 300.0]})
    scaled = StandardScaler().fit_transform(data)
    assert np.isfinite(scaled).all()
    assert abs(scaled.mean()) < 1e-10
    assert np.isclose(scaled.std(), 1.0)


def test_scaler_on_constant_column_stays_finite():
    scaled = StandardScaler().fit_transform(pd.DataFrame({"Amount": [5.0, 5.0, 5.0]}))
    assert np.isfinite(scaled).all()  # variance nulle gérée par sklearn (échelle = 1)


# --- Découpage -------------------------------------------------------------------
def test_split_is_disjoint_complete_and_stratified(small_df):
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(small_df, seed=3)
    idx = [set(X.index) for X in (X_tr, X_val, X_te)]
    assert not (idx[0] & idx[1]) and not (idx[0] & idx[2]) and not (idx[1] & idx[2])
    assert sum(len(i) for i in idx) == len(small_df)
    overall = small_df["Class"].mean()
    for y in (y_tr, y_val, y_te):
        assert abs(y.mean() - overall) < 0.005  # stratification conservée


def test_split_proportions_60_20_20(small_df):
    X_tr, X_val, X_te, *_ = split_data(small_df)
    n = len(small_df)
    assert abs(len(X_tr) / n - 0.6) < 0.01
    assert abs(len(X_val) / n - 0.2) < 0.01
    assert abs(len(X_te) / n - 0.2) < 0.01


def test_split_is_reproducible_with_same_seed(small_df):
    a = split_data(small_df, seed=11)
    b = split_data(small_df, seed=11)
    assert list(a[0].index) == list(b[0].index)
    assert list(a[2].index) == list(b[2].index)


def test_split_changes_with_other_seed(small_df):
    a = split_data(small_df, seed=1)
    b = split_data(small_df, seed=2)
    assert list(a[2].index) != list(b[2].index)


# --- Chargement ------------------------------------------------------------------
def test_load_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_data(tmp_path / "absent.csv")


def test_load_empty_file_raises(tmp_path):
    p = tmp_path / "vide.csv"
    p.write_text("")
    with pytest.raises(DataValidationError, match="vide"):
        load_data(p)


def test_load_roundtrip(tmp_path, df):
    p = tmp_path / "d.csv"
    df.to_csv(p, index=False)
    back = load_data(p)
    assert back.shape == df.shape
    validate_data(back)


def test_synthetic_data_has_expected_schema_and_rate():
    d = make_synthetic_data(n=10000, fraud_rate=0.0017, seed=0)
    assert list(d.columns) == FEATURE_COLUMNS + ["Class"]
    assert d["Class"].sum() == round(10000 * 0.0017)
