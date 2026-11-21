"""Accès aux données Kaggle : tests hors réseau (kagglehub simulé)."""

import sys
import types

import pytest

from app.data_source import (
    CSV_NAME,
    DataSourceError,
    download_dataset,
    get_dataset_path,
    locate_dataset,
)


@pytest.fixture(autouse=True)
def _no_env(monkeypatch):
    monkeypatch.delenv("CREDITCARD_CSV", raising=False)


def _fake_kagglehub(monkeypatch, folder=None, error=None):
    calls = []
    mod = types.ModuleType("kagglehub")

    def dataset_download(handle):
        calls.append(handle)
        if error:
            raise error
        return str(folder)

    mod.dataset_download = dataset_download
    monkeypatch.setitem(sys.modules, "kagglehub", mod)
    return calls


def test_locate_returns_none_when_absent(tmp_path):
    assert locate_dataset(tmp_path) is None


def test_locate_finds_local_csv(tmp_path):
    (tmp_path / CSV_NAME).write_text("x")
    assert locate_dataset(tmp_path) == tmp_path / CSV_NAME


def test_env_variable_takes_precedence(tmp_path, monkeypatch):
    other = tmp_path / "ailleurs.csv"
    other.write_text("x")
    (tmp_path / CSV_NAME).write_text("y")
    monkeypatch.setenv("CREDITCARD_CSV", str(other))
    assert locate_dataset(tmp_path) == other


def test_env_variable_with_missing_file_is_an_explicit_error(tmp_path, monkeypatch):
    monkeypatch.setenv("CREDITCARD_CSV", str(tmp_path / "absent.csv"))
    with pytest.raises(DataSourceError, match="absent"):
        locate_dataset(tmp_path)


def test_download_copies_csv_into_data_dir(tmp_path, monkeypatch):
    cache = tmp_path / "cache" / "v3"
    cache.mkdir(parents=True)
    (cache / CSV_NAME).write_text("Time,Class\n0,0\n")
    calls = _fake_kagglehub(monkeypatch, folder=cache)

    dest = download_dataset(tmp_path / "data")

    assert calls == ["mlg-ulb/creditcardfraud"]
    assert dest == tmp_path / "data" / CSV_NAME
    assert dest.read_text() == "Time,Class\n0,0\n"


def test_download_error_is_wrapped(tmp_path, monkeypatch):
    _fake_kagglehub(monkeypatch, error=ConnectionError("réseau coupé"))
    with pytest.raises(DataSourceError, match="impossible"):
        download_dataset(tmp_path)


def test_download_without_csv_in_folder(tmp_path, monkeypatch):
    empty = tmp_path / "vide"
    empty.mkdir()
    _fake_kagglehub(monkeypatch, folder=empty)
    with pytest.raises(DataSourceError, match="absent"):
        download_dataset(tmp_path / "data")


def test_download_when_kagglehub_not_installed(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "kagglehub", None)  # force ImportError
    with pytest.raises(DataSourceError, match="pip install kagglehub"):
        download_dataset(tmp_path)


def test_get_dataset_path_does_not_download_unless_asked(tmp_path, monkeypatch):
    calls = _fake_kagglehub(monkeypatch, folder=tmp_path)
    assert get_dataset_path(tmp_path, download=False) is None
    assert calls == []


def test_get_dataset_path_skips_download_if_local_file_exists(tmp_path, monkeypatch):
    (tmp_path / CSV_NAME).write_text("x")
    calls = _fake_kagglehub(monkeypatch, folder=tmp_path)
    assert get_dataset_path(tmp_path, download=True) == tmp_path / CSV_NAME
    assert calls == []
