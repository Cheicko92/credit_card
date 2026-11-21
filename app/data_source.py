"""Accès au jeu de données Kaggle « Credit Card Fraud Detection » (ULB).

Ordre de résolution (sans réseau d'abord) :
1. variable d'environnement CREDITCARD_CSV ;
2. data/creditcard.csv (ignoré par Git) ;
3. téléchargement via kagglehub, si demandé explicitement (download=True).

Équivalent du code Kaggle :

    import kagglehub
    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
"""

import os
import shutil
from pathlib import Path

KAGGLE_HANDLE = "mlg-ulb/creditcardfraud"
CSV_NAME = "creditcard.csv"
DEFAULT_DATA_DIR = Path(__file__).resolve().parents[1] / "data"


class DataSourceError(RuntimeError):
    """Jeu de données introuvable ou téléchargement impossible."""


def locate_dataset(data_dir=DEFAULT_DATA_DIR):
    """Retourne le chemin du CSV s'il est déjà disponible localement, sinon None."""
    env = os.environ.get("CREDITCARD_CSV")
    if env:
        p = Path(env)
        if not p.is_file():
            raise DataSourceError(f"CREDITCARD_CSV pointe vers un fichier absent : {p}")
        return p
    p = Path(data_dir) / CSV_NAME
    return p if p.is_file() else None


def download_dataset(data_dir=DEFAULT_DATA_DIR, handle=KAGGLE_HANDLE):
    """Télécharge le jeu avec kagglehub et copie creditcard.csv dans data_dir.

    Les identifiants Kaggle ne sont jamais écrits dans le code : kagglehub lit
    KAGGLE_USERNAME / KAGGLE_KEY ou ~/.kaggle/kaggle.json si le jeu l'exige.
    """
    try:
        import kagglehub
    except ImportError as exc:
        raise DataSourceError("kagglehub n'est pas installé : pip install kagglehub") from exc

    try:
        folder = Path(kagglehub.dataset_download(handle))
    except Exception as exc:  # noqa: BLE001 - kagglehub lève des erreurs réseau/auth variées
        raise DataSourceError(f"Téléchargement Kaggle impossible ({type(exc).__name__})") from exc

    found = next(folder.rglob(CSV_NAME), None)
    if found is None:
        raise DataSourceError(f"{CSV_NAME} absent du dossier téléchargé : {folder}")

    dest_dir = Path(data_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / CSV_NAME
    shutil.copyfile(found, dest)
    return dest


def get_dataset_path(data_dir=DEFAULT_DATA_DIR, download=False):
    """Chemin du CSV (local, ou téléchargé si download=True), sinon None."""
    path = locate_dataset(data_dir)
    if path is None and download:
        path = download_dataset(data_dir)
    return path
