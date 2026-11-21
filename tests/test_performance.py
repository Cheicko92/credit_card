"""Activité 8 : budget de performance (tests lents, hors Pull Request).

Seuils justifiés par le contexte : l'autorisation d'un paiement par carte doit répondre en
quelques dizaines de ms ; on alloue au modèle seul 10 ms au P95 (marge x4 sur la mesure
locale), un modèle < 1 Mo pour un déploiement léger, un débit > 10 000 prédictions/s en lot.
"""

import time
from pathlib import Path

import numpy as np
import pytest

from app.prediction import predict_fraud
from app.training import save_pipeline

pytestmark = pytest.mark.slow


def test_single_prediction_latency_p95(model, threshold, X_test):
    for _ in range(20):
        predict_fraud(model, X_test.iloc[[0]], threshold)
    times = []
    for i in range(200):
        row = X_test.iloc[[i % len(X_test)]]
        t0 = time.perf_counter()
        predict_fraud(model, row, threshold)
        times.append((time.perf_counter() - t0) * 1000)
    assert np.percentile(times, 95) < 10.0


def test_batch_throughput(model, threshold, X_test):
    batch = X_test.iloc[:1000]
    t0 = time.perf_counter()
    for _ in range(10):
        predict_fraud(model, batch, threshold)
    rate = 10 * len(batch) / (time.perf_counter() - t0)
    assert rate > 10_000


def test_saved_model_size(tmp_path, model, threshold):
    path = Path(tmp_path) / "m.joblib"
    save_pipeline(model, threshold, path)
    assert path.stat().st_size < 1_000_000
