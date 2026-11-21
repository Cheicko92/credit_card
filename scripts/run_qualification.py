"""Exécute l'évaluation complète et écrit les preuves dans reports/.

Usage : python scripts/run_qualification.py [--download] [--require-real] [--update-baseline]
Jeu réel : data/creditcard.csv, CREDITCARD_CSV ou téléchargement kagglehub (--download) ;
sinon repli sur un jeu synthétique de même schéma (--n lignes).
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import psutil  # noqa: E402
import sklearn  # noqa: E402
from sklearn.metrics import precision_recall_curve, roc_curve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import __version__  # noqa: E402
from app.data_source import get_dataset_path  # noqa: E402
from app.evaluation import evaluate_model  # noqa: E402
from app.prediction import predict_fraud, predict_proba_fraud  # noqa: E402
from app.preprocessing import (  # noqa: E402
    RANDOM_SEED,
    load_data,
    make_synthetic_data,
    split_data,
)
from app.training import save_pipeline, select_threshold, train_model  # noqa: E402

REPORTS = ROOT / "reports"
MIN_RECALL = 0.80  # exigences métier fixées AVANT observation du test
MAX_FPR = 0.01


def wilson(k, n, z=1.96):
    """Intervalle de Wilson à 95 % pour une proportion."""
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return [float(c - h), float(c + h)]


def get_data(n, download=False, require_real=False):
    """Jeu réel (local ou kagglehub) si disponible, sinon synthétique (sauf --require-real)."""
    csv = get_dataset_path(ROOT / "data", download=download)
    if csv is not None:
        return load_data(csv), "creditcard.csv (Kaggle)"
    if require_real:
        raise SystemExit(
            "Jeu Kaggle introuvable : ajoutez --download (kagglehub) ou placez "
            "data/creditcard.csv (ou définissez CREDITCARD_CSV)."
        )
    return make_synthetic_data(n=n), f"synthétique (n={n}, schéma Kaggle)"


def robustness(model, thr, X_test, y_test):
    rng = np.random.default_rng(RANDOM_SEED)
    base_pred, base_proba = predict_fraud(model, X_test, thr)
    base = evaluate_model(y_test, base_pred, base_proba)
    rows = []

    def add(name, Xp, expectation):
        pred, proba = predict_fraud(model, Xp, thr)
        m = evaluate_model(y_test, pred, proba)
        rows.append(
            {
                "scenario": name,
                "attendu": expectation,
                "delta_proba_moy": float(np.mean(np.abs(proba - base_proba))),
                "changement_classe_%": float(np.mean(pred != base_pred) * 100),
                "recall": m["recall"],
                "precision": m["precision"],
                "fpr": m["fpr"],
            }
        )

    v_cols = [f"V{i}" for i in range(1, 29)]
    for sigma in (0.05, 0.1, 0.3):
        Xp = X_test.copy()
        Xp[v_cols] = Xp[v_cols] + rng.normal(0, sigma, size=(len(Xp), 28))
        add(f"Bruit gaussien sigma={sigma} sur V1-V28", Xp, "sensibilité mesurée")
    Xp = X_test.copy()
    Xp["Amount"] = Xp["Amount"] * 1.05
    add("Amount +5 %", Xp, "variation faible")
    Xp = X_test.copy()
    Xp["Amount"] = 1e7
    add("Amount extrême (1e7)", Xp, "traité sans erreur, proba valide")
    Xp = X_test.copy()
    mask = rng.random(Xp[v_cols].shape) < 0.05
    Xp[v_cols] = Xp[v_cols].mask(mask)
    add("5 % de NaN dans V1-V28 (imputation médiane)", Xp, "traité (imputé)")
    Xp = X_test.copy()
    Xp[v_cols] = Xp[v_cols] + 1.0
    add("Décalage de distribution (+1 sur V1-V28)", Xp, "alerte de dérive")

    rejected = {}
    for name, mut in {
        "Valeur infinie": lambda d: d.assign(V1=np.where(d.index == d.index[0], np.inf, d["V1"])),
        "Colonne manquante": lambda d: d.drop(columns=["V5"]),
        "Type texte": lambda d: d.assign(Amount=d["Amount"].astype(str)),
    }.items():
        try:
            predict_fraud(model, mut(X_test.copy()), thr)
            rejected[name] = "NON rejeté (anomalie)"
        except Exception as exc:  # noqa: BLE001
            rejected[name] = f"rejeté ({type(exc).__name__})"
    return base, rows, rejected


def performance(path, model, thr, X_test):
    batch = X_test.iloc[:1000]
    single = X_test.iloc[:1]
    for _ in range(20):
        predict_fraud(model, single, thr)
    lat = []
    for i in range(300):
        row = X_test.iloc[[i % len(X_test)]]
        t0 = time.perf_counter()
        predict_fraud(model, row, thr)
        lat.append((time.perf_counter() - t0) * 1000)
    t0 = time.perf_counter()
    reps = 20
    for _ in range(reps):
        predict_fraud(model, batch, thr)
    dt = time.perf_counter() - t0
    proc = psutil.Process()
    return {
        "latence_unitaire_moy_ms": float(np.mean(lat)),
        "latence_unitaire_p50_ms": float(np.percentile(lat, 50)),
        "latence_unitaire_p95_ms": float(np.percentile(lat, 95)),
        "debit_lot_1000_pred_par_s": float(reps * len(batch) / dt),
        "taille_modele_ko": float(Path(path).stat().st_size / 1024),
        "memoire_rss_mo": float(proc.memory_info().rss / 1e6),
        "protocole": (
            "300 prédictions unitaires après 20 d'échauffement ; 20 lots de 1000 ; "
            "time.perf_counter ; machine unique, 1 processus"
        ),
    }


def drift_psi(X_train, X_new, bins=10):
    """PSI par variable (>0,25 = dérive forte, 0,1-0,25 = modérée)."""
    out = {}
    for c in ("Amount", "V1", "V3", "V10"):
        edges = np.unique(np.quantile(X_train[c], np.linspace(0, 1, bins + 1)))
        e = np.histogram(np.clip(X_train[c], edges[0], edges[-1]), edges)[0] / len(X_train)
        a = np.histogram(np.clip(X_new[c], edges[0], edges[-1]), edges)[0] / len(X_new)
        e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
        out[c] = float(np.sum((a - e) * np.log(a / e)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument(
        "--download", action="store_true", help="télécharger le jeu via kagglehub si absent"
    )
    ap.add_argument(
        "--require-real",
        action="store_true",
        help="échouer plutôt que d'utiliser le jeu synthétique",
    )
    args = ap.parse_args()
    REPORTS.mkdir(exist_ok=True)

    df, source = get_data(args.n, args.download, args.require_real)
    X_tr, X_val, X_te, y_tr, y_val, y_te = split_data(df)
    info = {
        "source": source,
        "n_total": len(df),
        "n_fraud": int(df["Class"].sum()),
        "fraud_rate_%": float(df["Class"].mean() * 100),
        "split": {
            "train": [len(X_tr), int(y_tr.sum())],
            "val": [len(X_val), int(y_val.sum())],
            "test": [len(X_te), int(y_te.sum())],
        },
    }

    results, models, thresholds = {}, {}, {}
    for kind, label in (("logreg", "Régression logistique"), ("tree", "Arbre de décision")):
        m = train_model(X_tr, y_tr, kind=kind)
        thr = select_threshold(y_val, predict_proba_fraud(m, X_val), MIN_RECALL, MAX_FPR)
        models[kind], thresholds[kind] = m, thr
        r = {}
        for part, (X, y) in {
            "train": (X_tr, y_tr),
            "val": (X_val, y_val),
            "test": (X_te, y_te),
        }.items():
            pred, proba = predict_fraud(m, X, thr)
            r[part] = evaluate_model(y, pred, proba)
        # Référence naïve : seuil par défaut 0,5
        pred5, proba5 = predict_fraud(m, X_te, 0.5)
        r["test_seuil_0.5"] = evaluate_model(y_te, pred5, proba5)
        r["threshold"] = thr
        t = r["test"]
        r["test_recall_ci95"] = wilson(t["tp"], t["tp"] + t["fn"])
        r["test_fpr_ci95"] = wilson(t["fp"], t["fp"] + t["tn"])
        results[label] = r

    # Modèle de référence naïf : "jamais de fraude" (illustration de l'accuracy)
    naive = evaluate_model(y_te, np.zeros(len(y_te), int), np.zeros(len(y_te)))

    # Choix du modèle sur la VALIDATION (PR-AUC), jamais sur le test
    best_label = max(results, key=lambda k: results[k]["val"]["pr_auc"])
    best_kind = "logreg" if best_label.startswith("Régression") else "tree"
    model, thr = models[best_kind], thresholds[best_kind]

    # Sauvegarde / rechargement / cohérence
    art = REPORTS / "pipeline.joblib"
    save_pipeline(model, thr, art)
    reloaded = joblib.load(art)["model"]
    max_gap = float(
        np.max(np.abs(predict_proba_fraud(model, X_te) - predict_proba_fraud(reloaded, X_te)))
    )

    # Courbes
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for label, kind in (("Régression logistique", "logreg"), ("Arbre de décision", "tree")):
        p = predict_proba_fraud(models[kind], X_te)
        fpr, tpr, _ = roc_curve(y_te, p)
        pr, rc, _ = precision_recall_curve(y_te, p)
        ax[0].plot(fpr, tpr, label=f"{label} (AUC={results[label]['test']['roc_auc']:.3f})")
        ax[1].plot(rc, pr, label=f"{label} (AP={results[label]['test']['pr_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(xlabel="Taux de faux positifs", ylabel="Rappel", title="Courbe ROC (test)")
    ax[1].axhline(y_te.mean(), color="k", ls="--", lw=0.8, label="Hasard")
    ax[1].set(xlabel="Rappel", ylabel="Précision", title="Courbe précision-rappel (test)")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS / "courbes_roc_pr.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
    for a, label in zip(ax, results, strict=True):
        r = results[label]["test"]
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        a.imshow(np.log1p(cm), cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            a.text(j, i, str(v), ha="center", va="center", fontsize=12)
        a.set_xticks([0, 1], ["Prédit normale", "Prédit fraude"])
        a.set_yticks([0, 1], ["Réel normale", "Réel fraude"])
        a.set_title(f"{label}\nseuil={results[label]['threshold']:.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(REPORTS / "matrices_confusion.png", dpi=130)
    plt.close(fig)

    base, rob_rows, rejected = robustness(model, thr, X_te, y_te)
    perf = performance(art, model, thr, X_te)
    psi_shift = drift_psi(X_tr, X_te.assign(V1=X_te["V1"] + 1.0, Amount=X_te["Amount"] * 3))
    psi_same = drift_psi(X_tr, X_te)

    env = {
        "code_version": __version__,
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "seed": RANDOM_SEED,
    }
    baseline = {
        "env": env,
        "dataset": info,
        "model": best_label,
        "threshold": thr,
        "min_recall_requirement": MIN_RECALL,
        "max_fpr_requirement": MAX_FPR,
        "validation_metrics": results[best_label]["val"],
    }

    out = {
        "info": info,
        "env": env,
        "results": results,
        "naive_never_fraud": naive,
        "selected_model": best_label,
        "reload_max_proba_gap": max_gap,
        "robustness_base": base,
        "robustness": rob_rows,
        "rejection": rejected,
        "performance": perf,
        "psi_shifted": psi_shift,
        "psi_same_distribution": psi_same,
    }
    (REPORTS / "resultats.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))

    bpath = ROOT / "tests" / "baseline.json"
    if args.update_baseline or not bpath.exists():
        bpath.write_text(json.dumps(baseline, indent=2, ensure_ascii=False))
        print("Baseline écrite (action volontaire) :", bpath)
    print(
        json.dumps(
            {k: out[k] for k in ("info", "selected_model", "performance")},
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
