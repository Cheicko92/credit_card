# AI-TestGuard

Stratégie de test et qualification d'un modèle de détection de fraude bancaire
(jeu Kaggle *Credit Card Fraud Detection*).

## Installation et exécution

```bash
pip install -r requirements.txt
ruff check .
pytest -m "not slow" --cov=app --cov-report=term-missing   # tests rapides (Pull Request)
pytest -m slow                                              # budget de performance
python scripts/run_qualification.py --download --require-real --update-baseline
```

La dernière commande télécharge le jeu Kaggle avec `kagglehub`
(`kagglehub.dataset_download("mlg-ulb/creditcardfraud")`), relance l'évaluation complète sur les données
réelles et **régénère la baseline** (action volontaire : à relire avant de la versionner).
Sans `--download`, le script utilise `data/creditcard.csv` s'il existe, sinon un jeu **synthétique**
de même schéma (les chiffres ne sont alors pas des performances réelles). Détails : `data/README.md`.

Sous Google Colab : `!pip install -q -r requirements.txt` puis la même commande avec `!python`.

## Structure

- `app/` : accès données Kaggle (kagglehub), validation/prétraitement, entraînement + seuil, prédiction, métriques
- `tests/` : unitaires, prédiction, intégration, robustesse (+ sécurité), non-régression, performance
- `tests/baseline.json` : référence **figée** (ne change que via `--update-baseline`, après revue)
- `reports/` : rapport de qualification, résultats, graphiques, preuves d'exécution
- `.github/workflows/tests.yml` : Ruff + pytest + couverture ; évaluation complète sur demande ou planifiée

## Rapport

`reports/rapport_qualification.md` : plan de test, matrice de traçabilité, résultats, anomalies,
décision et réponse à la question de synthèse.

## Anomalie connue

A-03 : un `Amount` hors plage sature le modèle (test `xfail` strict). Voir le rapport.
