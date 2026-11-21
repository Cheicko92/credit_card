# Données

Jeu **Credit Card Fraud Detection** (Kaggle, ULB) : <https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud>.
Fichier volumineux (~150 Mo) soumis aux conditions de sa source : il n'est **pas versionné**
(`data/creditcard.csv` est dans `.gitignore`).

## Obtenir les données (kagglehub)

```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
print("Path to dataset files:", path)
```

Ou, depuis le dépôt (copie `creditcard.csv` dans `data/`) :

```bash
pip install -r requirements.txt
python scripts/run_qualification.py --download --require-real
```

Ordre de résolution du code (`app/data_source.py`) : variable `CREDITCARD_CSV` → `data/creditcard.csv`
→ téléchargement kagglehub (uniquement avec `--download`). Aucun identifiant n'est écrit dans le code :
si Kaggle l'exige, définir `KAGGLE_USERNAME` et `KAGGLE_KEY` (variables d'environnement ou secrets GitHub).

## Sans données réelles

Sans fichier ni `--download`, `scripts/run_qualification.py` utilise un jeu **synthétique** de même
schéma (`app.preprocessing.make_synthetic_data`). Les chiffres ne sont alors **pas** des performances
sur données réelles. Avec `--require-real`, le script échoue plutôt que de se rabattre dessus.
