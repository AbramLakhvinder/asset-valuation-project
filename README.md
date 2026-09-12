# Blue Book for Bulldozers

### Estimation du prix des engins de chantier d'occasion

`Python 3.12` · `pandas` · `scikit-learn` · `CatBoost` · `unittest` · `JupyterLab` · `Git`

Ce projet de data science prédit le prix de vente aux enchères d'engins de chantier d'occasion à partir de leurs caractéristiques et de la date de vente, sur la base de la compétition Kaggle [« Blue Book for Bulldozers »](https://www.kaggle.com/competitions/bluebook-for-bulldozers/data).

**État actuel du dépôt :** l'étape de préparation des données (notebook 01) et son industrialisation (`src/preparation_03.py`, `src/fonctions_preparation_03.py`) sont poussées et testées. Les étapes suivantes (expériences de modélisation, comparaison global/segmenté, résultats finaux) sont en cours de revue et seront ajoutées au fil des prochains commits.

## Préparation des données

La préparation construit les variables utilisées par les modèles à partir du CSV brut : traitement des identifiants (`datasource`, `auctioneerID`), extraction de spécifications techniques, création de variables de date, d'âge et d'heures par an, regroupement des catégories rares, et imputation des valeurs manquantes ou hors bornes.

Les années de fabrication antérieures à 1950 ou postérieures à la vente sont rendues manquantes, puis imputées ; les ventes correspondantes sont conservées. Les heures nulles ou hors de l'intervalle ]0 ; 40 000] sont également rendues manquantes avant imputation. La division par un âge nul est évitée. Les **médianes, fréquences et regroupements de catégories sont appris uniquement sur l'apprentissage de chaque découpage**, puis appliqués aux ventes à prédire : c'est le mécanisme qui empêche toute fuite entre apprentissage et validation.

Deux traitements catégoriels sont comparés :

- **A** : les valeurs manquantes deviennent la modalité `Missing`.
- **B** : étend A en distinguant, via `variable_nan_structurel()`, les absences structurelles concentrées dans certains `ProductGroup` (encodées `Absent dans ce segment`).

`src/preparation_03.py` industrialise ces traitements dans `Preparation03`, une classe `BaseEstimator`/`TransformerMixin` scikit-learn : les statistiques sont apprises dans `fit()` et appliquées dans `transform()`. `src/fonctions_preparation_03.py` contient les fonctions autonomes utilisées par l'étude notebook.

## Organisation du dépôt

| Fichier | Rôle |
|---|---|
| [01 : Préparation](notebooks/01%20-%20Preparation.ipynb) | Étude : exploration et traitements A/B, cellule par cellule |
| [src/preparation_03.py](src/preparation_03.py) | Industrialisation : classe `Preparation03` (`fit`/`transform`) |
| [src/fonctions_preparation_03.py](src/fonctions_preparation_03.py) | Fonctions autonomes utilisées par l'étude notebook |
| [tests/](tests/) | Contrôles de non-fuite et d'équivalence des formats d'identifiants |

## Reproduire

Environnement utilisé : **Python 3.12 sous Windows**. Les versions des bibliothèques sont dans [requirements.txt](requirements.txt).

1. Télécharger `TrainAndValid.csv` depuis la page Kaggle et le placer dans `data/raw/bluebook-for-bulldozers/`.
2. Depuis la racine du dépôt, créer l'environnement et installer les dépendances :

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m jupyterlab
```

3. Choisir le noyau de cet environnement, puis exécuter le notebook 01, d'abord avec `FENETRE = "principal"`, puis avec `FENETRE = "confirmation"`.

Les données brutes sont exclues de Git.
