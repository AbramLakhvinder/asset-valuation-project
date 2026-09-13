"""Comparaison global / segmenté, sur des validations antérieures à 2012.

Forme industrialisée des expériences menées pas à pas dans le notebook 02.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

from src.preparation_03 import Preparation03


def rmsle_log(y, prediction):
    """RMSE sur la cible log1p(SalePrice), c'est-à-dire le RMSLE sur les prix.

    Calcul identique à mean_squared_error(y, prediction) ** 0.5 des notebooks.
    """
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(prediction)) ** 2)))


def check_split(train, validation):
    """Garde-fou à appeler avant toute expérience touchant les vraies données."""
    if train.empty or validation.empty:
        raise ValueError('Apprentissage et validation doivent être non vides.')
    if not train.saledate.max() < validation.saledate.min():
        raise ValueError('Chevauchement temporel apprentissage/validation.')
    if not validation.saledate.max() < pd.Timestamp('2012-01-01'):
        raise ValueError('Ces expériences ne doivent pas accéder au test 2012.')
    if not train.index.intersection(validation.index).empty:
        raise ValueError('Index communs entre apprentissage et validation.')


def baselines(train, validation):
    """Deux règles simples que les modèles doivent battre pour être utiles."""
    check_split(train, validation)
    ytr = np.log1p(train.SalePrice)
    moyenne_par_famille = ytr.groupby(train.ProductGroup).mean()
    return {
        'Moyenne log globale': np.full(len(validation), ytr.mean()),
        'Moyenne log par segment':
            validation.ProductGroup.map(moyenne_par_famille).fillna(ytr.mean()).to_numpy(),
    }


def entrainer_specialistes(Xtr, ytr, params, cat_features, min_segment=200):
    """Un CatBoost par famille assez fournie pour justifier son propre modèle.

    Les familles trop petites n'ont pas de spécialiste : elles s'appuieront sur
    le modèle global au moment de prédire.
    """
    variables = [c for c in Xtr.columns if c != 'ProductGroup']
    modeles = {}
    for groupe in sorted(Xtr.ProductGroup.unique()):
        lignes = Xtr.ProductGroup.eq(groupe)
        if lignes.sum() < min_segment:
            print(f'  {groupe} : {lignes.sum():,} lignes, trop peu — repli sur le global.',
                  flush=True)
            continue
        print(f'  spécialiste {groupe} : {lignes.sum():,} lignes', flush=True)
        modele = CatBoostRegressor(**params)
        modele.fit(Xtr.loc[lignes, variables], ytr.loc[lignes], cat_features=cat_features)
        modeles[groupe] = modele
    return modeles


def appliquer_specialistes(modeles, X, pred_global):
    """Chaque spécialiste remplace les prédictions de sa famille, le reste garde le global.

    Renvoie les prédictions et le masque des ventes restées sur le global :
    aucune ligne n'est exclue du score.
    """
    variables = [c for c in X.columns if c != 'ProductGroup']
    predictions = np.asarray(pred_global).copy()
    repli = np.ones(len(X), dtype=bool)
    for groupe, modele in modeles.items():
        lignes = X.ProductGroup.eq(groupe).to_numpy()
        if not lignes.any():
            continue
        predictions[lignes] = modele.predict(X.loc[lignes, variables])
        repli[lignes] = False
    return predictions, repli


def _enregistrer(directory, label, methode, rows, predictions, y, index):
    if directory is None:
        return
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(directory / f'{label}_{methode}_scores.csv', index=False)
    np.savez_compressed(directory / f'{label}_{methode}_predictions.npz',
                        y=y, index=index, **predictions)


def fit_pair(train, validation, methode, params, output_dir=None, label='principal',
             min_segment=200):
    """Entraîne le global et les spécialistes sur la même préparation.

    Même préparation, mêmes paramètres, mêmes lignes : seule l'architecture
    change. Le global sert aussi de repli. Aucun eval_set ni early stopping ne
    voit la validation.
    """
    check_split(train, validation)
    prep = Preparation03(methode=methode).fit(train)
    Xtr, Xva = prep.transform(train), prep.transform(validation)
    if not prep.fit_max_date_ < validation.saledate.min():
        raise ValueError('Préparation ajustée sur des ventes postérieures à la validation.')
    if {'SalePrice', 'logSalePrice', 'saledate'} & set(Xtr.columns):
        raise ValueError('Cible ou date de vente présente dans les variables explicatives.')
    ytr, yva = np.log1p(train.SalePrice), np.log1p(validation.SalePrice)

    print(f'\n[{label} — préparation {methode}]', flush=True)
    print(f'Apprentissage : {len(train):,} ventes jusqu\'au {train.saledate.max().date()} ; '
          f'validation : {len(validation):,} ventes '
          f'du {validation.saledate.min().date()} au {validation.saledate.max().date()}.',
          flush=True)
    print(f'{Xtr.shape[1]} variables, statistiques figées sur l\'apprentissage.', flush=True)

    rows, predictions = [], {}
    directory = Path(output_dir) if output_dir is not None else None

    debut = time.perf_counter()
    global_model = CatBoostRegressor(**params)
    global_model.fit(Xtr, ytr, cat_features=prep.cat_features_)
    predictions['global'] = global_model.predict(Xva)
    rows.append({'fenetre': label, 'methode': methode, 'architecture': 'global',
                 'rmsle': rmsle_log(yva, predictions['global']),
                 'secondes': time.perf_counter() - debut, 'replis': 0})
    _enregistrer(directory, label, methode, rows, predictions,
                 yva.to_numpy(), validation.index.to_numpy())
    print(f"GLOBAL - RMSLE validation = {rows[0]['rmsle']:.5f} "
          f"({rows[0]['secondes']:.0f} s)", flush=True)

    debut = time.perf_counter()
    cat_features = [c for c in prep.cat_features_ if c != 'ProductGroup']
    modeles = entrainer_specialistes(Xtr, ytr, params, cat_features, min_segment)
    segmente, repli = appliquer_specialistes(modeles, Xva, predictions['global'])
    predictions['segmente'] = segmente
    rows.append({'fenetre': label, 'methode': methode, 'architecture': 'segmente',
                 'rmsle': rmsle_log(yva, segmente),
                 'secondes': time.perf_counter() - debut, 'replis': int(repli.sum())})
    _enregistrer(directory, label, methode, rows, predictions,
                 yva.to_numpy(), validation.index.to_numpy())
    print(f"SEGMENTE - RMSLE validation = {rows[1]['rmsle']:.5f} "
          f"({rows[1]['secondes']:.0f} s, {int(repli.sum())} ventes sur le global)", flush=True)
    print(f"Gain global - segmente : {rows[0]['rmsle'] - rows[1]['rmsle']:+.5f} "
          '(positif = favorable aux spécialistes)', flush=True)

    # Détail par famille : global et spécialiste jugés sur les mêmes ventes.
    detail = []
    for groupe in sorted(Xva.ProductGroup.unique()):
        lignes = Xva.ProductGroup.eq(groupe).to_numpy()
        score_global = rmsle_log(yva.to_numpy()[lignes], predictions['global'][lignes])
        score_segmente = rmsle_log(yva.to_numpy()[lignes], segmente[lignes])
        detail.append({'segment': groupe, 'n': int(lignes.sum()),
                       'global': score_global, 'segmente': score_segmente,
                       'gain': score_global - score_segmente,
                       'replis': int((repli & lignes).sum())})
    if directory is not None:
        pd.DataFrame(detail).to_csv(directory / f'{label}_{methode}_segments.csv', index=False)

    return {'scores': pd.DataFrame(rows), 'predictions': predictions,
            'segments': pd.DataFrame(detail), 'y': yva.to_numpy(),
            'dates': validation.saledate.copy(), 'n_features': Xtr.shape[1]}


def architecture_gain(result, n_bootstrap=1000, seed=2012):
    """Variabilité hebdomadaire du gain ; intervalle descriptif, sans réentraînement.

    Les semaines sont rééchantillonnées en gardant appariées les erreurs des
    deux architectures : elles sont toujours jugées sur les mêmes ventes.
    """
    y = result['y']
    a, b = result['predictions']['global'], result['predictions']['segmente']
    semaine = result['dates'].dt.to_period('W').astype(str).to_numpy()
    erreurs = pd.DataFrame({'semaine': semaine, 'a': (y - a) ** 2, 'b': (y - b) ** 2, 'n': 1})
    par_semaine = erreurs.groupby('semaine')[['a', 'b', 'n']].sum().to_numpy()

    rng = np.random.default_rng(seed)
    tirages = rng.integers(0, len(par_semaine), size=(n_bootstrap, len(par_semaine)))
    sommes = par_semaine[tirages].sum(axis=1)
    gains = np.sqrt(sommes[:, 0] / sommes[:, 2]) - np.sqrt(sommes[:, 1] / sommes[:, 2])

    return {'gain': rmsle_log(y, a) - rmsle_log(y, b),
            'borne_basse_95': float(np.quantile(gains, .025)),
            'borne_haute_95': float(np.quantile(gains, .975)),
            'semaines': len(par_semaine)}
