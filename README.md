# 🏠 Prédiction du prix de l'immobilier

Projet de régression : prédire le prix de vente d'une maison (`SalePrice`) à partir de 15 caractéristiques
(qualité, surfaces, année de construction, quartier...). Les données viennent d'une compétition Kaggle.

**Résultat : erreur moyenne d'environ 19 000 $** (MAE), contre 54 700 $ pour une prédiction naïve au prix médian.

## Démarche

1. **Exploration** : distribution du prix (très asymétrique, d'où l'intérêt du logarithme), corrélations, analyse des plus grosses erreurs.
2. **Nouvelle variable** `SurfaceQualite` = surface × qualité : −900 $ d'erreur.
3. **Comparaison de 6 modèles** (Ridge, Random Forest, Gradient Boosting, avec et sans log) en validation croisée : tous à égalité, autour de 19 000 $.
4. **Choix du Gradient Boosting + log** : contrairement à Ridge, il ne s'emballe pas sur les maisons hors norme
   (Ridge + log prédit 1,1 M$ pour une maison vendue 160 000 $). C'est important pour une application où l'utilisateur saisit ses propres valeurs.
5. **Application Streamlit** pour estimer le prix d'une maison à partir d'un formulaire.

## Structure du projet

```
├── analyse.ipynb          # exploration, comparaison des modèles et justification des choix
├── src/
│   ├── preprocessing.py   # préparation des données (partagée par le script et l'application)
│   └── train.py           # entraînement du modèle final et soumission Kaggle
├── app.py                 # application Streamlit
└── pyproject.toml         # dépendances (Poetry)
```

## Lancer le projet

```bash
# 1. Installer les dépendances
poetry install

# 2. Télécharger les données depuis Kaggle et les placer dans data/raw/
#    (kaggle_train_set.csv et kaggle_test_set.csv, non versionnées : licence de la compétition)

# 3. Entraîner le modèle
poetry run python -m src.train

# 4. Lancer l'application
poetry run streamlit run app.py
```

## Outils

Python · pandas · scikit-learn · Streamlit · Poetry · Git