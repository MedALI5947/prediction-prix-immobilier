import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.preprocessing import preparer_donnees

# 1. Chargement des données
train = pd.read_csv("data/raw/kaggle_train_set.csv")
test = pd.read_csv("data/raw/kaggle_test_set.csv")

# 2. On retire les maisons très grandes vendues pas cher (valeurs aberrantes)
aberrantes = (train["GrLivArea"] > 4500) & (train["SalePrice"] < 300000)
train = train[~aberrantes]
print("Maisons retirées :", aberrantes.sum())

# 3. Préparation
X = preparer_donnees(train.drop(columns=["SalePrice"]))
y = train["SalePrice"]

colonnes_categorielles = ["KitchenQual", "Neighborhood", "ExterQual"]

pretraitement = ColumnTransformer(
    [("categoriel", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
      colonnes_categorielles)],
    remainder="passthrough",
)

# 4. Modèle : Gradient Boosting entraîné sur log(prix)
pipeline = Pipeline([
    ("pretraitement", pretraitement),
    ("modele", HistGradientBoostingRegressor(random_state=42)),
])
modele = TransformedTargetRegressor(regressor=pipeline, func=np.log, inverse_func=np.exp)

# 5. Évaluation en validation croisée
scores = cross_val_score(modele, X, y, cv=5, scoring="neg_mean_absolute_error")
print(f"MAE moyenne : {-scores.mean():.0f} $")

# 6. Entraînement final et sauvegarde
modele.fit(X, y)
joblib.dump(modele, "models/modele.joblib")
print("Modèle sauvegardé dans models/modele.joblib")

# 7. Fichier de soumission Kaggle
predictions = modele.predict(preparer_donnees(test))
soumission = pd.DataFrame({"Id": range(len(test)), "SalePrice": predictions})
soumission.to_csv("data/processed/submission.csv", index=False)
print("Soumission enregistrée dans data/processed/submission.csv")