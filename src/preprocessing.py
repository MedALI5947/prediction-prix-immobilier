import pandas as pd


def preparer_donnees(df):
    """Ajoute SurfaceQualite et retire GarageArea (redondante avec GarageCars)."""
    df = df.copy()
    df["SurfaceQualite"] = df["GrLivArea"] * df["OverallQual"]
    df = df.drop(columns=["GarageArea"])
    return df