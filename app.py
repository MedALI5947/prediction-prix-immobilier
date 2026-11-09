import joblib
import pandas as pd
import streamlit as st

from src.preprocessing import preparer_donnees

st.set_page_config(page_title="Prix immobilier", page_icon="🏠")
st.title("🏠 Estimation du prix d'une maison")
st.write("Renseignez les caractéristiques de la maison pour obtenir une estimation de son prix de vente.")


@st.cache_resource
def charger_modele():
    return joblib.load("models/modele.joblib")


modele = charger_modele()

quartiers = [
    "Blmngtn", "Blueste", "BrDale", "BrkSide", "ClearCr", "CollgCr", "Crawfor",
    "Edwards", "Gilbert", "IDOTRR", "MeadowV", "Mitchel", "NAmes", "NPkVill",
    "NWAmes", "NoRidge", "NridgHt", "OldTown", "SWISU", "Sawyer", "SawyerW",
    "Somerst", "StoneBr", "Timber", "Veenker",
]
qualites = {"Excellente": "Ex", "Bonne": "Gd", "Moyenne": "TA", "Faible": "Fa"}

col1, col2 = st.columns(2)

with col1:
    st.subheader("Qualité et emplacement")
    overall_qual = st.slider("Qualité globale (1 à 10)", 1, 10, 6)
    exter_qual = st.selectbox("Qualité de l'extérieur", list(qualites), index=2)
    kitchen_qual = st.selectbox("Qualité de la cuisine", list(qualites), index=2)
    neighborhood = st.selectbox("Quartier", quartiers, index=quartiers.index("NAmes"))
    year_built = st.number_input("Année de construction", 1870, 2010, 1972)
    year_remod = st.number_input("Année de rénovation", 1950, 2010, 1993)

with col2:
    st.subheader("Surfaces et équipements")
    gr_liv_area = st.number_input("Surface habitable (pieds²)", 300, 6000, 1442)
    first_flr = st.number_input("Surface du rez-de-chaussée (pieds²)", 300, 5000, 1088)
    total_bsmt = st.number_input("Surface du sous-sol (pieds²)", 0, 6500, 994)
    lot_area = st.number_input("Surface du terrain (pieds²)", 1000, 220000, 9458)
    tot_rms = st.slider("Nombre de pièces (hors salles de bain)", 2, 14, 6)
    full_bath = st.slider("Salles de bain complètes", 0, 3, 2)
    garage_cars = st.slider("Garage (nombre de voitures)", 0, 4, 2)
    fireplaces = st.slider("Cheminées", 0, 3, 1)

if st.button("Estimer le prix", type="primary"):
    maison = pd.DataFrame([{
        "OverallQual": overall_qual,
        "GrLivArea": gr_liv_area,
        "TotalBsmtSF": total_bsmt,
        "1stFlrSF": first_flr,
        "YearBuilt": year_built,
        "YearRemodAdd": year_remod,
        "GarageCars": garage_cars,
        "GarageArea": 0,  # non utilisée par le modèle, retirée par preparer_donnees
        "FullBath": full_bath,
        "TotRmsAbvGrd": tot_rms,
        "KitchenQual": qualites[kitchen_qual],
        "Fireplaces": fireplaces,
        "LotArea": lot_area,
        "Neighborhood": neighborhood,
        "ExterQual": qualites[exter_qual],
    }])

    prix = modele.predict(preparer_donnees(maison))[0]

    st.success(f"Prix estimé : **{prix:,.0f} $**".replace(",", " "))
    st.caption("Erreur moyenne du modèle : environ 19 000 $.")