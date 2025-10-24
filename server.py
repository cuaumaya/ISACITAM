# server.py
# API de predicción OBV (CatBoost + Huber) con fallback "demo sensible a sliders"

import os
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor, Pool
import joblib
from fastapi.middleware.cors import CORSMiddleware

# ---------- CONFIG ----------
MODEL_DIR = "models"
CATBOOST_PATH = os.path.join(MODEL_DIR, "catboost_base.cbm")
HUBER_PATH = os.path.join(MODEL_DIR, "meta_huber.pkl")

PLAYER_FEATS = [
    "player_season_obv_90", "player_season_xgchain_90", "player_season_xa_90",
    "player_season_f3_lbp_ratio", "player_season_carry_length", "player_season_left_foot_ratio",
    "player_season_pressured_pass_length_ratio", "player_season_pressure_regains_90",
    "player_season_defensive_action_regains_90", "player_season_obv_shot_90",
    "player_season_shot_on_target_ratio", "Age"
]
TEAM_CONTROLS_12 = [
    "team_season_xgd_pg", "team_season_possession", "team_season_obv_pg",
    "team_season_gd_pg", "team_season_successful_passes_pg", "team_season_ppda",
    "team_season_yellow_cards_pg", "team_season_deep_progressions_pg",
    "team_season_obv_conceded_pg", "team_season_successful_crosses_into_box_pg",
    "team_season_goals_from_corners_conceded_pg", "team_season_passing_ratio"
]
TEAM_BASELINE = ["obv_for_to_team_t0"]
CATS_OPTIONAL = ["primary_position", "competition_name"]
INTERACTION_FEATS = ["possession_xgd_interaction", "style_fit_direct", "intensity_diff", "pass_quality_context"]
FEATURE_ORDER = PLAYER_FEATS + TEAM_CONTROLS_12 + TEAM_BASELINE + CATS_OPTIONAL + INTERACTION_FEATS

DEFAULTS = {
    # Jugador
    "player_season_obv_90": 0.15,
    "player_season_xgchain_90": 0.25,
    "player_season_xa_90": 0.08,
    "player_season_f3_lbp_ratio": 0.35,
    "player_season_carry_length": 8.5,
    "player_season_left_foot_ratio": 0.30,
    "player_season_pressured_pass_length_ratio": 0.50,
    "player_season_pressure_regains_90": 1.2,
    "player_season_defensive_action_regains_90": 0.8,
    "player_season_obv_shot_90": 0.05,
    "player_season_shot_on_target_ratio": 0.38,
    "Age": 25,
    # Equipo
    "team_season_xgd_pg": 0.30,
    "team_season_possession": 55.0,
    "team_season_obv_pg": 1.20,
    "team_season_gd_pg": 0.35,
    "team_season_successful_passes_pg": 350.0,
    "team_season_ppda": 10.5,
    "team_season_yellow_cards_pg": 2.0,
    "team_season_deep_progressions_pg": 8.0,
    "team_season_obv_conceded_pg": 0.80,
    "team_season_successful_crosses_into_box_pg": 1.2,
    "team_season_goals_from_corners_conceded_pg": 0.05,
    "team_season_passing_ratio": 1.8,
    # Baseline y categorías
    "obv_for_to_team_t0": 1.20,
    "primary_position": "NA",
    "competition_name": "Liga MX",
}

# ---------- CARGA MODELOS ----------
cat_model, meta_model = None, None
if os.path.exists(CATBOOST_PATH):
    cat_model = CatBoostRegressor()
    cat_model.load_model(CATBOOST_PATH)
if os.path.exists(HUBER_PATH):
    meta_model = joblib.load(HUBER_PATH)

# ---------- FASTAPI ----------
app = FastAPI(title="OBV Contextual API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # en prod limita a tu dominio
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- ESQUEMAS ----------
class PlayerProfile(BaseModel):
    obv_90: float
    xgchain_90: float
    xa_90: float
    carry_length: float
    pressure_regains_90: float
    age: int

class TeamProfile(BaseModel):
    possession: float
    ppda: float
    passing_ratio: float
    xgd_pg: float
    obv_pg: float

class CatsProfile(BaseModel):
    primary_position: str = "NA"
    competition_name: str = "Liga MX"

class PredictRequest(BaseModel):
    player: PlayerProfile
    team: TeamProfile
    cats: CatsProfile = CatsProfile()

# ---------- HELPERS ----------
def build_feature_row(player: PlayerProfile, team: TeamProfile, cats: CatsProfile) -> pd.DataFrame:
    vals = DEFAULTS.copy()

    # Mapas jugador
    vals["player_season_obv_90"] = player.obv_90
    vals["player_season_xgchain_90"] = player.xgchain_90
    vals["player_season_xa_90"] = player.xa_90
    vals["player_season_carry_length"] = player.carry_length
    vals["player_season_pressure_regains_90"] = player.pressure_regains_90
    vals["Age"] = player.age

    # Equipo
    vals["team_season_possession"] = team.possession
    vals["team_season_ppda"] = team.ppda
    vals["team_season_passing_ratio"] = team.passing_ratio
    vals["team_season_xgd_pg"] = team.xgd_pg
    vals["team_season_obv_pg"] = team.obv_pg

    # Baseline
    vals["obv_for_to_team_t0"] = vals["team_season_obv_pg"]

    # Categóricas
    vals["primary_position"] = cats.primary_position or "NA"
    vals["competition_name"] = cats.competition_name or "Liga MX"

    # Interacciones (idénticas a entrenamiento)
    vals["possession_xgd_interaction"] = vals["team_season_possession"] * vals["team_season_xgd_pg"]
    denom = 0.3 if 0.3 != 0 else 1.0
    vals["style_fit_direct"] = vals["team_season_passing_ratio"] * (vals["player_season_xgchain_90"] / denom)
    vals["intensity_diff"] = vals["player_season_pressure_regains_90"] - (1.0 / max(vals["team_season_ppda"], 1e-6))
    vals["pass_quality_context"] = vals["player_season_obv_90"] * vals["team_season_possession"]

    row = {k: vals.get(k, np.nan) for k in FEATURE_ORDER}
    df = pd.DataFrame([row])
    for c in CATS_OPTIONAL:
        if c in df.columns:
            df[c] = df[c].astype("string")
    return df

def predict_dict(X: pd.DataFrame):
    # Base
    if cat_model is not None:
        cat_idx = [X.columns.get_loc(c) for c in CATS_OPTIONAL if c in X.columns]
        base_pred = float(cat_model.predict(Pool(X, cat_features=cat_idx if cat_idx else None))[0])
    else:
        # Fallback "demo sensible"
        possession = float(X["team_season_possession"].iloc[0])
        ppda = float(X["team_season_ppda"].iloc[0])
        xgd = float(X["team_season_xgd_pg"].iloc[0])
        obv90 = float(X["player_season_obv_90"].iloc[0])
        xgc90 = float(X["player_season_xgchain_90"].iloc[0])
        age = float(X["Age"].iloc[0])
        base_pred = (
            0.8 * (possession / 100.0) +
            0.15 * xgd + 0.7 * obv90 + 0.4 * xgc90 +
            0.05 * (1.0 - abs(age - 27) / 20.0) -
            0.02 * (ppda / 20.0)
        )

    # Meta
    if meta_model is not None:
        yhat = float(meta_model.predict(np.array([[base_pred]], dtype=float))[0])
    else:
        yhat = base_pred

    # Factores (para radar/UX)
    possessionFactor = float(X["team_season_possession"].iloc[0]) / 100.0
    ppdaFactor = max(0.0, 1.0 - float(X["team_season_ppda"].iloc[0]) / 20.0)
    ageFactor = 1.0 - abs(float(X["Age"].iloc[0]) - 27.0) / 20.0
    passing_ratio = float(X["team_season_passing_ratio"].iloc[0])
    xgc90 = float(X["player_season_xgchain_90"].iloc[0])
    styleFit = passing_ratio * (xgc90 / 0.3 if 0.3 != 0 else 1.0)
    possessionInteraction = possessionFactor * float(X["team_season_xgd_pg"].iloc[0])
    intensityDiff = float(X["player_season_pressure_regains_90"].iloc[0]) - (1.0 / max(float(X["team_season_ppda"].iloc[0]), 1e-6))

    factors = {
        "possession": int(possessionFactor * 100),
        "intensity": int(ppdaFactor * 100),
        "age": int(max(0, min(1, ageFactor)) * 100),
        "style": int(min(100, styleFit * 50)),
    }
    compatibility = min(100.0, (possessionInteraction * 100.0) + (styleFit * 20.0) + max(0.0, 30.0 - abs(intensityDiff * 20.0)))
    confidence = min(95.0, 70.0 + (ageFactor * 10.0) + (possessionFactor * 15.0))

    return {
        "predictedOBV": round(yhat, 3),
        "confidence": round(confidence, 1),
        "compatibility": int(compatibility),
        "factors": factors,
        "mode": "stacking" if (cat_model and meta_model) else ("catboost" if cat_model else "demo")
    }

# ---------- ENDPOINT ----------
@app.post("/predict")
def predict(req: PredictRequest):
    X = build_feature_row(req.player, req.team, req.cats)
    out = predict_dict(X)
    return out