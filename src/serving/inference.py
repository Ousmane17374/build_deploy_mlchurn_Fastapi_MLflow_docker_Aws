# src/serving/inference.py

from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import mlflow


# -------------------------------------------------------------------
# Paths (train.py exporte ici: src/serving/model/{model, feature_columns.json, threshold.txt})
# -------------------------------------------------------------------
SERVING_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR = SERVING_DIR / "model"                 # contient MLmodel + artifacts/
FEATURES_PATH = SERVING_DIR / "feature_columns.json"
THRESHOLD_PATH = SERVING_DIR / "threshold.txt"

# Lazy-loaded globals (évite de recharger à chaque requête)
_model = None
_feature_cols: list[str] | None = None
_threshold: float | None = None


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def _require_file(path: Path, hint: str):
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier manquant: {path}. {hint}"
        )


def _load_threshold() -> float:
    _require_file(THRESHOLD_PATH, "Lance d'abord: python -m src.models.train (il génère threshold.txt).")
    return float(THRESHOLD_PATH.read_text(encoding="utf-8").strip())


def _load_feature_cols() -> list[str]:
    _require_file(FEATURES_PATH, "Lance d'abord: python -m src.models.train (il génère feature_columns.json).")
    return json.loads(FEATURES_PATH.read_text(encoding="utf-8"))


def _load_model():
    _require_file(MODEL_DIR, "Lance d'abord: python -m src.models.train (il exporte le modèle dans src/serving/model/model).")
    # Charge le modèle MLflow exporté localement
    return mlflow.pyfunc.load_model(str(MODEL_DIR))


def _coerce_numeric_inputs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Force les types numériques sur les colonnes numériques attendues.
    Tes données UI peuvent arriver en str, donc on sécurise.
    """
    df = df.copy()

    # Colonnes numériques attendues par ton schéma FastAPI (CustomerData)
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Gestion NA numériques -> 0 (cohérent avec preprocess_data)
    for c in numeric_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0)

    # tenure / SeniorCitizen doivent être int
    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].astype(int)
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


def _final_cast_for_mlflow(df: pd.DataFrame) -> pd.DataFrame:
    """
    IMPORTANT: Certains modèles MLflow/pyfunc refusent object dtype.
    Après build_features(), tout devrait être numérique,
    mais on sécurise en convertissant object -> numeric quand possible.
    """
    df = df.copy()

    # Convertir toutes les colonnes object en numeric si possible
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in obj_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Remplacer NaN par 0 (sécurise les dummies manquants)
    df = df.fillna(0)

    # Cast final en float (safe pour MLflow)
    # (LightGBM accepte float; et ça évite les erreurs Int64/nullable)
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float)

    return df


# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def predict(payload: dict) -> dict:
    """
    payload: dict brut (strings + numbers) venant de FastAPI/Gradio.
    Retour: dict {prediction, probability, threshold}
    """

    global _model, _feature_cols, _threshold

    # Lazy load
    if _model is None:
        _model = _load_model()
    if _feature_cols is None:
        _feature_cols = _load_feature_cols()
    if _threshold is None:
        _threshold = _load_threshold()

    # 1) Payload -> DataFrame (1 ligne)
    df = pd.DataFrame([payload])

    # 2) Coercion des num au plus tôt (UI envoie parfois des str)
    df = _coerce_numeric_inputs(df)

    # 3) Pipeline EXACT training : preprocess + build_features
    # NOTE: preprocess_data ne touche au target que s'il existe, donc OK sans Churn dans payload
    from src.data.preprocess import preprocess_data
    from src.features.build_features import build_features

    df = preprocess_data(df, target_col="Churn")
    df = build_features(df, target_col="Churn")

    # 4) Align columns EXACTEMENT comme train (feature_columns.json)
    #    -> ajoute toutes les colonnes manquantes à 0
    for col in _feature_cols:
        if col not in df.columns:
            df[col] = 0

    #    -> supprime celles en trop et respecte l'ordre
    df = df[_feature_cols]

    # 5) Cast final numeric/bool
    df = _final_cast_for_mlflow(df)

    # 6) Predict
    # MLflow pyfunc: predict() renvoie souvent un array shape (n,)
    yhat = _model.predict(df)

    # Selon le wrapper, yhat peut être:
    # - numpy array
    # - pandas Series
    # - list
    proba = float(yhat[0])

    pred = int(proba >= _threshold)

    return {
        "prediction": pred,
        "probability": round(proba, 6),
        "threshold": float(_threshold),
    }
