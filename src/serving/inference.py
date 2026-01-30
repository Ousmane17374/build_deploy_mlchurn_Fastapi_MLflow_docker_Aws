# src/serving/inference.py
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import mlflow
import mlflow.lightgbm


# -------------------------------------------------------------------
# Paths (train.py exporte ici: src/serving/model/{model, feature_columns.json, threshold.txt})
# -------------------------------------------------------------------
SERVING_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR = SERVING_DIR / "model"                 # contient MLmodel + artifacts/
FEATURES_PATH = SERVING_DIR / "feature_columns.json"
THRESHOLD_PATH = SERVING_DIR / "threshold.txt"

_model = None
_feature_cols: list[str] | None = None
_threshold: float | None = None


def _require_file(path: Path, hint: str):
    if not path.exists():
        raise FileNotFoundError(f"Fichier manquant: {path}. {hint}")


def _load_threshold() -> float:
    _require_file(THRESHOLD_PATH, "Lance: python -m src.models.train (génère threshold.txt).")
    return float(THRESHOLD_PATH.read_text(encoding="utf-8").strip())


def _load_feature_cols() -> list[str]:
    _require_file(FEATURES_PATH, "Lance: python -m src.models.train (génère feature_columns.json).")
    return json.loads(FEATURES_PATH.read_text(encoding="utf-8"))


def _load_model():
    _require_file(MODEL_DIR, "Lance: python -m src.models.train (exporte le modèle dans src/serving/model/model).")
    # ✅ IMPORTANT: on charge en flavor lightgbm pour avoir predict_proba()
    return mlflow.lightgbm.load_model(str(MODEL_DIR))


def _coerce_numeric_inputs(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

    if "tenure" in df.columns:
        df["tenure"] = df["tenure"].astype(int)

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].astype(int)

    return df


def _final_cast(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # MLflow + LightGBM : safest => float
    df = df.fillna(0)

    # ✅ Dernière sécurité: pas de dtype object en entrée du modèle
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        # En théorie, ça ne doit jamais arriver si build_features(serving=True) a bien fait le job,
        # mais on sécurise pour éviter l'erreur UI.
        df = pd.get_dummies(df, columns=obj_cols, drop_first=True)

    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float)

    return df


def predict(payload: dict) -> dict:
    """
    payload brut (strings + numbers) venant de FastAPI/Gradio.
    Retour: dict {prediction, probability, threshold}
    """
    global _model, _feature_cols, _threshold

    if _model is None:
        _model = _load_model()
    if _feature_cols is None:
        _feature_cols = _load_feature_cols()
    if _threshold is None:
        _threshold = _load_threshold()

    # 1) payload -> df
    df = pd.DataFrame([payload])

    # 2) types numériques
    df = _coerce_numeric_inputs(df)

    # 3) même pipeline que train
    from src.data.preprocess import preprocess_data
    from src.features.build_features import build_features

    df = preprocess_data(df, target_col="Churn")

    # ✅ FIX PRINCIPAL : en serving on FORCE les colonnes fixes (sinon 1 ligne => nunique=1 => pas d'encodage)
    df = build_features(df, target_col="Churn", serving=True)

    # 4) alignement colonnes (one-hot stable)
    # add missing cols
    for col in _feature_cols:
        if col not in df.columns:
            df[col] = 0

    # drop extra cols (si jamais)
    df = df[_feature_cols]

    # 5) cast final
    df = _final_cast(df)

    # 6) proba churn
    proba = float(_model.predict_proba(df)[:, 1][0])
    pred = int(proba >= float(_threshold))

    return {
        "prediction": pred,
        "probability": round(proba, 6),
        "threshold": float(_threshold),
    }
