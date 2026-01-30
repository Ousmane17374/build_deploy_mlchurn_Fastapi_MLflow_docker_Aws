# src/serving/inference.py
from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
import mlflow
import mlflow.lightgbm


SERVING_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR = SERVING_DIR / "model"
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


def _assert_no_object(df: pd.DataFrame, step_name: str):
    """Bloque si des colonnes object restent (sinon LightGBM plantera)."""
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if obj_cols:
        # on montre un exemple de valeur pour aider à diagnostiquer
        examples = {c: df[c].iloc[0] for c in obj_cols[:10]}
        raise ValueError(
            f"[{step_name}] Colonnes dtype=object détectées: {obj_cols}. "
            f"Exemples: {examples}"
        )


def _final_cast_numeric_strict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cast final STRICT:
    - aucune colonne object/string ne passe
    - tout devient float (LightGBM-friendly)
    """
    df = df.copy().fillna(0)

    # Si on a encore des object => on bloque AVANT LightGBM
    _assert_no_object(df, "final_cast_before")

    # Convertit tout ce qui est numérique en float
    for c in df.columns:
        if pd.api.types.is_bool_dtype(df[c]):
            df[c] = df[c].astype(int)
        if pd.api.types.is_numeric_dtype(df[c]):
            df[c] = df[c].astype(float)

    # Double check (parano)
    _assert_no_object(df, "final_cast_after")

    return df


def predict(payload: dict) -> dict:
    global _model, _feature_cols, _threshold

    if _model is None:
        _model = _load_model()
    if _feature_cols is None:
        _feature_cols = _load_feature_cols()
    if _threshold is None:
        _threshold = _load_threshold()

    # 1) payload -> df
    df = pd.DataFrame([payload])

    # 2) coercion num
    df = _coerce_numeric_inputs(df)

    # 3) preprocess + features (SERVING MODE)
    from src.data.preprocess import preprocess_data
    from src.features.build_features import build_features

    # ✅ Preuve du fichier réellement importé dans Docker
    # (Si ça ne pointe pas vers /app/src/features/build_features.py, tu importes autre chose.)
    try:
        print("DEBUG — build_features importé depuis:", build_features.__file__)
    except Exception:
        # au cas où __file__ n'est pas accessible (rare)
        print("DEBUG — build_features importé (chemin indisponible)")

    df = preprocess_data(df, target_col="Churn")

    # ✅ FIX: serving=True (sinon 1 ligne => nunique=1 => pas d’encodage)
    df = build_features(df, target_col="Churn", serving=True)

    # ✅ Vérif 1 (immédiate après encodage)
    print("DEBUG — dtypes après build_features(serving=True):")
    print(df.dtypes)
    _assert_no_object(df, "after_build_features")

    # 4) alignement colonnes (train vs serve)
    for col in _feature_cols:
        if col not in df.columns:
            df[col] = 0

    # On ne garde que l'ordre/ensemble attendu
    df = df[_feature_cols]

    # ✅ Vérif 2 (après alignement)
    print("DEBUG — dtypes après alignement colonnes:")
    print(df.dtypes)
    _assert_no_object(df, "after_align")

    # 5) cast final strict
    df = _final_cast_numeric_strict(df)

    # ✅ Vérif 3 (juste avant modèle)
    print("DEBUG — dtypes juste avant predict_proba:")
    print(df.dtypes)
    _assert_no_object(df, "before_model")

    # 6) predict
    proba = float(_model.predict_proba(df)[:, 1][0])
    pred = int(proba >= float(_threshold))

    return {
        "prediction": pred,
        "probability": round(proba, 6),
        "threshold": float(_threshold),
    }
