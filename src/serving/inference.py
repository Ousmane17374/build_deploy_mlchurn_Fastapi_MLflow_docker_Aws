from pathlib import Path
import json
import pandas as pd
import mlflow

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

SERVING_DIR = Path(__file__).resolve().parent / "model"
MODEL_DIR = SERVING_DIR / "model"  # contient MLmodel + artifacts
FEATURES_PATH = SERVING_DIR / "feature_columns.json"
THRESHOLD_PATH = SERVING_DIR / "threshold.txt"

_model = None
_feature_cols = None
_threshold = None


def _load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Missing threshold file: {THRESHOLD_PATH}")
    return float(THRESHOLD_PATH.read_text(encoding="utf-8").strip())


def _load_feature_cols():
    if not FEATURES_PATH.exists():
        raise FileNotFoundError(f"Missing feature columns file: {FEATURES_PATH}")
    return json.loads(FEATURES_PATH.read_text(encoding="utf-8"))


def _load_model():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Missing exported MLflow model folder: {MODEL_DIR}")
    return mlflow.pyfunc.load_model(str(MODEL_DIR))


def predict(payload: dict) -> dict:
    """
    payload: dict avec les colonnes brutes (comme ton schema FastAPI)
    return: dict JSON (prediction + proba + threshold)
    """
    global _model, _feature_cols, _threshold

    if _model is None:
        _model = _load_model()
    if _feature_cols is None:
        _feature_cols = _load_feature_cols()
    if _threshold is None:
        _threshold = _load_threshold()

    # 1) payload -> dataframe
    df = pd.DataFrame([payload])

    # 2) preprocess + features (API n'a pas Churn => on passe target_col mais sans colonne)
    df = preprocess_data(df, target_col="Churn")
    df = build_features(df, target_col="Churn", serving=True)
  # doit tolérer l'absence de Churn (corrigé)

    # 3) aligner colonnes au training
    for c in _feature_cols:
        if c not in df.columns:
            df[c] = 0
    df = df[_feature_cols]

    # 4) predict proba
    # MLflow pyfunc predict() renvoie souvent un array shape (n,)
    proba = float(_model.predict(df)[0])
    pred = int(proba >= _threshold)

    return {
        "prediction": pred,
        "probability": proba,
        "threshold": _threshold
    }
