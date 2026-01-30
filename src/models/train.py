import mlflow
import pandas as pd
import numpy as np
import json
import shutil
from pathlib import Path

import mlflow.lightgbm  # correct MLflow LightGBM module import
from lightgbm import LGBMClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from sklearn.metrics import recall_score, precision_score, f1_score

# (your project pipeline imports: your folder layout is src/data and src/features)
from src.data.load_data import load_data
from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

# (your evaluation helper)
from src.models.evaluate import evaluate_model

# (your data validation helper)
from src.utils.validate_data import validate_telco_data

# ------------------------------------------------------------
# Set MLflow experiment (PROJECT NAME)
# ------------------------------------------------------------
mlflow.set_experiment("Build & Deploy ML churn model with FastAPI, MLFlow, Docker, & AWS")


def train_model(
    data_path: str,
    target_col: str = "Churn",
    threshold: np.float32 = np.float32(0.5120157),
    registered_model_name: str = "churn_lightgbm",
):
    """
    Trains a LightGBM model and logs with MLflow.

    Args:
        data_path (str): Path to the CSV file.
        target_col (str): Name of the target column.

    """

    # ------------------------------------------------------------
    # Load + preprocess + build features (must match serving)
    # ------------------------------------------------------------
    df = load_data(data_path)

    # ------------------------------------------------------------
    # Data validation (Great Expectations) - must run BEFORE preprocess
    # because preprocess may drop columns like customerID
    # ------------------------------------------------------------
    ok, failed = validate_telco_data(df)
    if not ok:
        raise ValueError(f"Data validation failed. Failed expectations: {failed}")

    df = preprocess_data(df, target_col=target_col)
    df = build_features(df, target_col=target_col)

    # ------------------------------------------------------------
    # SAVE FEATURE COLUMNS + THRESHOLD (CRITICAL FOR SERVING)
    # Because get_dummies creates columns that must match in production.
    # ------------------------------------------------------------
    serving_dir = Path("src/serving/model")
    serving_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c != target_col]
    with open(serving_dir / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, ensure_ascii=False, indent=2)

    with open(serving_dir / "threshold.txt", "w", encoding="utf-8") as f:
        f.write(str(float(threshold)))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ------------------------------------------------------------
    # Model = your Optuna best params + imbalance handling
    # ------------------------------------------------------------
    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.01879986840002807,
        max_depth=1,
        num_leaves=124,
        min_child_samples=30,
        subsample=0.6113533556358944,
        colsample_bytree=0.7544685619637796,
        reg_alpha=0.12306766729519758,
        reg_lambda=1.2133040783636315,
        class_weight="balanced",  # critical: keep your chosen imbalance strategy
        random_state=42,
        n_jobs=-1,
    )

    with mlflow.start_run() as run:
        # Train model
        model.fit(X_train, y_train)

        # --------------------------------------------------------
        # Predict with probabilities + threshold (your chosen setup)
        # --------------------------------------------------------
        proba = model.predict_proba(X_test)[:, 1]
        preds = (proba >= threshold).astype(int)

        # Metrics (aligned with churn objective + standard metrics)
        acc = accuracy_score(y_test, preds)
        rec = recall_score(y_test, preds, pos_label=1)
        prec = precision_score(y_test, preds, pos_label=1)
        f1 = f1_score(y_test, preds, pos_label=1)
        auc = roc_auc_score(y_test, proba)

        # log params, metrics, and model
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("learning_rate", 0.01879986840002807)
        mlflow.log_param("max_depth", 1)
        mlflow.log_param("num_leaves", 124)
        mlflow.log_param("min_child_samples", 30)
        mlflow.log_param("subsample", 0.6113533556358944)
        mlflow.log_param("colsample_bytree", 0.7544685619637796)
        mlflow.log_param("reg_alpha", 0.12306766729519758)
        mlflow.log_param("reg_lambda", 1.2133040783636315)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("threshold", float(threshold))

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("roc_auc", auc)

        # log the sklearn/lightgbm model to MLflow
        mlflow.lightgbm.log_model(
            model,
            artifact_path="model",
            registered_model_name=registered_model_name,
        )

        # log dataset so it shows in MLflow UI
        train_ds = mlflow.data.from_pandas(df, source=data_path)
        mlflow.log_input(train_ds, context="training")

        print(f"Model trained. Accuracy: {acc:.4f}, Recall: {rec:.4f}")
        print("=== Classification report (thresholded) ===")
        print(classification_report(y_test, preds, digits=3))

        # ------------------------------------------------------------
        # Evaluation (externalized in evaluate.py)
        # ------------------------------------------------------------
        evaluate_model(model, X_test, y_test, threshold=threshold)

        # ------------------------------------------------------------
        # EXPORT SERVING ARTIFACTS (STABLE PATH FOR DOCKER/FASTAPI)
        # Robust approach: download artifacts from MLflow to stable folder
        #
        # IMPORTANT (Windows):
        # Avoid passing "file:D:/..." URIs directly; instead, download by run_id + artifact_path.
        # This prevents WinError 123 / path parsing issues.
        # ------------------------------------------------------------
        dst_model_dir = serving_dir / "model"

        # clean previous exported model
        if dst_model_dir.exists():
            shutil.rmtree(dst_model_dir)

        # Download the model artifacts to our stable folder
        # This will create: src/serving/model/model/...
        mlflow.artifacts.download_artifacts(
            run_id=run.info.run_id,
            artifact_path="model",
            dst_path=str(serving_dir),
        )

        print(f"✅ Exported serving model to: {dst_model_dir}")
        print(f"✅ Exported feature columns to: {serving_dir / 'feature_columns.json'}")
        print(f"✅ Exported threshold to: {serving_dir / 'threshold.txt'}")


if __name__ == "__main__":
    train_model(
        data_path="data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv",
        target_col="Churn",
    )
