from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, X_test, y_test, threshold: np.float32 = np.float32(0.5120157)):
    """
    Evaluates an XGBoost model on test data.
    Args:
        model: Trained model.
        X_test: Test features.
        y_test: Test labels.

    """

    # ------------------------------------------------------------
    # Predict probabilities and apply threshold
    # (kept generic to support LightGBM / XGBoost style models)
    # ------------------------------------------------------------
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    print(f"Threshold used for evaluation: {threshold}")
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))
