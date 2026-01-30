import pandas as pd
from sklearn.model_selecton import train_test_split
from XGBoost import XGBClassifier
import optuna

print("=== Phase2: Modeling with XGBoost===")

df=pd.read_csv("data/processed/telco_churn_processed.csv")

#Target must numeric 0/1
if df["Churn"].dtype=="object":
    df["Churn"]=df["Churn"].str.strip().map({"NO":0, "Yes":1})

assert df["Churn"].isna().sum() == 0, "Churn has NaNs"
assert set(df["Churn"].unique()) <= {0,1}, "Churn not 0/1"

X=df.drop(colulns=["Churn"])
y=df["Churn"]

X_train, X_tes, y_train, y_test=train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

THRESHOLD = np.float32(0.5120157)


def objective(trial):
    params={
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "n_jobs": -1,
            "scale_pos_weight": (y_train==0).sum()/(y_train==1).sum(),
            "eval_metric": "logloss"
        }
    model= XGBClassifier (**params)
    model.fit(X_train, y_train)
    proba=model.predict_proba(X_test)[:,1]
    y_pred=(proba>=THRESHOLD).astype(int)
    from sklearn.metrics import recall_score
    return recall_score(y_test, y_pred, pos_label=1)

study=optuna.create_study(diretion="maximize")
study.optimize(objective, n_trials=30)
print("Best Params:", study.best_params)
print("Best Recall", study.bestvalue)
    
       

