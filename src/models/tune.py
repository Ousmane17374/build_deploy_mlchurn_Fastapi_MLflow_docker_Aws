import optuna
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, recall_score


def tune_model(X, y):
    """
    Tunes and lightgbm model using optuna

    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.

    """
    # Explicit scorer: optimize recall for churners (class 1)
    recall_scorer = make_scorer(recall_score, pos_label=1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 800),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),

            # gestion du déséquilibre (aligné avec ton meilleur modèle)
            "class_weight": "balanced",

            "random_state": 42,
            "n_jobs": -1
        }

        model = LGBMClassifier(**params)

        # cross_val_score expects: (estimator, X, y, ...)
        scores = cross_val_score(model, X, y, cv=3, scoring=recall_scorer)
        return scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("Best Params:", study.best_params)

    return study.best_params
