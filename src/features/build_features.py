# src/features/build_features.py
import pandas as pd

BINARY_COLS_FIXED = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

MULTI_CAT_COLS_FIXED = [
    "InternetService",
    "Contract",
    "PaymentMethod",
]


def _encode_binary(col: pd.Series, name: str) -> pd.Series:
    col = col.astype(str).str.strip()

    if name.lower() == "gender":
        return (
            col.str.lower()
            .map({"male": 0, "female": 1})
            .fillna(0)
            .astype(int)
        )

    return (
        col.str.lower()
        .map({"yes": 1, "no": 0})
        .fillna(0)
        .astype(int)
    )


def build_features(
    df: pd.DataFrame,
    target_col: str = "Churn",
    serving: bool = False,
) -> pd.DataFrame:
    """
    Feature engineering Telco churn.
    - OK training (multi-lignes)
    - OK serving (1 ligne)
    - Ne renvoie JAMAIS de colonnes dtype=object
    """
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Normaliser les catégories "No internet service"/"No phone service"
    replace_map = {
        "No internet service": "No",
        "No phone service": "No",
    }
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].replace(replace_map)

    # Sélection colonnes cat
    if serving:
        binary_cols = [c for c in BINARY_COLS_FIXED if c in df.columns and c != target_col]
        multi_cat_cols = [c for c in MULTI_CAT_COLS_FIXED if c in df.columns and c != target_col]
    else:
        obj_cols = [c for c in df.select_dtypes(include="object").columns if c != target_col]
        binary_cols = [c for c in obj_cols if df[c].dropna().nunique() <= 2]
        multi_cat_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    # Encodage binaire
    for c in binary_cols:
        df[c] = _encode_binary(df[c], c)

    # One-hot multi-cat
    if multi_cat_cols:
        df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # bool -> int
    bool_cols = df.select_dtypes(include="bool").columns
    if len(bool_cols) > 0:
        df[bool_cols] = df[bool_cols].astype(int)

    # GARANTIE: plus aucun object
    obj_left = [c for c in df.select_dtypes(include="object").columns if c != target_col]
    if obj_left:
        df = pd.get_dummies(df, columns=obj_left, drop_first=True)

    return df
