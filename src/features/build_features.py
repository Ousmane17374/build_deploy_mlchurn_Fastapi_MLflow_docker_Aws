import pandas as pd

BINARY_COLS_FIXED = [
    "gender", "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

MULTI_CAT_COLS_FIXED = ["InternetService", "Contract", "PaymentMethod"]


def _map_binary_series(s: pd.Series, col_name: str = "") -> pd.Series:
    s = s.astype(str).str.strip()

    # Normalisation "Yes/No"
    if col_name.lower() != "gender":
        s_lower = s.str.lower()
        if set(s_lower.dropna().unique()) <= {"yes", "no"}:
            return s_lower.map({"yes": 1, "no": 0}).astype("Int64")

    # Gender
    if col_name.lower() == "gender":
        s_lower = s.str.lower()
        if set(s_lower.dropna().unique()) <= {"male", "female"}:
            return s_lower.map({"male": 0, "female": 1}).astype("Int64")

    # Fallback stable mapping
    uniq = sorted(list(pd.Series(s.dropna().unique()).astype(str)))
    if len(uniq) == 1:
        return s.map({uniq[0]: 0}).astype("Int64")
    if len(uniq) == 2:
        mapping = {uniq[0]: 0, uniq[1]: 1}
        return s.map(mapping).astype("Int64")

    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn", serving: bool = False) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    # Merge categories
    cols_service = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    cols_present = [c for c in cols_service if c in df.columns]
    if cols_present:
        df[cols_present] = df[cols_present].replace({
            "No phone service": "No",
            "No internet service": "No",
        })

    # fixed columns in serving
    if serving:
        binary_cols = [c for c in BINARY_COLS_FIXED if c in df.columns and c != target_col]
        multi_cat_cols = [c for c in MULTI_CAT_COLS_FIXED if c in df.columns and c != target_col]
    else:
        obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
        binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
        multi_cat_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    # apply encoding
    for c in binary_cols:
        df[c] = _map_binary_series(df[c], col_name=c).fillna(0).astype(int)

    if multi_cat_cols:
        df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

    # bool -> int
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)

    return df
