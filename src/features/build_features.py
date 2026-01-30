import pandas as pd


def _normalize_str_series(s: pd.Series) -> pd.Series:
    """Normalize string-like categorical series: strip and keep original casing for known patterns when possible."""
    # Keep NaN as NaN, only normalize strings
    return s.where(s.isna(), s.astype(str).str.strip())


def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Deterministic binary encoding for 2-category features.
    Must be consistent between training and serving.
    """
    s = _normalize_str_series(s)

    # Unique non-null values
    vals = pd.Series(s.dropna().unique()).astype(str).tolist()
    valset = set(vals)

    # Handle common "Yes/No" (case-insensitive)
    lower_set = {v.lower() for v in valset}
    if lower_set == {"yes", "no"}:
        # preserve original values mapping by normalizing to lower for mapping
        return s.astype(str).str.strip().str.lower().map({"yes": 1, "no": 0}).astype("Int64")

    # Handle gender (case-insensitive)
    if lower_set == {"male", "female"}:
        return s.astype(str).str.strip().str.lower().map({"male": 0, "female": 1}).astype("Int64")

    # Generic deterministic mapping for any other 2-category feature
    if len(valset) == 2:
        sorted_vals = sorted(vals, key=lambda x: x.lower())
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    # Not binary
    return s


def build_features(df: pd.DataFrame, target_col: str | None = "Churn") -> pd.DataFrame:
    """
    Feature engineering for Telco churn dataset.
    Compatible with:
      - training data (target present)
      - inference payloads (no target)
    """

    df = df.copy()
    df.columns = df.columns.str.strip()

    print(f"\\ starting feature engineering on {df.shape[1]} columns...")

    # Merge special categories into "No" to make columns binary where intended
    service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    cols_present = [c for c in service_cols if c in df.columns]
    if cols_present:
        df[cols_present] = df[cols_present].replace(
            {"No phone service": "No", "No internet service": "No"}
        )

    # Identify categorical columns (object-like) excluding target if present
    obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
    if target_col is not None and target_col in obj_cols:
        obj_cols.remove(target_col)

    # Identify numeric columns robustly
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if target_col is not None and target_col in num_cols:
        num_cols.remove(target_col)

    print(f"   Found {len(obj_cols)} categorical features and {len(num_cols)} numeric columns.")

    # Split categorical by cardinality
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cat_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    print(f" Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cat_cols)}")
    if binary_cols:
        print(f" Binary : {binary_cols}")
    if multi_cat_cols:
        print(f" Multi-Category: {multi_cat_cols}")

    # Apply binary encoding
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c])
        print(f"  {c}: {original_dtype} -> binary (0/1)")

    # Convert boolean columns to int
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f" Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    # One-hot encoding for multi-category features
    if multi_cat_cols:
        print(f" Applying one-hot encoding to {len(multi_cat_cols)} multi-category columns...")
        original_shape = df.shape

        df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)

        new_features = df.shape[1] - original_shape[1] + len(multi_cat_cols)
        print(f" created  {new_features} new features from {len(multi_cat_cols)}.")

    # Convert nullable integers to int (LightGBM likes plain int)
    for c in binary_cols:
        if c in df.columns:
            df[c] = df[c].fillna(0).astype(int)

    print(f" Feature engineering complete: {df.shape[1]} final features")
    return df
