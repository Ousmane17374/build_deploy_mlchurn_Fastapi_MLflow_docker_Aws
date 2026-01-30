import pandas as pd

def _map_binary_series(s: pd.Series) -> pd.Series:
    """
    Apply deterministic binary encoding to 2-category features.
    This function implements the core binary encoding logic that converts
    categorical features with exactly two values into 0/1 integers. The mapping
    as deterministic and must be consistent betwen training and serving.
    """
    # Get unique values and remoce NaN
    vals = list(pd.Series(s.dropna().unique()).astype(str))
    valset = set(vals)

    # === DETERMINISTIC BINARY MAPPING ===
    # Critical: these exact mapping are hardcoded in serving pipeline

    # yes/no mapping (most common patern in Telecom data)
    if valset == {"Yes", "No"}:
        return s.map({"Yes": 1, "No": 0}).astype("Int64")

    # Gender mapping (demographic feature)
    if valset == {"Male", "Female"}:
        return s.map({"Male": 0, "Female": 1}).astype("Int64")

    # ==Generic binary mapping==
    # for any other 2-category feature, use stable alphabetical ordering
    if len(vals) == 2:
        # sort values to ensure consistent mapping accross runs
        sorted_vals = sorted(vals)
        mapping = {sorted_vals[0]: 0, sorted_vals[1]: 1}
        return s.astype(str).map(mapping).astype("Int64")

    # == NON-BINARY-FEATURE ==
    # return unchanged - will be handled by one-hot encoding
    return s


def build_features(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Apply complete feature enginering pipeline for trainning data.

    This is the main feature engineering function that transforms raw customer
    into ML-ready features. The transformations must be exactly replicated in
    the serving pipeline to ensure prediction accuracy.

    """

    df = df.copy()
    print(f"\ starting feature engineering on {df.shape[1]} columns...")

    # ---------------------------------------------------------------------
    # EXTRA STEP (IMPORTANT FOR TELCO DATASET):
    # Some columns have 3 categories: "Yes", "No", and "No internet service"/"No phone service".
    # You asked to merge the "* service" category into "No" so they become binary features.
    #
    # #MultipleLines, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
    # #StreamingTV, StreamingMovies,
    #
    # cols=['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    #       'TechSupport', 'StreamingTV', 'StreamingMovies']
    #
    # df[cols] = df[cols].replace({'No phone service':'No', 'No internet service':'No'})
    # df[cols].nunique()
    # ---------------------------------------------------------------------
    cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    cols_present = [c for c in cols if c in df.columns]
    if cols_present:
        df[cols_present] = df[cols_present].replace({
            'No phone service': 'No',
            'No internet service': 'No'
        })
        # optional debug print (kept minimal)
        # print("Merged service categories into 'No' for:", cols_present)

    # ==step 1: identify Feature Types==
    # Find categorical columns (object dtype) excluding the target variable
    obj_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != target_col]
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.to_list()

    print(f"   Found {len(obj_cols)} categorical features and {len(num_cols)} numeric columns.")

    # ==step 2: Split Categorical by cardinality ==
    # Binary features: exactly 2 unique values get binary encoding
    # Multi-categorical features: more than 2 unique values get one-hot encoding
    binary_cols = [c for c in obj_cols if df[c].dropna().nunique() == 2]
    multi_cat_cols = [c for c in obj_cols if df[c].dropna().nunique() > 2]

    print(f" Binary features: {len(binary_cols)} | Multi-category features: {len(multi_cat_cols)}")

    if binary_cols:
        print(f" Binary : {binary_cols}")
    if multi_cat_cols:
        print(f" Multi-Category: {multi_cat_cols}")

    # ==step 3: Apply Binary Encodings==
    # Convert 2-category features to 0/1 integers using deterministic mappings
    for c in binary_cols:
        original_dtype = df[c].dtype
        df[c] = _map_binary_series(df[c].astype(str))
        print(f"  {c}: {original_dtype} -> binary (0/1)")

    # ==step 4: Convert boolean columns ==
    # LightGBM requires integers input not booleans
    bool_cols = df.select_dtypes(include=["bool"]).columns.tolist()
    if bool_cols:
        df[bool_cols] = df[bool_cols].astype(int)
        print(f" Converted {len(bool_cols)} boolean columns to int: {bool_cols}")

    # ==step 5: One-Hot Encoding for Multi-Category features==
    # CRITICAL: use drop_first=True prevents multicollinearity
    if multi_cat_cols:
        print(f" Applying one-hot encoding to {len(multi_cat_cols)} multi-category columns...")
        original_shape = df.shape

        # Apply one-hot encoding with drop_first=True (same as serving)
        df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
        new_features = df.shape[1] - original_shape[1] + len(multi_cat_cols)
        print(f" created  {new_features} new features from {len(multi_cat_cols)}.")

    # == Step6: Data Type Clean-up ==
    # Convert nullable integers (Int64) to standard integers (int64) for LightGBM compatibility
    for c in binary_cols:
        if pd.api.types.is_integer_dtype(df[c]):
            # fill any NaN values with 0 and convert to int
            df[c] = df[c].fillna(0).astype(int)

    print(f" Feature engineering complete: {df.shape[1]} final features")
    return df

    # ---------------------------------------------------------------------
    # Legacy / duplicated code below was unreachable because of return df.
    # I keep it here as reference, but it should not run.
    # ---------------------------------------------------------------------
    #
    # # Apply binary encoding to 2-category features
    # for col in df.select_dtypes(include=["object","category"]).columns:
    #     if col == target_col:
    #         continue
    #     if df[col].nunique(dropna=True) == 2:
    #         df[col] = _map_binary_series(df[col])
    #
    # # One-hot encode multi-category features
    # multi_cat_cols = [
    #     col for col in df.select_dtypes(include=["object","category"]).columns
    #     if col != target_col and df[col].nunique(dropna=True) > 2
    # ]
    # df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=True)
    #
    # return df
