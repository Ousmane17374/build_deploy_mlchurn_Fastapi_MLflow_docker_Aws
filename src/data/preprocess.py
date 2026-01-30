import pandas as pd

def preprocess_data(df: pd.DataFrame, target_col: str = "Churn") -> pd.DataFrame:
    """
    Basic cleaning for Telco churn dataset.
    - trim column names
    - drop obvious ID columns
    - fix TotalCharges to numeric
    - map target Churn to 0/1 if needed
    - simple NA handling
    """
    df = df.copy()

    # tidy headers
    df.columns = df.columns.str.strip()

    # drop IDs if present
    for col in ["customerID", "CustomerID", "customer_id"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # target to 0/1 if it's yes/no
    if target_col in df.columns and df[target_col].dtype == object:
        mapped = (
            df[target_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
        )
        df[target_col] = mapped  # laisse NaN si valeurs inattendues (à gérer ailleurs si besoin)

    # TotalCharges often has blanks -> coerce to NaN
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # SeniorCitizen should be 0/1 ints
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = df["SeniorCitizen"].fillna(0).astype(int)

    # Numeric NA handling
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(0)

    return df
