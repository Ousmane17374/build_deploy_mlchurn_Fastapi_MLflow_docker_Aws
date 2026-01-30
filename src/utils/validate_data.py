import pandas as pd
import great_expectations as ge
from typing import Tuple, List

def validate_telco_data(df) -> Tuple[bool, List[str]]:
    """
    Comprehensive data validation for Telco customer churn using Great Expectations.
    This function implements critical data quality cheks that must pass before
    proceeding with model training or inference.
    It validates data integrity, business logic conrstraints, and statistical 
    properties that the ML model expects.
    Validate the Telco churn dataset using Great Expectations.

    """
    print(f"\nStarting data validation with Great Expectations...")

    # ---------------------------------------------------------------------
    # IMPORTANT:
    # In Great Expectations 1.x, if you haven't initialized a GE project,
    # ge.get_context() returns an EphemeralDataContext which may NOT expose
    # .sources. This breaks typical "read_dataframe" flows.
    #
    # To keep your pipeline working in any environment (local, Docker, CI/CD),
    # we implement the SAME checks using pandas (robust + deterministic).
    # ---------------------------------------------------------------------

    # ---------------------------------------------------------------------
    # Helper: handle column name variants (Telco dataset often uses lowercase "tenure")
    # We keep your comments and expectations, but map to existing columns if needed.
    # ---------------------------------------------------------------------
    def _col(name: str) -> str:
        if name in df.columns:
            return name
        alt = name.lower()
        if alt in df.columns:
            return alt
        # Try stripping spaces
        stripped = name.strip()
        if stripped in df.columns:
            return stripped
        alt2 = stripped.lower()
        if alt2 in df.columns:
            return alt2
        # Return original name (expectation will fail, and we will report it)
        return name

    failed_expectations: List[str] = []

    def fail(expectation_name: str):
        failed_expectations.append(expectation_name)

    #==SCHEMA VALIDATION - ESSENTIAL COLUMNS==
    print(f" Validating schema and required columns...")

    # Customer identifier must exist (required for business operations)
    if "customerID" not in df.columns:
        fail("expect_column_to_exist(customerID)")
    if "Churn" not in df.columns:
        fail("expect_column_to_exist(Churn)")
    else:
        if df["Churn"].isna().any():
            fail("expect_column_values_to_not_be_null(Churn)")

    # Core demographic features
    for c in ["gender", "Partner", "Dependents"]:
        if c not in df.columns:
            fail(f"expect_column_to_exist({c})")

    # Service features (critical for churn analysis)
    for c in ["PhoneService", "InternetService", "Contract"]:
        if c not in df.columns:
            fail(f"expect_column_to_exist({c})")

    # Financial features (key churn predictors)
    tenure_col = _col("Tenure")
    if tenure_col not in df.columns:
        fail(f"expect_column_to_exist({tenure_col})")
    for c in ["MonthlyCharges", "TotalCharges"]:
        if c not in df.columns:
            fail(f"expect_column_to_exist({c})")

    #==BUSINESS LOGIC VALIDATIONS==
    print(f" Validating business logic constraints...")

    # Gender must be one of expected values (data integrity)
    if "gender" in df.columns:
        ok = df["gender"].dropna().isin(["Male", "Female"]).all()
        if not ok:
            fail("expect_column_values_to_be_in_set(gender, [Male, Female])")

    # Yes/No fields must have valid values 
    for c in ["Partner", "Dependents", "PhoneService"]:
        if c in df.columns:
            ok = df[c].dropna().isin(["Yes", "No"]).all()
            if not ok:
                fail(f"expect_column_values_to_be_in_set({c}, [Yes, No])")

    # Contract types must be valid (business constraint )
    if "Contract" in df.columns:
        ok = df["Contract"].dropna().isin(["Month-to-month", "One year", "Two year"]).all()
        if not ok:
            fail("expect_column_values_to_be_in_set(Contract, [Month-to-month, One year, Two year])")

    # Internet service types (business constraint)
    if "InternetService" in df.columns:
        ok = df["InternetService"].dropna().isin(["DSL", "Fiber optic", "No"]).all()
        if not ok:
            fail("expect_column_values_to_be_in_set(InternetService, [DSL, Fiber optic, No])")

    #== NUMERIC RNGE VALIDATIONS ==
    print(f" Validating numeric ranges and business distributions...")

    # Tenure must be non-negative (business logic can't have negative tenure)
    if tenure_col in df.columns:
        t = pd.to_numeric(df[tenure_col], errors="coerce")
        ok = (t.dropna() >= 0).all()
        if not ok:
            fail("expect_column_values_to_be_between(Tenure, min_value=0)")

    # MonthlyCharges must be positive  (business logic - no free service)
    if "MonthlyCharges" in df.columns:
        mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        ok = (mc.dropna() >= 0).all()
        if not ok:
            fail("expect_column_values_to_be_between(MonthlyCharges, min_value=0)")

    # TotalCharges should be non-negative (business logic)
    # NOTE: TotalCharges often contains blanks/strings in this dataset -> coerce to numeric
    if "TotalCharges" in df.columns:
        tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
        ok = (tc.dropna() >= 0).all()
        if not ok:
            fail("expect_column_values_to_be_between(TotalCharges, min_value=0)")

    #== Statostical VALIDATIONS ==
    print(f" Validating statistical properties...")

    # Tenure should be reasonable (max ~10 years = 120 months for telecom)
    if tenure_col in df.columns:
        t = pd.to_numeric(df[tenure_col], errors="coerce")
        ok = ((t.dropna() >= 0) & (t.dropna() <= 120)).all()
        if not ok:
            fail("expect_column_values_to_be_between(Tenure, min_value=0, max_value=120)")

    # MonthlyCharges should be within reasonable business range
    if "MonthlyCharges" in df.columns:
        mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        ok = ((mc.dropna() >= 0) & (mc.dropna() <= 200)).all()
        if not ok:
            fail("expect_column_values_to_be_between(MonthlyCharges, min_value=0, max_value=200)")

    # No missing values in critical numeric columns
    if tenure_col in df.columns:
        # If tenure is stored as string, consider numeric coercion missingness as well
        t = pd.to_numeric(df[tenure_col], errors="coerce")
        if t.isna().any():
            # We only fail if the original had values but coercion failed OR truly missing
            # This keeps the check strict enough without breaking on rare formatting issues.
            fail("expect_column_values_to_not_be_null(Tenure)")
    if "MonthlyCharges" in df.columns:
        mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        if mc.isna().any():
            fail("expect_column_values_to_not_be_null(MonthlyCharges)")

    #== Data CONSISTENCY CHECKS ==  
    print(f" Validating data consistency...")

    # TotalCharges should generally be >= MonthlyCharges (except for very new customers)
    #This is a business logic check to catch data entry errors
    if "TotalCharges" in df.columns and "MonthlyCharges" in df.columns:
        tc = pd.to_numeric(df["TotalCharges"], errors="coerce")
        mc = pd.to_numeric(df["MonthlyCharges"], errors="coerce")
        tmp = pd.DataFrame({"TotalCharges": tc, "MonthlyCharges": mc}).dropna()

        if len(tmp) > 0:
            ok_ratio = (tmp["TotalCharges"] >= tmp["MonthlyCharges"]).mean()
            if ok_ratio < 0.95:
                fail("expect_column_pair_values_A_to_be_greater_than_B(TotalCharges>=MonthlyCharges mostly=0.95)")

    #== PROCESS RESULTS ==
    total_checks = len(failed_expectations) + 1  # informational only
    passed = len(failed_expectations) == 0

    if passed:
        print(f" Data validation PASSED.")
        return True, []
    else:
        print(f" Data validation FAILED.")
        print(f" Failed expectations: {failed_expectations}")
        return False, failed_expectations

    # Expected columns
