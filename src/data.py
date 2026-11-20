import pandas as pd


def load_data(path):
    """Load the Telco churn CSV and apply the cleaning steps from EDA."""
    df = pd.read_csv(path)

    # 11 new customers (tenure = 0) have a blank TotalCharges -> they were never billed
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)

    # Target as 0/1
    df["churn_flag"] = (df["Churn"] == "Yes").astype(int)

    return df