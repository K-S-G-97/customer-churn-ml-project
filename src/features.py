"""Feature decisions from the leakage audit (notebooks/02_leakage_and_features.ipynb)."""

TARGET = "churn_flag"

# Dropped: customerID (ID), TotalCharges (redundant), gender (no signal + fairness risk)
DROP_COLUMNS = ["customerID", "Churn", "churn_flag", "TotalCharges", "gender"]

NUMERIC_FEATURES = ["tenure", "MonthlyCharges", "SeniorCitizen"]

CATEGORICAL_FEATURES = [
    "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod",
]