import joblib
import pandas as pd
from langchain_core.tools import tool

# 1. Load your saved pipeline (.pkl handles feature selection & preprocessing automatically)
pipeline = joblib.load(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\src\models\best_donor_model_xgb.pkl")

# 2. Load donor dataset into memory (or index by donor_id column)
DONOR_CSV_PATH = r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\donor_features_v2.csv"
df_donors = pd.read_csv(DONOR_CSV_PATH)

# Ensure donor_id column is formatted consistently (string or int)
if "donor_id" in df_donors.columns:
    df_donors["donor_id"] = df_donors["donor_id"].astype(str)


@tool
def get_donor_prediction(donor_id: str) -> str:
    """Queries the donor CSV dataset and runs the trained XGBoost model to compute donation likelihood."""
    # Find the specific donor row in the DataFrame
    matching_rows = df_donors[df_donors["donor_id"] == str(donor_id)]

    if matching_rows.empty:
        return f"Error: Donor ID #{donor_id} was not found in {DONOR_CSV_PATH}."

    donor_row = matching_rows.iloc[[0]]

    # Generate prediction using your saved pipeline
    probability = float(pipeline.predict_proba(donor_row)[0])

    # Extract key metrics for agent context
    total_rev = donor_row.get("revenue_last_365_days", [0]).values[0]
    recent_donations = donor_row.get("donations_last_365_days", [0]).values[0]
    avg_cadence = donor_row.get("avg_days_between_donations", [0]).values[0]

    return f"""
    DONOR PREDICTION RESULTS:
    - Donor ID: #{donor_id}
    - Likelihood of Donation (90-day window): {probability:.1%} (Score: {probability:.4f})
    
    FEATURE METRICS (From CSV):
    - 365-day Total Revenue: ${total_rev:,.2f}
    - 365-day Donations: {recent_donations}
    - Avg Cadence Between Gifts: {avg_cadence} days
    """