import os
import sys
import pathlib
import joblib
import pandas as pd

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# ----------------------------------------------------------------------
# 1. Project Root Path Resolution
# ----------------------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.retriever import RAGRetriever

# ----------------------------------------------------------------------
# 2. Pipeline Class Definition & Model Loading
# ----------------------------------------------------------------------

FEATURE_COLS = [
    "donation_count",
    "total_donations",
    "average_donation",
    "campaign_count",
    "days_since_first_donation",
    "donor_tenure_days",
    "avg_days_between_donations",
    "donations_last_365_days",
    "revenue_last_365_days",
    "frequency_velocity_365d",
    "monetary_velocity_365d",
]

# Definition included to ensure smooth unpickling from joblib
class DonorInferencePipeline:
    def __init__(self, model, feature_cols):
        self.model = model
        self.feature_cols = feature_cols

    def preprocess(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df_proc = raw_df.copy()
        if "avg_days_between_donations" in df_proc.columns:
            df_proc["avg_days_between_donations"] = df_proc["avg_days_between_donations"].fillna(-1)
        df_proc = df_proc.fillna(0)
        return df_proc[self.feature_cols]

    def predict_proba(self, raw_df: pd.DataFrame):
        X_proc = self.preprocess(raw_df)
        return self.model.predict_proba(X_proc)[:, 1]


MODEL_PATH = r"C:\Users\mdesc\Documents\Projects\MpactAI\app\src\models\best_donor_model_xgb.pkl"
CSV_PATH = r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\donor_features_v2.csv"

_pipeline_instance = None
_csv_data_instance = None


def get_model_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Run training script first.")
        _pipeline_instance = joblib.load(MODEL_PATH)
    return _pipeline_instance


def get_donor_dataframe():
    global _csv_data_instance
    if _csv_data_instance is None:
        if not os.path.exists(CSV_PATH):
            raise FileNotFoundError(f"Data file '{CSV_PATH}' not found.")
        _csv_data_instance = pd.read_csv(CSV_PATH)
    return _csv_data_instance


# ----------------------------------------------------------------------
# 3. Agent Tools
# ----------------------------------------------------------------------

@tool
def get_donor_prediction(donor_id: str) -> str:
    """Queries donor_features_v2.csv by donor ID and executes the trained XGBoost ML model to compute donation likelihood."""
    try:
        df_donors = get_donor_dataframe()
        pipeline = get_model_pipeline()
    except Exception as e:
        return f"System Error initializing prediction resources: {str(e)}"

    # Check for matching donor row
    matching_rows = pd.DataFrame()
    if "donor_id" in df_donors.columns:
        matching_rows = df_donors[df_donors["donor_id"].astype(str) == str(donor_id)]
    else:
        # Fallback to integer index if 'donor_id' column is missing
        try:
            idx = int(donor_id)
            if 0 <= idx < len(df_donors):
                matching_rows = df_donors.iloc[[idx]]
        except ValueError:
            pass

    if matching_rows.empty:
        return f"Error: Donor ID #{donor_id} was not found in {CSV_PATH}."

    donor_row = matching_rows.iloc[[0]]

    # Run prediction via loaded .pkl pipeline
    probability = float(pipeline.predict_proba(donor_row)[0])

    # Extract key feature metrics safely for LLM context
    rev = donor_row["revenue_last_365_days"].values[0] if "revenue_last_365_days" in donor_row.columns else 0.0
    donations = donor_row["donations_last_365_days"].values[0] if "donations_last_365_days" in donor_row.columns else 0
    cadence = donor_row["avg_days_between_donations"].values[0] if "avg_days_between_donations" in donor_row.columns else 0.0

    return f"""
    DONOR PREDICTION RESULTS:
    - Donor ID: #{donor_id}
    - Likelihood of Donation (90-day window): {probability:.1%} (Score: {probability:.4f})
    
    KEY CSV METRICS:
    - 365-day Total Revenue: ${rev:,.2f}
    - 365-day Donation Count: {donations}
    - Avg Days Between Gifts: {cadence} days
    """


@tool
def search_knowledge_base(query: str) -> str:
    """Searches organizational document knowledge base for outreach guidelines or program context."""
    retriever = RAGRetriever()
    results = retriever.get_context_for_llm(query, top_k=2)
    return results["raw_context"]


# ----------------------------------------------------------------------
# 4. System Prompt & Agent Class
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI Donor Engagement Specialist for Future Horizons Youth Foundation.

Your objective is to combine ML predictive scores with strategic fundraising action plans.

WORKFLOW:
1. ALWAYS call `get_donor_prediction(donor_id)` first when asked about a specific donor.
2. Evaluate the probability score alongside the CSV feature metrics.
3. Formulate a recommendation using these thresholds:
   - High Likelihood (≥ 75%): Personal 1-on-1 outreach or major gift stewardship.
   - Moderate Likelihood (40% - 74%): Targeted campaign updates, quarterly touchpoints.
   - Low Likelihood (< 40%): Automated re-engagement or general newsletter list.
4. Output a clear summary containing:
   - Donor ID
   - Likelihood of donation (%)
   - Recommended action
   - Core reasoning/drivers based on model results.
5. Formulate an email to send to the donor.
   - Include a subject line, greeting, body, and closing.
   - Ensure the email is concise, personalized, and aligned with the recommended action.
   - Always cite the source of your information (CSV metrics and ML model) in your reasoning.
"""


class DonorAgent:
    def __init__(self, model_name: str = "llama3.2"):
        self.tools = [get_donor_prediction, search_knowledge_base]
        self.llm = ChatOllama(model=model_name, temperature=0.1)
        self.agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT
        )

    def evaluate_donor(self, query: str) -> str:
        response = self.agent.invoke({"messages": [("user", query)]})
        return response["messages"][-1].content


# ----------------------------------------------------------------------
# 5. Execution Test
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("\n🤖 Initializing Dynamic ML-Powered Donor Agent...")
    agent = DonorAgent(model_name="llama3.2")

    # Change ID '0' or '1045' to match a valid row/ID in your CSV file
    user_query = "How much has donor #DNR-06256 donated in total"

    print("\n" + "=" * 70)
    print(f"📩 USER REQUEST: {user_query}")
    print("=" * 70 + "\n")

    result = agent.evaluate_donor(user_query)

    print("\n" + "=" * 70)
    print("📊 AGENT ANALYSIS & RECOMMENDATION")
    print("=" * 70)
    print(result)