import os
import sys
import pathlib
import joblib
import pandas as pd
from xgboost import XGBClassifier
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent

# Ensure project root path resolution
ROOT_DIR = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from rag.retriever import RAGRetriever

# ----------------------------------------------------------------------
# 1. Load ML Model & Feature Preprocessing
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

# Option A: Load from native XGBoost JSON file
xgb_model = XGBClassifier()
xgb_model.load_model("best_donor_model_xgb.json")


def preprocess_and_predict(raw_data: dict) -> float:
    """Preprocesses a single donor feature dictionary and returns predicted probability."""
    df = pd.DataFrame([raw_data])
    if "avg_days_between_donations" in df.columns:
        df["avg_days_between_donations"] = df["avg_days_between_donations"].fillna(-1)
    df = df.fillna(0)
    X = df[FEATURE_COLS]
    prob = xgb_model.predict_proba(X)[0][1]
    return float(prob)


# Simulated donor feature database for testing
MOCK_DONOR_DATABASE = {
    "1045": {
        "donation_count": 8,
        "total_donations": 1850.0,
        "average_donation": 231.25,
        "campaign_count": 3,
        "days_since_first_donation": 420,
        "donor_tenure_days": 420,
        "avg_days_between_donations": 35.0,
        "donations_last_365_days": 4,
        "revenue_last_365_days": 950.0,
        "frequency_velocity_365d": 1.2,
        "monetary_velocity_365d": 1.15
    },
    "2088": {
        "donation_count": 2,
        "total_donations": 100.0,
        "average_donation": 50.0,
        "campaign_count": 1,
        "days_since_first_donation": 650,
        "donor_tenure_days": 650,
        "avg_days_between_donations": 300.0,
        "donations_last_365_days": 0,
        "revenue_last_365_days": 0.0,
        "frequency_velocity_365d": 0.0,
        "monetary_velocity_365d": 0.0
    }
}


# ----------------------------------------------------------------------
# 2. Define Agent Tools
# ----------------------------------------------------------------------

@tool
def get_donor_prediction(donor_id: str) -> str:
    """Queries the trained XGBoost ML model to compute donation activation probability and feature metrics for a donor ID."""
    donor_data = MOCK_DONOR_DATABASE.get(donor_id)
    if not donor_data:
        return f"Error: Donor ID #{donor_id} was not found in the database."

    probability = preprocess_and_predict(donor_data)

    return f"""
    DONOR PREDICTION RESULTS:
    - Donor ID: #{donor_id}
    - Donation Likelihood (90-day window): {probability:.1%} (Probability score: {probability:.4f})
    
    KEY FEATURE HIGHLIGHTS:
    - Total 365-day Revenue: ${donor_data['revenue_last_365_days']:,.2f}
    - Recent 365-day Donations: {donor_data['donations_last_365_days']}
    - Donation Cadence: Every {donor_data['avg_days_between_donations']} days
    - Monetary Velocity Score: {donor_data['monetary_velocity_365d']}x
    """


@tool
def search_knowledge_base(query: str) -> str:
    """Searches organizational documents for outreach guidelines or program details."""
    retriever = RAGRetriever()
    results = retriever.get_context_for_llm(query, top_k=2)
    return results["raw_context"]


# ----------------------------------------------------------------------
# 3. Define System Prompt & Agent
# ----------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI Donor Engagement Specialist for Future Horizons Youth Foundation.

You connect machine learning predictive scores with strategic fundraising recommendations.

YOUR WORKFLOW:
1. When asked about a donor, always call `get_donor_prediction(donor_id)` first.
2. Analyze the probability score and feature highlights returned by the ML model.
3. Formulate a recommended action using these engagement guidelines:
   - High Likelihood (≥ 75%): Highly personalized 1-on-1 outreach or major gift stewardship.
   - Moderate Likelihood (40% - 74%): Standard engagement, quarterly updates, or mid-tier campaign invites.
   - Low Likelihood (< 40%): Automated re-engagement campaign or general newsletter touchpoints.
4. Output a clear, structured recommendation detailing:
   - Donor ID
   - Likelihood of donation (%)
   - Recommended action
   - Key drivers/reasons behind the score.
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
# 4. Execution Demo
# ----------------------------------------------------------------------

if __name__ == "__main__":
    print("\n🤖 Initializing ML-Powered Donor AI Agent...")
    agent = DonorAgent(model_name="llama3.2")

    user_query = "Assess donor #1045 and recommend an action based on their prediction score."

    print("\n" + "=" * 70)
    print(f"📩 USER REQUEST: {user_query}")
    print("=" * 70 + "\n")

    result = agent.evaluate_donor(user_query)

    print("\n" + "=" * 70)
    print("📊 AGENT ANALYSIS & RECOMMENDATION")
    print("=" * 70)
    print(result)