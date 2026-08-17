import os
import sys
import pathlib
import datetime
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

# ----------------------------------------------------------------------
# 1. Project Root & Path Setup
# ----------------------------------------------------------------------
ROOT_DIR = pathlib.Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Attempt imports from your local project modules with fallback placeholders
try:
    from src.rag.retriever import RAGRetriever
except ImportError:
    RAGRetriever = None

try:
    from src.agents.grant_agent import GrantAgent
except ImportError:
    GrantAgent = None

try:
    from src.agents.donor_agent import DonorAgent, get_model_pipeline, get_donor_dataframe
except ImportError:
    DonorAgent, get_model_pipeline, get_donor_dataframe = None, None, None


# ----------------------------------------------------------------------
# 2. Page Setup & Configuration
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="ImpactAI Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar Navigation
st.sidebar.title("ImpactAI")
st.sidebar.caption("AI & ML Operating System for Nonprofits")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate Modules",
    [
        "🧠 Knowledge Assistant",
        "📝 Grant Assistant",
        "👥 Donor Analytics",
        "📈 Donor Prediction",
        "🤖 AI Recommendations",
        "⚙️ Model Information",
    ],
)


# ----------------------------------------------------------------------
# Page 1: 🧠 Knowledge Assistant (RAG Search)
# ----------------------------------------------------------------------
if page == "🧠 Knowledge Assistant":
    st.title("🧠 Knowledge Assistant")
    st.write("Query organizational documents, campaign histories, and policy guidelines via RAG.")

    query = st.text_input("Ask a question about organizational context or documents:", 
                          placeholder="e.g., What are our target outcomes for the middle school STEM initiative?")

    if st.button("Search Knowledge Base", type="primary"):
        if query.strip():
            with st.spinner("Searching RAG vector storage..."):
                if RAGRetriever is not None:
                    try:
                        retriever = RAGRetriever()
                        results = retriever.get_context_for_llm(query, top_k=3)
                        st.subheader("Retrieved Context")
                        st.info(results.get("raw_context", "No context returned."))
                    except Exception as e:
                        st.error(f"Error querying RAG retriever: {e}")
                else:
                    st.warning("`RAGRetriever` module not found. Check path import.")
        else:
            st.warning("Please enter a query.")


# ----------------------------------------------------------------------
# Page 2: 📝 Grant Assistant (Grant Agent)
# ----------------------------------------------------------------------
elif page == "📝 Grant Assistant":
    st.title("📝 Grant Assistant")
    st.write("Generate tailored, evidence-based grant proposal sections powered by Ollama & RAG.")

    col1, col2 = st.columns([2, 1])
    with col1:
        program_name = st.text_input("Program Name:", "STEM Innovators")
        requested_amount = st.text_input("Grant Amount ($):", "50000")
        target_group = st.text_input("Target Group:", "Middle School Students (Grades 6-8)")
    with col2:
        model_name = st.selectbox("LLM Engine:", ["llama3.2", "mistral", "llama3"])

    proposal_request = f"Draft a ${requested_amount} grant proposal for '{program_name}' targeting {target_group}. Include Statement of Need and Expected Impact."

    if st.button("Generate Grant Proposal", type="primary"):
        with st.spinner("Grant Agent researching knowledge base and drafting sections..."):
            if GrantAgent is not None:
                try:
                    agent = GrantAgent(model_name=model_name)
                    proposal_text = agent.generate_proposal(proposal_request)
                    st.markdown("### Generated Proposal Draft")
                    st.markdown(proposal_text)
                except Exception as e:
                    st.error(f"Error executing GrantAgent: {e}")
            else:
                st.warning("`GrantAgent` module not found. Ensure `src/agents/grant_agent.py` exists.")


# ----------------------------------------------------------------------
# Page 3: 👥 Donor Analytics (EDA Dashboard)
# ----------------------------------------------------------------------
elif page == "👥 Donor Analytics":
    st.title("👥 Donor Analytics")
    st.write("Exploratory Data Analysis and active donor distribution trends.")

    csv_path = "data\raw\donor_features_v2.csv"
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)

        # Overview Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Donors", f"{len(df):,}")
        m2.metric("Total Lifetime Revenue", f"${df['total_donations'].sum():,.2f}")
        m3.metric("Avg Gift Size", f"${df['average_donation'].mean():,.2f}")
        active_count = (df['donations_last_90_days'] > 0).sum() if 'donations_last_90_days' in df.columns else 0
        m4.metric("Active (Last 90 Days)", f"{active_count:,}")

        st.markdown("---")

        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("365-Day Revenue vs Donation Frequency")
            fig = px.scatter(
                df, 
                x="donations_last_365_days", 
                y="revenue_last_365_days",
                size="average_donation" if "average_donation" in df.columns else None,
                color_discrete_sequence=["#1f77b4"]
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Distribution of Days Between Gifts")
            if "avg_days_between_donations" in df.columns:
                fig2 = px.histogram(df[df["avg_days_between_donations"] > 0], x="avg_days_between_donations", nbins=20)
                st.plotly_chart(fig2, use_container_width=True)

        st.subheader("Raw Donor Feature Sample")
        st.dataframe(df.head(10), use_container_width=True)
    else:
        st.error(f"Dataset '{csv_path}' not found in current directory.")


# ----------------------------------------------------------------------
# Page 4: 📈 Donor Prediction (ML Inference)
# ----------------------------------------------------------------------
elif page == "📈 Donor Prediction":
    st.title("📈 Donor Prediction")
    st.write("Run XGBoost model inference to estimate 90-day donation activation likelihood.")

    model_path = "src\models\best_donor_model_xgb.pkl"
    csv_path = "data\raw\donor_features_v2.csv"

    if os.path.exists(model_path) and os.path.exists(csv_path):
        pipeline = joblib.load(model_path)
        df_donors = pd.read_csv(csv_path)

        donor_ids = df_donors["donor_id"].astype(str).tolist() if "donor_id" in df_donors.columns else [str(i) for i in range(len(df_donors))]
        
        selected_id = st.selectbox("Select Donor ID:", donor_ids)

        if st.button("Run Prediction Model", type="primary"):
            if "donor_id" in df_donors.columns:
                row = df_donors[df_donors["donor_id"].astype(str) == str(selected_id)].iloc[[0]]
            else:
                row = df_donors.iloc[[int(selected_id)]]

            prob = float(pipeline.predict_proba(row)[0])

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Donation Likelihood Score", f"{prob:.1%}")
                if prob >= 0.75:
                    st.success("Risk Tier: HIGH LIKELIHOOD (Target for major gift / personal contact)")
                elif prob >= 0.40:
                    st.warning("Risk Tier: MODERATE LIKELIHOOD (Target for general campaign)")
                else:
                    st.error("Risk Tier: LOW LIKELIHOOD (Automated re-engagement list)")

            with col2:
                st.subheader("Donor Feature Values")
                st.json(row.to_dict(orient="records")[0])
    else:
        st.error("Model file (`best_donor_model_xgb.pkl`) or CSV dataset not found.")


# ----------------------------------------------------------------------
# Page 5: 🤖 AI Recommendations (Donor Agent)
# ----------------------------------------------------------------------
elif page == "🤖 AI Recommendations":
    st.title("🤖 AI Recommendations")
    st.write("Combine predictive ML scores with LLM reasoning to generate strategic donor outreach plans.")

    csv_path = "data\raw\donor_features_v2.csv"
    if os.path.exists(csv_path):
        df_donors = pd.read_csv(csv_path)
        donor_ids = df_donors["donor_id"].astype(str).tolist() if "donor_id" in df_donors.columns else [str(i) for i in range(len(df_donors))]

        selected_id = st.selectbox("Select Donor ID for Evaluation:", donor_ids)

        if st.button("Generate AI Recommendation Plan", type="primary"):
            with st.spinner("DonorAgent evaluating ML predictions and formulating outreach strategy..."):
                if DonorAgent is not None:
                    try:
                        agent = DonorAgent()
                        query = f"Assess donor #{selected_id} and recommend an action based on their prediction score."
                        recommendation = agent.evaluate_donor(query)
                        st.markdown("### Strategic Outreach Plan")
                        st.markdown(recommendation)
                    except Exception as e:
                        st.error(f"Error invoking DonorAgent: {e}")
                else:
                    st.warning("`DonorAgent` module not found in `src/agents/donor_agent.py`.")


# ----------------------------------------------------------------------
# Page 6: ⚙️ Model Information (MLOps Metadata)
# ----------------------------------------------------------------------
elif page == "⚙️ Model Information":
    st.title("⚙️ Model Information & MLOps Registry")
    st.write("Live operational metrics and training parameters for the active XGBoost donor prediction model.")

    # High-level operational metrics
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Architecture", "XGBoost (Tuned)")
    m2.metric("Model Version", "v3.0.1")
    m3.metric("ROC-AUC Benchmark", "0.9666")
    m4.metric("PR-AUC Benchmark", "0.9623")

    st.markdown("---")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Model Artifact Metadata")
        metadata = {
            "Model Name": "XGBoost Classifier (Hyperparameter Tuned)",
            "Framework": "xgboost 2.x / scikit-learn",
            "Target Metric": "is_active_90d (Binary Classification)",
            "Training Date": "2026-08-15",
            "Cross-Validation": "5-Fold Stratified CV",
            "Total Features": 11,
            "Primary Artifact File": "best_donor_model_xgb.pkl",
            "Native Model File": "best_donor_model_xgb.json",
        }
        st.table(pd.DataFrame(list(metadata.items()), columns=["Attribute", "Value"]))

    with col2:
        st.subheader("Hyperparameters")
        hyperparams = {
            "eval_metric": "logloss",
            "learning_rate": 0.1,
            "max_depth": 5,
            "n_estimators": 100,
            "subsample": 0.8,
            "random_state": 42
        }
        st.json(hyperparams)

    st.markdown("---")
    st.subheader("Feature List (11 Features)")
    features_df = pd.DataFrame({
        "Feature Name": [
            "donation_count", "total_donations", "average_donation", "campaign_count",
            "days_since_first_donation", "donor_tenure_days", "avg_days_between_donations",
            "donations_last_365_days", "revenue_last_365_days", "frequency_velocity_365d",
            "monetary_velocity_365d"
        ],
        "Description": [
            "Lifetime total donation count", "Lifetime revenue sum ($)", "Average donation amount ($)",
            "Total distinct campaigns participated in", "Days elapsed since first gift",
            "Total donor tenure in days", "Average days between gifts (-1 for non-repeat)",
            "Donations in past 365 days", "Revenue generated in past 365 days",
            "Frequency velocity ratio (365d vs lifetime)", "Monetary velocity ratio (365d vs lifetime)"
        ]
    })
    st.table(features_df)