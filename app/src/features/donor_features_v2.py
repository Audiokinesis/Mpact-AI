import numpy as np
import pandas as pd


def generate_donor_features():
    print("Loading datasets...")
    # Adjust file paths if reading directly from local directories (e.g., r"C:\raw\donors.csv")
    df_donors = pd.read_csv(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\donors.csv")
    df_donations = pd.read_csv(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\donations.csv")

    # Date parsing
    df_donors["donor_since"] = pd.to_datetime(df_donors["donor_since"])
    df_donations["donation_date"] = pd.to_datetime(
        df_donations["donation_date"]
    )

    # Reference date anchored to the most recent transaction in the dataset
    reference_date = df_donations["donation_date"].max()
    print(f"Reference date set to: {reference_date.strftime('%Y-%m-%d')}")

    # ---------------------------------------------------------
    # 1. Base Aggregations & Recency Metrics
    # ---------------------------------------------------------
    base_agg = (
        df_donations.groupby("donor_id")
        .agg(
            donation_count=("donation_id", "count"),
            total_donations=("amount", "sum"),
            average_donation=("amount", "mean"),
            first_donation_date=("donation_date", "min"),
            last_donation_date=("donation_date", "max"),
            campaign_count=("campaign_id", "nunique"),
        )
        .reset_index()
    )

    # Calculate Recency & Tenure Features (in days)
    base_agg["days_since_last_donation"] = (
        reference_date - base_agg["last_donation_date"]
    ).dt.days
    base_agg["days_since_first_donation"] = (
        reference_date - base_agg["first_donation_date"]
    ).dt.days

    # ---------------------------------------------------------
    # 2. Inter-Donation Interval (Cadence)
    # ---------------------------------------------------------
    df_sorted = df_donations.sort_values(["donor_id", "donation_date"])
    df_sorted["days_since_prev_donation"] = df_sorted.groupby("donor_id")[
        "donation_date"
    ].diff().dt.days

    avg_intervals = (
        df_sorted.groupby("donor_id")["days_since_prev_donation"]
        .mean()
        .rename("avg_days_between_donations")
        .reset_index()
    )

    # ---------------------------------------------------------
    # 3. Rolling Window Aggregations (Counts & Revenue)
    # ---------------------------------------------------------
    d30_date = reference_date - pd.Timedelta(days=30)
    d90_date = reference_date - pd.Timedelta(days=90)
    d365_date = reference_date - pd.Timedelta(days=365)

    rolling_metrics = (
        df_donations.groupby("donor_id")
        .agg(
            donations_last_30_days=(
                "donation_id",
                lambda x: (
                    df_donations.loc[x.index, "donation_date"] >= d30_date
                ).sum(),
            ),
            revenue_last_30_days=(
                "amount",
                lambda x: df_donations.loc[
                    x.index[
                        df_donations.loc[x.index, "donation_date"] >= d30_date
                    ],
                    "amount",
                ].sum(),
            ),
            donations_last_90_days=(
                "donation_id",
                lambda x: (
                    df_donations.loc[x.index, "donation_date"] >= d90_date
                ).sum(),
            ),
            revenue_last_90_days=(
                "amount",
                lambda x: df_donations.loc[
                    x.index[
                        df_donations.loc[x.index, "donation_date"] >= d90_date
                    ],
                    "amount",
                ].sum(),
            ),
            donations_last_365_days=(
                "donation_id",
                lambda x: (
                    df_donations.loc[x.index, "donation_date"] >= d365_date
                ).sum(),
            ),
            revenue_last_365_days=(
                "amount",
                lambda x: df_donations.loc[
                    x.index[
                        df_donations.loc[x.index, "donation_date"] >= d365_date
                    ],
                    "amount",
                ].sum(),
            ),
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # 4. Merge Features & Calculate Velocity Metrics
    # ---------------------------------------------------------
    features = df_donors.merge(base_agg, on="donor_id", how="left")
    features = features.merge(avg_intervals, on="donor_id", how="left")
    features = features.merge(rolling_metrics, on="donor_id", how="left")

    # Donor tenure relative to dataset reference date
    features["donor_tenure_days"] = (
        reference_date - features["donor_since"]
    ).dt.days

    # --- Velocity Metrics ---
    # Frequency Velocity: Ratio of past 365-day donations to total lifetime count
    features["frequency_velocity_365d"] = (
        features["donations_last_365_days"] / features["donation_count"]
    )

    # Monetary Velocity: Proportion of lifetime value contributed in the last year
    features["monetary_velocity_365d"] = (
        features["revenue_last_365_days"] / features["total_donations"]
    )

    # Short-term Momentum Velocity: Annualized 90-day rate vs actual 365-day activity
    # (> 1.0 indicates acceleration; < 1.0 indicates deceleration)
    features["short_term_momentum"] = (
        features["donations_last_90_days"] * 4
    ) / (features["donations_last_365_days"] + 1)

    # Recency-to-Tenure Ratio: Scale recency against donor lifecycle stage
    features["recency_tenure_ratio"] = (
        features["days_since_last_donation"] / features["donor_tenure_days"]
    )

    # ---------------------------------------------------------
    # 5. Clean & Format Output
    # ---------------------------------------------------------
    numeric_fills = {
        "donation_count": 0,
        "total_donations": 0.0,
        "average_donation": 0.0,
        "campaign_count": 0,
        "donations_last_30_days": 0,
        "revenue_last_30_days": 0.0,
        "donations_last_90_days": 0,
        "revenue_last_90_days": 0.0,
        "donations_last_365_days": 0,
        "revenue_last_365_days": 0.0,
        "frequency_velocity_365d": 0.0,
        "monetary_velocity_365d": 0.0,
        "short_term_momentum": 0.0,
        "recency_tenure_ratio": 1.0,  # Max staleness for inactive donors
    }
    features.fillna(value=numeric_fills, inplace=True)

    # Drop raw timestamp columns prior to saving
    features.drop(
        columns=["first_donation_date", "last_donation_date", "donor_since"],
        inplace=True,
    )

    features.to_csv(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\donor_features_v2.csv", index=False)
    print(
        f"Feature engineering complete! {len(features):,} donors saved to 'donor_features.csv'"
    )


if __name__ == "__main__":
    generate_donor_features()