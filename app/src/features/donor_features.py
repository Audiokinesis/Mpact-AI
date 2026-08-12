import numpy as np
import pandas as pd


def generate_donor_features():
    # 1. Load Data
    print("Loading CSV files...")
    df_donors = pd.read_csv(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\donors.csv")
    df_donations = pd.read_csv(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\donations.csv")

    # Convert date column
    df_donations["donation_date"] = pd.to_datetime(df_donations["donation_date"])

    # Establish reference date (latest donation date in dataset)
    reference_date = df_donations["donation_date"].max()
    print(f"Reference date set to: {reference_date.strftime('%Y-%m-%d')}")

    # 2. Overall Aggregations
    overall_features = (
        df_donations.groupby("donor_id")
        .agg(
            donation_count=("donation_id", "count"),
            total_donations=("amount", "sum"),
            average_donation=("amount", "mean"),
            latest_donation_date=("donation_date", "max"),
            campaign_count=("campaign_id", "nunique"),
        )
        .reset_index()
    )

    # Calculate days since last donation
    overall_features["days_since_last_donation"] = (
        reference_date - overall_features["latest_donation_date"]
    ).dt.days
    overall_features.drop(columns=["latest_donation_date"], inplace=True)

    # 3. Rolling Time Window Features (Count of donations)
    date_30d = reference_date - pd.Timedelta(days=30)
    date_90d = reference_date - pd.Timedelta(days=90)
    date_365d = reference_date - pd.Timedelta(days=365)

    d30_counts = (
        df_donations[df_donations["donation_date"] >= date_30d]
        .groupby("donor_id")["donation_id"]
        .count()
        .rename("donations_last_30_days")
    )

    d90_counts = (
        df_donations[df_donations["donation_date"] >= date_90d]
        .groupby("donor_id")["donation_id"]
        .count()
        .rename("donations_last_90_days")
    )

    d365_counts = (
        df_donations[df_donations["donation_date"] >= date_365d]
        .groupby("donor_id")["donation_id"]
        .count()
        .rename("donations_last_365_days")
    )

    # 4. Merge All Features Back to Master Donors Dataset
    df_features = df_donors[["donor_id"]].merge(
        overall_features, on="donor_id", how="left"
    )
    df_features = df_features.merge(d30_counts, on="donor_id", how="left")
    df_features = df_features.merge(d90_counts, on="donor_id", how="left")
    df_features = df_features.merge(d365_counts, on="donor_id", how="left")

    # 5. Handle Missing Values (Donors with no activity/donations)
    df_features["donation_count"] = (
        df_features["donation_count"].fillna(0).astype(int)
    )
    df_features["total_donations"] = (
        df_features["total_donations"].fillna(0.0).round(2)
    )
    df_features["average_donation"] = (
        df_features["average_donation"].fillna(0.0).round(2)
    )
    df_features["campaign_count"] = (
        df_features["campaign_count"].fillna(0).astype(int)
    )

    df_features["donations_last_30_days"] = (
        df_features["donations_last_30_days"].fillna(0).astype(int)
    )
    df_features["donations_last_90_days"] = (
        df_features["donations_last_90_days"].fillna(0).astype(int)
    )
    df_features["donations_last_365_days"] = (
        df_features["donations_last_365_days"].fillna(0).astype(int)
    )

    # Reorder columns explicitly
    column_order = [
        "donor_id",
        "donation_count",
        "total_donations",
        "average_donation",
        "days_since_last_donation",
        "donations_last_30_days",
        "donations_last_90_days",
        "donations_last_365_days",
        "campaign_count",
    ]
    df_features = df_features[column_order]

    # 6. Export Output
    output_filename = "donor_features.csv"
    df_features.to_csv(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\\" + output_filename, index=False)
    print(
        f"Features successfully generated for {len(df_features):,} donors -> '{output_filename}'"
    )

    # Quick preview
    print("\nFeature Output Sample:")
    print(df_features.head())


if __name__ == "__main__":
    generate_donor_features()