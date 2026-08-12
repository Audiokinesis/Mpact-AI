import random
from datetime import datetime, timedelta
import pandas as pd

# Configuration Parameters
NUM_DONORS = 10_000
NUM_DONATIONS = 50_000
NUM_CAMPAIGNS = 100

START_DATE = datetime(2021, 1, 1)
END_DATE = datetime(2026, 1, 1)


def get_random_date(start: datetime, end: datetime) -> datetime:
    delta_days = (end - start).days
    if delta_days <= 0:
        return start
    return start + timedelta(days=random.randint(0, delta_days))


def main():
    random.seed(42)

    # ---------------------------------------------------------
    # 1. Generate Campaigns (100)
    # ---------------------------------------------------------
    campaign_types = [
        "Direct Mail",
        "Email Blast",
        "Gala",
        "Peer-to-Peer",
        "Social Media",
        "Recurring Drive",
    ]
    causes = [
        "Clean Water Initiative",
        "Annual Fund",
        "Emergency Relief",
        "Youth Education",
        "Community Health",
        "Environmental Protection",
    ]

    campaigns = []
    for idx in range(1, NUM_CAMPAIGNS + 1):
        c_date = get_random_date(START_DATE, END_DATE)
        c_type = random.choice(campaign_types)
        c_cause = random.choice(causes)

        campaigns.append(
            {
                "campaign_id": f"CMP-{idx:04d}",
                "campaign_name": f"{c_cause} - {c_type} {c_date.year}",
                "campaign_type": c_type,
                "campaign_date": c_date.strftime("%Y-%m-%d"),
            }
        )

    df_campaigns = pd.DataFrame(campaigns)

    # ---------------------------------------------------------
    # 2. Generate Donors (10,000)
    # ---------------------------------------------------------
    age_groups = ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
    age_weights = [0.08, 0.20, 0.25, 0.22, 0.15, 0.10]

    locations = [
        "NY",
        "CA",
        "TX",
        "FL",
        "IL",
        "PA",
        "OH",
        "GA",
        "NC",
        "MI",
        "WA",
        "CO",
    ]

    donor_types = ["Individual", "Corporate", "Foundation", "Major Donor"]
    donor_type_weights = [0.85, 0.08, 0.04, 0.03]

    donors = []
    donor_since_lookup = {}

    for idx in range(1, NUM_DONORS + 1):
        donor_id = f"DNR-{idx:05d}"
        # donor_since set within the 5-year window
        since_date = get_random_date(
            START_DATE, END_DATE - timedelta(days=30)
        )
        donor_type = random.choices(donor_types, weights=donor_type_weights)[0]

        donor_since_lookup[donor_id] = (since_date, donor_type)

        donors.append(
            {
                "donor_id": donor_id,
                "age_group": random.choices(age_groups, weights=age_weights)[0],
                "location": random.choice(locations),
                "donor_since": since_date.strftime("%Y-%m-%d"),
                "donor_type": donor_type,
            }
        )

    df_donors = pd.DataFrame(donors)

    # ---------------------------------------------------------
    # 3. Generate Donations (50,000)
    # ---------------------------------------------------------
    donor_ids = list(donor_since_lookup.keys())
    campaign_ids = df_campaigns["campaign_id"].tolist()

    donations = []
    for idx in range(1, NUM_DONATIONS + 1):
        donor_id = random.choice(donor_ids)
        donor_since, donor_type = donor_since_lookup[donor_id]

        # Donation date must be >= donor_since date
        donation_date = get_random_date(donor_since, END_DATE)

        # Realistic tiered donation distribution based on donor profile
        if donor_type == "Major Donor":
            amount = round(random.uniform(2500.0, 50000.0), 2)
        elif donor_type == "Foundation":
            amount = round(random.uniform(1000.0, 25000.0), 2)
        elif donor_type == "Corporate":
            amount = round(random.uniform(500.0, 10000.0), 2)
        else:
            # Individual long-tail distribution
            tier = random.random()
            if tier < 0.70:
                amount = round(random.uniform(10.0, 100.0), 2)
            elif tier < 0.92:
                amount = round(random.uniform(100.0, 500.0), 2)
            else:
                amount = round(random.uniform(500.0, 2500.0), 2)

        donations.append(
            {
                "donation_id": f"DON-{idx:06d}",
                "donor_id": donor_id,
                "donation_date": donation_date.strftime("%Y-%m-%d"),
                "amount": amount,
                "campaign_id": random.choice(campaign_ids),
            }
        )

    df_donations = pd.DataFrame(donations)

    # ---------------------------------------------------------
    # Save to CSV Files
    # ---------------------------------------------------------
    df_donors.to_csv(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\donors.csv", index=False)
    df_donations.to_csv(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\donations.csv", index=False)
    df_campaigns.to_csv(r"C:\Users\mdesc\Documents\Projects\MpactAI\app\data\raw\campaigns.csv", index=False)

    print("Data generation complete:")
    print(f" - {len(df_donors):,} donors saved to donors.csv")
    print(f" - {len(df_donations):,} donations saved to donations.csv")
    print(f" - {len(df_campaigns):,} campaigns saved to campaigns.csv")


if __name__ == "__main__":
    main()