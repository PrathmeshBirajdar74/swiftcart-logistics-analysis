"""
week2_cleaning_pipeline.py
----------------------------
SwiftCart Logistics — Week 2: Data Collection, Cleaning, and Preprocessing

A runnable version of the cleaning pipeline described in the Week 2 report:
handles missing values, removes duplicates, flags outliers with a
zone/time-segmented IQR rule, standardizes inconsistent zone labels, and
normalizes numeric features ahead of clustering.

Usage:
    python week2_cleaning_pipeline.py
(Run data/generate_data.py first if data/delivery_records.csv doesn't exist.)
"""

import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "delivery_records.csv")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "delivery_records_cleaned.csv")

ZONE_MAPPING = {
    "N. Zone 1": "North Zone 1",
    "N. Zone 2": "North Zone 2",
    "N Zone 1": "North Zone 1",
    "N Zone 2": "North Zone 2",
}


def load_raw_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        sys.exit("delivery_records.csv not found in this folder. Run generate_data.py first.")
    return pd.read_csv(DATA_PATH, parse_dates=[
        "dispatch_time", "delivered_time", "promised_window_start", "promised_window_end"
    ])


def report_data_quality(df: pd.DataFrame) -> None:
    print("=== Initial Data Quality Report ===")
    print(f"Rows: {len(df)}")
    print("\nMissing values per column:")
    print(df.isna().sum()[df.isna().sum() > 0])
    print(f"\nDuplicate delivery_id rows: {df.duplicated('delivery_id').sum()}")
    print(f"Distinct zone_id values (before standardization): {df['zone_id'].nunique()}")


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["van_id", "dispatch_time"]).copy()

    # Impute missing delivery_outcome from the most recent scan per van (forward-fill)
    df["delivery_outcome"] = df.groupby("van_id")["delivery_outcome"].ffill()
    df["delivery_outcome"] = df["delivery_outcome"].fillna("UNKNOWN")

    # Impute missing package weight with the column median (robust to outliers)
    df["package_weight_kg"] = df["package_weight_kg"].fillna(df["package_weight_kg"].median())

    # Drop rows where promised_window_end is missing and unrecoverable
    before = len(df)
    df = df.dropna(subset=["promised_window_end"])
    print(f"Dropped {before - len(df)} rows with unrecoverable promised_window_end")

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = (
        df.sort_values("delivered_time")
        .drop_duplicates(subset="delivery_id", keep="last")
    )
    print(f"Removed {before - len(df)} duplicate delivery_id rows")
    return df


def flag_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["duration_min"] = (df["delivered_time"] - df["dispatch_time"]).dt.total_seconds() / 60
    df["dispatch_hour_bucket"] = df["dispatch_time"].dt.hour // 4

    grouped = df.groupby(["zone_id", "dispatch_hour_bucket"])["duration_min"]
    q1 = grouped.transform(lambda s: s.quantile(0.25))
    q3 = grouped.transform(lambda s: s.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["is_outlier"] = ~df["duration_min"].between(lower, upper)
    n_outliers = df["is_outlier"].sum()
    print(f"Flagged {n_outliers} delivery-duration outliers (zone/time-segmented IQR)")
    return df


def standardize_zone_names(df: pd.DataFrame) -> pd.DataFrame:
    before = df["zone_id"].nunique()
    df = df.copy()
    df["zone_id"] = df["zone_id"].str.strip().replace(ZONE_MAPPING)
    after = df["zone_id"].nunique()
    print(f"Standardized zone_id values: {before} distinct labels -> {after}")
    return df


def normalize_for_clustering(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.loc[~df["is_outlier"]].copy()
    zone_features = clean.groupby("zone_id").agg(
        avg_daily_volume=("delivery_id", "count"),
        avg_duration=("duration_min", "mean"),
        failed_attempt_rate=("first_attempt_success", lambda s: 1 - s.mean()),
    ).reset_index()

    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        zone_features[["avg_daily_volume", "avg_duration", "failed_attempt_rate"]]
    )
    zone_features[["avg_daily_volume_z", "avg_duration_z", "failed_attempt_rate_z"]] = scaled
    print("\n=== Zone features, standardized (ready for clustering) ===")
    print(zone_features.round(2))
    return zone_features


if __name__ == "__main__":
    df = load_raw_data()
    report_data_quality(df)

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = flag_outliers(df)
    df = standardize_zone_names(df)

    print("\n=== Post-Cleaning Data Quality Report ===")
    print(f"Rows remaining: {len(df)}")
    print(f"Outlier rows flagged (retained, not dropped): {df['is_outlier'].sum()}")
    print(f"Missing values remaining:\n{df.isna().sum()[df.isna().sum() > 0]}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned dataset to {OUTPUT_PATH}")

    normalize_for_clustering(df)
