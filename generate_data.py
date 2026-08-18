"""
generate_data.py
-----------------
Simulates SwiftCart Logistics' delivery dataset.

This single generator is shared across Week 1 (strategic planning),
Week 2 (cleaning/preprocessing), and Week 3 (advanced analysis) so that
every week's script works from a consistent, reproducible dataset.

Run directly to write data/delivery_records.csv:
    python generate_data.py
"""

import os
import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_delivery_data(n_rows: int = 2000, inject_dirty_data: bool = True) -> pd.DataFrame:
    """Generate a simulated last-mile delivery dataset for SwiftCart Logistics.

    Parameters
    ----------
    n_rows : int
        Number of delivery records to simulate.
    inject_dirty_data : bool
        If True, deliberately introduces missing values, duplicate rows,
        and a few outlier durations so the Week 2 cleaning pipeline has
        real issues to detect and fix.
    """
    rng = np.random.default_rng(RANDOM_SEED)

    zones = ["North Zone 1", "North Zone 2", "South Zone 1",
             "South Zone 2", "East Zone 1", "West Zone 1"]
    zone_base_distance = {"North Zone 1": 6, "North Zone 2": 9, "South Zone 1": 5,
                           "South Zone 2": 12, "East Zone 1": 8, "West Zone 1": 14}
    zone_congestion = {"North Zone 1": 1.0, "North Zone 2": 1.3, "South Zone 1": 0.8,
                        "South Zone 2": 1.6, "East Zone 1": 1.1, "West Zone 1": 1.9}
    van_ids = [f"VAN-{i:02d}" for i in range(1, 61)]
    depot_ids = ["DEPOT-A", "DEPOT-B", "DEPOT-C"]

    dates = pd.date_range("2026-05-04", periods=70, freq="D")

    rows = []
    for i in range(n_rows):
        zone = rng.choice(zones)
        date = pd.Timestamp(rng.choice(dates))
        dispatch_hour = int(rng.integers(7, 19))
        dispatch_time = date + pd.Timedelta(hours=dispatch_hour, minutes=int(rng.integers(0, 60)))

        distance = max(0.5, rng.normal(zone_base_distance[zone], 2.5))
        congestion = zone_congestion[zone] * (1 + 0.15 * np.sin(date.dayofweek))
        base_speed = 28  # km/h effective urban delivery speed
        duration_min = max(4, (distance / base_speed) * 60 * congestion + rng.normal(0, 4))
        delivered_time = dispatch_time + pd.Timedelta(minutes=duration_min)

        promised_start = dispatch_time.normalize() + pd.Timedelta(hours=9)
        promised_end = dispatch_time.normalize() + pd.Timedelta(hours=dispatch_hour + 2)

        weight = max(0.2, rng.gamma(2.0, 2.0))
        volume = int(rng.poisson(6) + 1)
        cost = max(15, 45 + 3.2 * distance + 1.8 * weight + rng.normal(0, 6))
        first_attempt = bool(rng.random() > (0.06 + 0.05 * (congestion - 1)))
        outcome = "DELIVERED" if first_attempt else "REDELIVERY_REQUIRED"

        rows.append({
            "delivery_id": f"D{i:05d}",
            "van_id": rng.choice(van_ids),
            "depot_id": rng.choice(depot_ids),
            "zone_id": zone,
            "dispatch_time": dispatch_time,
            "delivered_time": delivered_time,
            "promised_window_start": promised_start,
            "promised_window_end": promised_end,
            "distance_km": round(distance, 2),
            "package_weight_kg": round(weight, 2),
            "shipment_volume": volume,
            "transport_cost": round(cost, 2),
            "first_attempt_success": first_attempt,
            "delivery_outcome": outcome,
        })

    df = pd.DataFrame(rows)

    if inject_dirty_data:
        # Missing values (mirrors the real-world gaps described in Week 2)
        df.loc[df.sample(15, random_state=1).index, "package_weight_kg"] = np.nan
        df.loc[df.sample(10, random_state=2).index, "delivery_outcome"] = np.nan
        df.loc[df.sample(6, random_state=3).index, "promised_window_end"] = pd.NaT

        # A few duplicate delivery_id rows (simulates a driver re-scan)
        dup_rows = df.sample(8, random_state=4).copy()
        df = pd.concat([df, dup_rows], ignore_index=True)

        # A handful of implausible durations (data-entry errors / breakdowns)
        outlier_idx = df.sample(5, random_state=5).index
        df.loc[outlier_idx, "delivered_time"] = df.loc[outlier_idx, "dispatch_time"] + pd.Timedelta(minutes=1)
        breakdown_idx = df.sample(4, random_state=6).index
        df.loc[breakdown_idx, "delivered_time"] = df.loc[breakdown_idx, "dispatch_time"] + pd.Timedelta(minutes=420)

        # Inconsistent zone naming
        messy_idx = df.sample(20, random_state=7).index
        df.loc[messy_idx, "zone_id"] = df.loc[messy_idx, "zone_id"].str.replace(
            "North", "N.", regex=False
        )

    return df.reset_index(drop=True)


if __name__ == "__main__":
    df = generate_delivery_data()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(out_dir, "delivery_records.csv")
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")
    print(df.head())
