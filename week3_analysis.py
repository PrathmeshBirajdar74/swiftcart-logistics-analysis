
"""
week3_analysis.py
-------------------
SwiftCart Logistics — Week 3: Advanced Data Analysis and Visualization

Simulates a delivery dataset, runs exploratory data analysis, and produces
six charts (saved to ./charts) used in the Week 3 report.

Usage:
    python week3_analysis.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", context="notebook")
PALETTE = ["#1F4E5F", "#3E8E9E", "#7FB3BF", "#C9622D", "#44546A"]
sns.set_palette(PALETTE)

rng = np.random.default_rng(42)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "charts")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# 1. Simulate the dataset
# ---------------------------------------------------------------
n = 2000
zones = ["North Zone 1", "North Zone 2", "South Zone 1", "South Zone 2", "East Zone 1", "West Zone 1"]
zone_base_distance = {"North Zone 1": 6, "North Zone 2": 9, "South Zone 1": 5,
                       "South Zone 2": 12, "East Zone 1": 8, "West Zone 1": 14}
zone_congestion = {"North Zone 1": 1.0, "North Zone 2": 1.3, "South Zone 1": 0.8,
                    "South Zone 2": 1.6, "East Zone 1": 1.1, "West Zone 1": 1.9}

dates = pd.date_range("2026-05-04", periods=70, freq="D")  # 10 weeks

rows = []
for i in range(n):
    zone = rng.choice(zones)
    date = rng.choice(dates)
    distance = max(0.5, rng.normal(zone_base_distance[zone], 2.5))
    congestion = zone_congestion[zone] * (1 + 0.15 * np.sin(pd.Timestamp(date).dayofweek))
    base_speed = 28  # km/h effective urban delivery speed
    delivery_time = (distance / base_speed) * 60 * congestion + rng.normal(0, 4)
    delivery_time = max(4, delivery_time)
    weight = max(0.2, rng.gamma(2.0, 2.0))
    volume = rng.poisson(6) + 1
    cost = 45 + 3.2 * distance + 1.8 * weight + rng.normal(0, 6)
    cost = max(15, cost)
    first_attempt = rng.random() > (0.06 + 0.05 * (congestion - 1))
    rows.append([f"D{i:05d}", date, zone, round(distance, 2), round(delivery_time, 1),
                 round(weight, 2), volume, round(cost, 2), first_attempt])

df = pd.DataFrame(rows, columns=["delivery_id", "date", "zone", "distance_km",
                                  "delivery_time_min", "package_weight_kg",
                                  "shipment_volume", "transport_cost", "first_attempt_success"])

# inject a few missing values and outliers to mirror Week 2's cleaning context (already resolved here)
df.loc[df.sample(15, random_state=1).index, "package_weight_kg"] = np.nan
df["package_weight_kg"] = df["package_weight_kg"].fillna(df["package_weight_kg"].median())

df.to_csv(os.path.join(SCRIPT_DIR, "swiftcart_week3_dataset.csv"), index=False)
numeric_cols = ["distance_km", "delivery_time_min", "package_weight_kg", "shipment_volume", "transport_cost"]
print(df[numeric_cols].describe())
print(df.groupby("zone")["delivery_time_min"].mean())

# ---------------------------------------------------------------
# 2. EDA — summary stats + correlation
# ---------------------------------------------------------------
summary = df[["distance_km", "delivery_time_min", "package_weight_kg",
              "shipment_volume", "transport_cost"]].describe().T
summary.to_csv(os.path.join(SCRIPT_DIR, "summary_stats.csv"))

corr = df[["distance_km", "delivery_time_min", "package_weight_kg",
           "shipment_volume", "transport_cost"]].corr()
print(corr)

# ---------------------------------------------------------------
# 3. Visualizations
# ---------------------------------------------------------------

# Chart 1: Distribution of delivery times
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.histplot(df["delivery_time_min"], bins=30, kde=True, color=PALETTE[0], ax=ax)
ax.set_title("Distribution of Delivery Times", fontsize=13, weight="bold")
ax.set_xlabel("Delivery Time (minutes)")
ax.set_ylabel("Number of Deliveries")
fig.tight_layout()
fig.savefig(f"{OUT}/01_delivery_time_distribution.png", dpi=150)
plt.close(fig)

# Chart 2: Average transport cost by zone (bar)
zone_cost = df.groupby("zone")["transport_cost"].mean().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
sns.barplot(x=zone_cost.values, y=zone_cost.index, ax=ax, palette=PALETTE)
ax.set_title("Average Transport Cost by Zone", fontsize=13, weight="bold")
ax.set_xlabel("Average Cost (\u20b9)")
ax.set_ylabel("")
fig.tight_layout()
fig.savefig(f"{OUT}/02_avg_cost_by_zone.png", dpi=150)
plt.close(fig)

# Chart 3: Distance vs delivery time scatter, colored by zone
fig, ax = plt.subplots(figsize=(7.5, 5))
sns.scatterplot(data=df, x="distance_km", y="delivery_time_min", hue="zone",
                 palette=PALETTE + ["#8C8C8C"], alpha=0.6, s=35, ax=ax)
ax.set_title("Delivery Distance vs. Delivery Time", fontsize=13, weight="bold")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Delivery Time (minutes)")
ax.legend(title="Zone", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/03_distance_vs_time_scatter.png", dpi=150)
plt.close(fig)

# Chart 4: Boxplot of transport cost by zone
fig, ax = plt.subplots(figsize=(7.5, 4.5))
order = df.groupby("zone")["transport_cost"].median().sort_values().index
sns.boxplot(data=df, x="zone", y="transport_cost", order=order, palette=PALETTE, ax=ax)
ax.set_title("Transport Cost Spread by Zone", fontsize=13, weight="bold")
ax.set_xlabel("")
ax.set_ylabel("Transport Cost (\u20b9)")
ax.tick_params(axis="x", rotation=25)
fig.tight_layout()
fig.savefig(f"{OUT}/04_cost_boxplot_by_zone.png", dpi=150)
plt.close(fig)

# Chart 5: Correlation heatmap
fig, ax = plt.subplots(figsize=(6.5, 5.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="crest", square=True,
            cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title("Correlation Between Key Variables", fontsize=13, weight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/05_correlation_heatmap.png", dpi=150)
plt.close(fig)

# Chart 6: Daily average cost trend over time
daily = df.groupby("date")["transport_cost"].mean().rolling(5).mean()
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(daily.index, daily.values, color=PALETTE[3], linewidth=2)
ax.set_title("5-Day Rolling Average Transport Cost Over Time", fontsize=13, weight="bold")
ax.set_xlabel("Date")
ax.set_ylabel("Avg. Transport Cost (\u20b9)")
fig.autofmt_xdate()
fig.tight_layout()
fig.savefig(f"{OUT}/06_cost_trend_over_time.png", dpi=150)
plt.close(fig)

print("Charts written to", OUT)
print(summary)
