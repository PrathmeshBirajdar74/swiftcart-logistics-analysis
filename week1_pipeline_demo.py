"""
week1_pipeline_demo.py
-----------------------
SwiftCart Logistics — Week 1: Strategic Planning and Data Exploration

A runnable demonstration of the analytical pipeline proposed in the Week 1
report: load data, run quick EDA, segment delivery zones with clustering,
predict delivery duration (ETA) with regression, and generate a simple
optimized delivery route with Google OR-Tools.

Usage:
    python week1_pipeline_demo.py
(Run data/generate_data.py first if data/delivery_records.csv doesn't exist.)
"""

import os
import sys
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(REPO_ROOT, "data", "delivery_records.csv")


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        sys.exit("data/delivery_records.csv not found. Run `python data/generate_data.py` first.")
    df = pd.read_csv(DATA_PATH, parse_dates=["dispatch_time", "delivered_time"])
    df["duration_min"] = (df["delivered_time"] - df["dispatch_time"]).dt.total_seconds() / 60
    return df


def quick_eda(df: pd.DataFrame) -> None:
    print("\n=== Quick EDA ===")
    print(f"Rows: {len(df)}")
    print("\nOn-time proxy — average duration by zone (minutes):")
    print(df.groupby("zone_id")["duration_min"].mean().round(1).sort_values())
    print("\nAverage duration by dispatch hour:")
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    print(df.groupby("dispatch_hour")["duration_min"].mean().round(1))


def zone_clustering(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Zone Clustering (KMeans) ===")
    zone_features = df.groupby("zone_id").agg(
        avg_daily_volume=("delivery_id", "count"),
        avg_duration=("duration_min", "mean"),
        failed_attempt_rate=("first_attempt_success", lambda s: 1 - s.mean()),
    ).reset_index()

    X = StandardScaler().fit_transform(
        zone_features[["avg_daily_volume", "avg_duration", "failed_attempt_rate"]]
    )
    kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
    zone_features["cluster"] = kmeans.fit_predict(X)
    print(zone_features.sort_values("cluster"))
    return zone_features


def eta_regression(df: pd.DataFrame) -> None:
    print("\n=== ETA Regression (GradientBoostingRegressor) ===")
    df = df.dropna(subset=["duration_min", "distance_km"]).copy()
    df["dispatch_hour"] = df["dispatch_time"].dt.hour
    df["zone_code"] = df["zone_id"].astype("category").cat.codes

    features = ["distance_km", "dispatch_hour", "zone_code"]
    X = df[features]
    y = df["duration_min"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"Mean Absolute Error on held-out test set: {mae:.2f} minutes")
    print("Feature importances:", dict(zip(features, model.feature_importances_.round(3))))


def simple_route_optimization(df: pd.DataFrame) -> None:
    """
    Small illustrative VRP: pick one van's stops for a single day and solve
    a shortest-route ordering with OR-Tools. Falls back to a message if
    OR-Tools isn't installed, since it's an optional heavier dependency.
    """
    print("\n=== Route Optimization (OR-Tools) ===")
    try:
        from ortools.constraint_solver import pywrapcp, routing_enums_pb2
    except ImportError:
        print("Google OR-Tools not installed. Install with: pip install ortools")
        return

    # Take one van's stops on one day as a small illustrative example
    sample = df.dropna(subset=["distance_km"]).copy()
    sample["date"] = sample["dispatch_time"].dt.date
    top = (sample.groupby(["van_id", "date"]).size()
           .sort_values(ascending=False).index[0])
    van_id, date = top
    stops = sample[(sample["van_id"] == van_id) & (sample["date"] == date)].head(8)

    n = len(stops) + 1  # +1 for depot
    # Build a simple distance matrix using each stop's distance_km from depot
    # as a stand-in for pairwise distances (illustrative, not geocoded).
    dist_from_depot = [0] + stops["distance_km"].round(1).tolist()
    matrix = [[abs(dist_from_depot[i] - dist_from_depot[j]) + (dist_from_depot[j] if i == 0 else 0)
               for j in range(n)] for i in range(n)]

    manager = pywrapcp.RoutingIndexManager(n, 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(matrix[from_node][to_node] * 100)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    solution = routing.SolveWithParameters(search_parameters)

    if solution:
        index = routing.Start(0)
        route = []
        while not routing.IsEnd(index):
            route.append(manager.IndexToNode(index))
            index = solution.Value(routing.NextVar(index))
        route.append(manager.IndexToNode(index))
        print(f"Optimized stop order for {van_id} on {date}: {route}")
    else:
        print("No solution found.")


if __name__ == "__main__":
    df = load_data()
    quick_eda(df)
    zone_clustering(df)
    eta_regression(df)
    simple_route_optimization(df)
