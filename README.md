# SwiftCart Logistics — Data Analysis Internship Project

This repository contains the Python code behind my weekly YuvaIntern
logistics data analysis submissions. All weeks are built around a single
consistent fictional scenario, **SwiftCart Logistics** — a regional
e-commerce carrier working to improve last-mile delivery efficiency.

Each week has a corresponding Word document report (submitted separately
through the YuvaIntern portal) with the full write-up, methodology, and
explanations. This repo holds the actual runnable Python behind those
reports.

## Project structure

```
swiftcart-logistics-analysis/
├── data/
│   └── generate_data.py          # Shared simulated dataset generator (used by every week)
├── week1_strategic_planning/
│   └── week1_pipeline_demo.py    # EDA, zone clustering, ETA regression, route optimization demo
├── week2_data_cleaning/
│   └── week2_cleaning_pipeline.py# Missing values, duplicates, outlier detection, normalization
├── week3_visualization/
│   ├── week3_analysis.py         # EDA + 6 matplotlib/seaborn visualizations
│   └── charts/                   # Generated chart images (also embedded in the Week 3 report)
├── reports/                      # Word document reports submitted through YuvaIntern each week
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running each week

Generate the shared dataset first (all three weeks read from this):

```bash
python data/generate_data.py
```

Then run any week's script independently:

```bash
python week1_strategic_planning/week1_pipeline_demo.py
python week2_data_cleaning/week2_cleaning_pipeline.py
python week3_visualization/week3_analysis.py
```

## Week 1 — Strategic Planning and Data Exploration

Defines the SwiftCart scenario and KPIs (On-Time Delivery Rate, Cost per
Delivery, Average Delivery Time, First-Attempt Success Rate), then
demonstrates the proposed analytical approach end-to-end:

- Quick EDA (on-time proxy and duration by zone/hour)
- KMeans clustering of delivery zones by demand and risk profile
- A GradientBoostingRegressor predicting delivery duration (ETA)
- A small illustrative Google OR-Tools route optimization

## Week 2 — Data Collection, Cleaning, and Preprocessing

Takes the same simulated dataset (with deliberately injected missing
values, duplicates, and outliers) and runs it through a cleaning pipeline:

- Missing value imputation (forward-fill, median, targeted row drops)
- Duplicate removal
- Zone/time-segmented IQR outlier flagging
- Standardization of inconsistent zone naming
- Feature normalization ahead of clustering

Outputs a cleaned CSV (`week2_data_cleaning/delivery_records_cleaned.csv`).

## Week 3 — Advanced Data Analysis and Visualization

Simulates a larger delivery dataset and produces exploratory statistics
plus six visualizations exploring delivery time, cost, and their drivers:

1. Distribution of delivery times
2. Average transport cost by zone
3. Delivery distance vs. delivery time (scatter, by zone)
4. Transport cost spread by zone (boxplot)
5. Correlation heatmap across key variables
6. Rolling average transport cost over time

Chart images are saved to `week3_visualization/charts/` and are the same
images embedded in the Week 3 Word report.

## Notes

- All data is **simulated** for this internship exercise — SwiftCart
  Logistics is a fictional scenario, not a real company.
- `data/generate_data.py` uses a fixed random seed (42), so results are
  reproducible across runs and across weeks.
