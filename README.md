# NFL Prospect Scouting Engine

A modular Python engine that ranks NFL skill-position rookies using **process-oriented** EPA metrics rather than raw counting stats.

## Metrics

| Metric | Description |
|---|---|
| **Success Rate** | % of plays with positive EPA |
| **Explosiveness** | Mean EPA on 20+ yard plays |
| **Reliability** | Reception rate / YPC in 4th quarter, score within 8 |
| **CSS Score** | Weighted composite (40 / 35 / 25) |

## Setup

```bash
# 1. Create & activate virtual environment
virtualenv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main.py
```

Outputs are saved to `outputs/css_scatter.png`.

## Project Structure

```
NFLScouter/
├── src/
│   ├── ingestion.py      # nflreadpy data loading
│   ├── features.py       # EPA feature engineering
│   ├── scoring.py        # MinMaxScaler + CSS ranking
│   ├── visualization.py  # Seaborn scatter plot
│   └── pipeline.py       # End-to-end orchestration
├── tests/
│   ├── test_features.py
│   └── test_scoring.py
├── outputs/              # Generated plots (gitignored)
├── main.py
└── requirements.txt
```

## Configuration

Edit `main.py` to adjust:
- `SEASONS` — default `[2023, 2024, 2025]`
- `MIN_SNAPS` — default `50`
- `WEIGHTS` — default `{"success_rate": 0.40, "explosiveness": 0.35, "reliability": 0.25}`

## Caching

Speed up repeated runs by caching nflreadpy downloads to disk:

```bash
export NFLREADPY_CACHE=filesystem
python main.py
```

## ML Clustering

The final `css_df` returned by `pipeline.run()` is tidy and ready for K-Means:

```python
from sklearn.cluster import KMeans
from src.pipeline import run

css_df = run()
X = css_df[["success_rate_scaled", "explosiveness_scaled", "reliability_scaled"]]
kmeans = KMeans(n_clusters=4, random_state=42).fit(X)
css_df["cluster"] = kmeans.labels_
```
