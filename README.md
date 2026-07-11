# Meridian — Human Development Index Estimator, **Live demo:** https://hdi-project-n2p8.onrender.com

A full-stack machine learning web app that classifies a country into one of
the four official HDI tiers — **Low, Medium, High, Very High** — from its
raw development indicators.

## What it does

Given four indicators for a country:

- Life expectancy at birth (years)
- Mean years of schooling (years, adults 25+)
- Expected years of schooling (years, school-entry age)
- GNI per capita, PPP (international $)

...the app returns:

- A predicted HDI **category** (Low / Medium / High / Very High)
- The model's **confidence** and full probability breakdown across all tiers
- The **continuous HDI score** (0–1), computed with the same UNDP-style
  formula used to generate training labels

## Tech stack

| Layer      | Tools |
|------------|-------|
| Data / ML  | Python, Pandas, NumPy, scikit-learn, joblib |
| Visualization | Matplotlib, Seaborn |
| Backend    | Flask (REST API + server-rendered page) |
| Frontend   | HTML, CSS, vanilla JavaScript |

## Project structure

```
hdi_project/
├── app.py                     # Flask backend (routes + prediction API)
├── requirements.txt
├── data/
│   ├── generate_data.py       # Builds the synthetic training dataset
│   └── hdi_dataset.csv        # Generated dataset (3,000 rows)
├── model/
│   ├── train_model.py         # EDA + trains & saves the RandomForest
│   ├── hdi_model.joblib        # Trained classifier
│   ├── scaler.joblib            # Feature scaler
│   └── label_encoder.joblib     # Category label encoder
├── templates/
│   └── index.html             # Frontend page
└── static/
    ├── css/style.css
    ├── js/script.js
    └── plots/                 # EDA & evaluation charts (PNG)
```

## How the model was built

1. **`data/generate_data.py`** simulates 3,000 realistic country profiles.
   Each indicator is drawn from a correlated "development factor" so that,
   like real countries, higher life expectancy tends to accompany higher
   schooling and income. The official **UNDP HDI formula** (min–max
   normalization of each dimension, then geometric mean) is applied to
   compute a continuous HDI score, which is bucketed into the four tiers
   using the UNDP's published thresholds (≥0.800 Very High, 0.700–0.799
   High, 0.550–0.699 Medium, <0.550 Low).

2. **`model/train_model.py`** loads that dataset, produces EDA plots
   (class balance, correlation heatmap, feature boxplots), trains a
   `RandomForestClassifier` (300 trees, balanced class weights) on
   standardized features, evaluates it (confusion matrix, classification
   report — ~95% held-out accuracy), and serializes the model, scaler, and
   label encoder with `joblib`.

3. **`app.py`** loads those artifacts once at startup and exposes:
   - `GET /` — renders the frontend
   - `POST /predict` — accepts JSON with the four indicators, returns the
     predicted tier, confidence, per-tier probabilities, and computed HDI
     score
   - `GET /api/sample/<tier>` — returns example input values for a tier
     (used by the "Load an example" buttons)

## Running it locally

```bash
cd hdi_project
pip install -r requirements.txt

# (optional — pre-trained artifacts are already included)
python data/generate_data.py
python model/train_model.py

python app.py
```

Then open **http://localhost:5000** in your browser.

## Retraining with real data

To use real UNDP figures instead of the synthetic dataset, replace
`data/hdi_dataset.csv` with a CSV that has these columns, then re-run
`model/train_model.py`:

```
life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita, hdi_score, hdi_category
```

## Notes

- The figures produced by this app are **indicative estimates from a
  demonstration model**, not official UNDP statistics.
- The synthetic dataset approximates real-world correlations but is not
  drawn from actual country data — swap in the UNDP's Human Development
  Report tables for production use.
  Add live demo link
