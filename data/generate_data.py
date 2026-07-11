"""
generate_data.py
-----------------
Simulates a realistic country-level dataset with the four raw indicators used
by the UNDP to compute the Human Development Index:

    1. Life Expectancy at Birth (years)
    2. Mean Years of Schooling (years)
    3. Expected Years of Schooling (years)
    4. GNI per Capita, PPP (constant 2017 international $)

The actual UNDP HDI formula is used to compute a continuous HDI score, which
is then bucketed into the four official tiers:

    Very High  : HDI >= 0.800
    High       : 0.700 <= HDI < 0.800
    Medium     : 0.550 <= HDI < 0.700
    Low        : HDI <  0.550

The output is data/hdi_dataset.csv, used to train the classifier in
model/train_model.py.
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N_SAMPLES = 3000

# ---------------------------------------------------------------------------
# 1. Simulate correlated "country archetypes" so the data looks realistic
#    (poorer countries tend to have lower life expectancy & schooling too)
# ---------------------------------------------------------------------------
# Draw an underlying "development latent factor" for each synthetic country
development_factor = np.random.beta(a=2.0, b=2.0, size=N_SAMPLES)  # 0 (low) -> 1 (very high)

# Life expectancy: 45 - 85 years, correlated with development factor
life_expectancy = 45 + development_factor * 40 + np.random.normal(0, 3, N_SAMPLES)
life_expectancy = np.clip(life_expectancy, 40, 88)

# Mean years of schooling: 1 - 14 years
mean_schooling = 1 + development_factor * 12 + np.random.normal(0, 1.2, N_SAMPLES)
mean_schooling = np.clip(mean_schooling, 0.2, 14.5)

# Expected years of schooling: 4 - 20 years
expected_schooling = 5 + development_factor * 14 + np.random.normal(0, 1.5, N_SAMPLES)
expected_schooling = np.clip(expected_schooling, 3, 21)

# GNI per capita (PPP $): log-normal, correlated with development factor
log_gni = 6.0 + development_factor * 5.2 + np.random.normal(0, 0.5, N_SAMPLES)
gni_per_capita = np.exp(log_gni)
gni_per_capita = np.clip(gni_per_capita, 300, 150000)

# ---------------------------------------------------------------------------
# 2. Compute the official UNDP HDI (Technical Notes, min-max normalization)
# ---------------------------------------------------------------------------
LE_MIN, LE_MAX = 20, 85
MYS_MAX = 15
EYS_MAX = 18
GNI_MIN, GNI_MAX = 100, 75000

life_expectancy_index = (life_expectancy - LE_MIN) / (LE_MAX - LE_MIN)
mean_school_index = mean_schooling / MYS_MAX
expected_school_index = expected_schooling / EYS_MAX
education_index = (mean_school_index + expected_school_index) / 2
income_index = (np.log(gni_per_capita) - np.log(GNI_MIN)) / (np.log(GNI_MAX) - np.log(GNI_MIN))

# clip indices into [0, 1] as UNDP does
life_expectancy_index = np.clip(life_expectancy_index, 0, 1)
education_index = np.clip(education_index, 0, 1)
income_index = np.clip(income_index, 0, 1)

hdi = (life_expectancy_index * education_index * income_index) ** (1 / 3)

# ---------------------------------------------------------------------------
# 3. Bucket into the four official HDI tiers
# ---------------------------------------------------------------------------
def bucket_hdi(score):
    if score >= 0.800:
        return "Very High"
    elif score >= 0.700:
        return "High"
    elif score >= 0.550:
        return "Medium"
    else:
        return "Low"

hdi_category = np.array([bucket_hdi(s) for s in hdi])

# ---------------------------------------------------------------------------
# 4. Assemble & save the dataset
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "life_expectancy": life_expectancy.round(1),
    "mean_years_schooling": mean_schooling.round(1),
    "expected_years_schooling": expected_schooling.round(1),
    "gni_per_capita": gni_per_capita.round(0),
    "hdi_score": hdi.round(4),
    "hdi_category": hdi_category,
})

df.to_csv("/home/claude/hdi_project/data/hdi_dataset.csv", index=False)
print("Saved dataset with shape:", df.shape)
print(df["hdi_category"].value_counts())
print(df.head())
