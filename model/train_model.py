"""
train_model.py
---------------
Loads data/hdi_dataset.csv, performs light EDA (saved as PNGs for the
frontend/report), trains a RandomForestClassifier to predict the HDI
category (Very High / High / Medium / Low) from the four raw indicators,
evaluates it, and serializes the trained model + scaler with joblib so the
Flask backend can load them at request time.
"""

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

sns.set_theme(style="whitegrid")

DATA_PATH = "/home/claude/hdi_project/data/hdi_dataset.csv"
PLOTS_DIR = "/home/claude/hdi_project/static/plots"
MODEL_DIR = "/home/claude/hdi_project/model"

FEATURES = [
    "life_expectancy",
    "mean_years_schooling",
    "expected_years_schooling",
    "gni_per_capita",
]
TARGET = "hdi_category"
CATEGORY_ORDER = ["Low", "Medium", "High", "Very High"]

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# ---------------------------------------------------------------------------
# 2. EDA plots
# ---------------------------------------------------------------------------
# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=TARGET, order=CATEGORY_ORDER, palette="viridis")
plt.title("HDI Category Distribution")
plt.xlabel("HDI Category")
plt.ylabel("Number of Countries")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/class_distribution.png", dpi=120)
plt.close()

# Correlation heatmap
plt.figure(figsize=(6, 5))
corr = df[FEATURES + ["hdi_score"]].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/correlation_heatmap.png", dpi=120)
plt.close()

# Feature distributions by category
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
for ax, feat in zip(axes.flat, FEATURES):
    sns.boxplot(data=df, x=TARGET, y=feat, order=CATEGORY_ORDER, ax=ax, palette="magma")
    ax.set_title(feat.replace("_", " ").title())
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/feature_boxplots.png", dpi=120)
plt.close()

print("EDA plots saved to", PLOTS_DIR)

# ---------------------------------------------------------------------------
# 3. Prepare train/test data
# ---------------------------------------------------------------------------
X = df[FEATURES].values
y = df[TARGET].values

le = LabelEncoder()
le.fit(CATEGORY_ORDER)
y_encoded = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Train model
# ---------------------------------------------------------------------------
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    min_samples_leaf=3,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
print(f"\nTest Accuracy: {acc:.4f}\n")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=le.classes_, yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Accuracy: {acc:.2%})")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/confusion_matrix.png", dpi=120)
plt.close()

# Feature importance
importances = model.feature_importances_
plt.figure(figsize=(6, 4))
sns.barplot(x=importances, y=[f.replace("_", " ").title() for f in FEATURES], palette="crest")
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{PLOTS_DIR}/feature_importance.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 6. Save model artifacts
# ---------------------------------------------------------------------------
joblib.dump(model, f"{MODEL_DIR}/hdi_model.joblib")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.joblib")
joblib.dump(le, f"{MODEL_DIR}/label_encoder.joblib")

print("\nModel, scaler, and label encoder saved to", MODEL_DIR)
