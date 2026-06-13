# -*- coding: utf-8 -*-
# shap_analysis.py  -  SHAP Explainability for Decision Tree Model
# Run from project root:  python notebooks/shap_analysis.py

import subprocess
import sys
import os

# Force UTF-8 output on Windows so print() never throws UnicodeEncodeError
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Auto-install SHAP if not present ─────────────────────────────────────────
try:
    import shap
except ImportError:
    print("SHAP not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "shap", "-q"])
    import shap

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")          # file-only backend — no popup window
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────────────────────────────────────
# 0. Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_PATH  = BASE_DIR / "data" / "raw" / "Placement_Data_Full_Class.csv"
MODEL_PATH = BASE_DIR / "models" / "decision_tree_model.pkl"
CHARTS_DIR = BASE_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 55)
print("   SHAP Explainability  -  Decision Tree Model")
print("=" * 55)

# ─────────────────────────────────────────────────────────────────────────────
# 1. Load & Prepare Data  (same cleaning as training notebook)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading data...")

df = pd.read_csv(DATA_PATH)
df = df.drop(columns=["sl_no", "salary"])
df["status"] = df["status"].map({"Placed": 1, "Not Placed": 0})

X = df.drop(columns=["status"])
y = df["status"]

# Identify column types (avoids deprecated 'object' alias in newer pandas)
num_cols = X.select_dtypes(include="number").columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# Same split as training notebook  (random_state=42, stratify=y)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   [OK]  Test students : {X_test.shape[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Load Saved Model
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Loading saved model...")

pipeline     = joblib.load(MODEL_PATH)
preprocessor = pipeline.named_steps["preprocessor"]
classifier   = pipeline.named_steps["classifier"]

print(f"   [OK]  Model loaded  :  {MODEL_PATH.name}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Build Feature Names & Transform Test Data
# ─────────────────────────────────────────────────────────────────────────────
cat_feature_names = (
    preprocessor
    .named_transformers_["cat"]
    .get_feature_names_out(cat_cols)
    .tolist()
)
all_feature_names = num_cols + cat_feature_names

# Transform test data through preprocessor only (SMOTE is NOT applied here)
X_test_np = preprocessor.transform(X_test)

print(f"   [OK]  Total features after encoding : {len(all_feature_names)}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Compute SHAP Values
# ─────────────────────────────────────────────────────────────────────────────
print("\n[3/5] Computing SHAP values  (may take a few seconds)...")

explainer   = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_test_np)

# Handle both old and new SHAP API:
#   Old SHAP  -> shap_values is a list  [class_0_arr, class_1_arr]
#   New SHAP  -> shap_values is ndarray  shape (n, features, 2)
if isinstance(shap_values, list):
    shap_vals_placed = shap_values[1]
    base_val_placed  = float(explainer.expected_value[1])
elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
    shap_vals_placed = shap_values[:, :, 1]
    base_val_placed  = float(explainer.expected_value[1])
else:
    shap_vals_placed = shap_values
    base_val_placed  = float(np.atleast_1d(explainer.expected_value)[0])

print(f"   [OK]  SHAP values ready  |  base value = {base_val_placed:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 5a. Chart — Waterfall Plot  (explains ONE student's prediction)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Generating charts...")

SAMPLE_IDX = 0      # change this number to explain a different test student

explanation = shap.Explanation(
    values        = shap_vals_placed[SAMPLE_IDX],
    base_values   = base_val_placed,
    data          = X_test_np[SAMPLE_IDX],
    feature_names = all_feature_names,
)

shap.plots.waterfall(explanation, max_display=10, show=False)
plt.title(f"SHAP Waterfall - Student #{SAMPLE_IDX + 1}", pad=14, fontsize=13)
plt.tight_layout()
waterfall_path = CHARTS_DIR / "shap_waterfall.png"
plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   [OK]  Saved: charts/shap_waterfall.png")

# ─────────────────────────────────────────────────────────────────────────────
# 5b. Chart — Summary Plot  (feature importance across ALL test students)
# ─────────────────────────────────────────────────────────────────────────────
shap.summary_plot(
    shap_vals_placed,
    X_test_np,
    feature_names=all_feature_names,
    show=False,
)
plt.title("SHAP Summary - Feature Importance (All Test Students)", pad=14, fontsize=13)
plt.tight_layout()
summary_path = CHARTS_DIR / "shap_summary.png"
plt.savefig(summary_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"   [OK]  Saved: charts/shap_summary.png")

# ─────────────────────────────────────────────────────────────────────────────
# 6. Actionable Feedback  (first 3 test students)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Actionable Feedback:")

for idx in range(min(3, len(X_test))):
    pred  = pipeline.predict(X_test.iloc[[idx]])[0]
    label = "Placed [YES]" if pred == 1 else "Not Placed [NO]"

    print(f"\n  {'-' * 45}")
    print(f"  Student #{idx + 1}  ->  Prediction: {label}")
    print(f"  {'-' * 45}")

    sample_shap    = shap_vals_placed[idx]
    feature_impact = sorted(
        zip(all_feature_names, sample_shap),
        key=lambda x: x[1],
        reverse=True,
    )

    helpers = [(f, v) for f, v in feature_impact          if v > 0][:2]
    hurters = [(f, v) for f, v in reversed(feature_impact) if v < 0][:2]

    if helpers:
        print("  [+] Factors HELPING placement:")
        for fname, fval in helpers:
            print(f"      (+) {fname:<35}  +{fval:.4f}")
    else:
        print("  (no positive factors found)")

    if hurters:
        print("  [-] Factors HURTING placement:")
        for fname, fval in hurters:
            print(f"      (-) {fname:<35}  {fval:.4f}")
    else:
        print("  (no negative factors found)")

# ─────────────────────────────────────────────────────────────────────────────
# 7. Simple Bar Chart  (easy to read version of waterfall)
# ─────────────────────────────────────────────────────────────────────────────

from matplotlib.patches import Patch

sample_shap    = shap_vals_placed[SAMPLE_IDX]
feature_impact = list(zip(all_feature_names, sample_shap))

# Top 8 features by absolute impact
top8 = sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)[:8]
names  = [f[0] for f in top8]
values = [f[1] for f in top8]
colors = ["#27ae60" if v > 0 else "#e74c3c" for v in values]

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.barh(names, values, color=colors, height=0.55, edgecolor="white")

# Value labels on bars
for bar, val in zip(bars, values):
    xpos = val + 0.005 if val >= 0 else val - 0.005
    ha   = "left"      if val >= 0 else "right"
    ax.text(xpos, bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}", va="center", ha=ha, fontsize=10, fontweight="bold")

ax.axvline(x=0, color="black", linewidth=1)
ax.set_xlabel("Impact on Placement Prediction", fontsize=12)
ax.set_title(
    f"Student #{SAMPLE_IDX + 1}  -  Why this prediction?",
    fontsize=13, pad=12
)
ax.invert_yaxis()

legend_elements = [
    Patch(facecolor="#27ae60", label="Helped placement (+)"),
    Patch(facecolor="#e74c3c", label="Hurt placement  (-)"),
]
ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
plt.tight_layout()

simple_path = CHARTS_DIR / "shap_simple.png"
plt.savefig(simple_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n   [OK]  Saved: charts/shap_simple.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Plain-language Summary  (no jargon)
# ─────────────────────────────────────────────────────────────────────────────
pred  = pipeline.predict(X_test.iloc[[SAMPLE_IDX]])[0]
label = "PLACED" if pred == 1 else "NOT PLACED"

positives = [(f, v) for f, v in feature_impact if v > 0]
negatives = [(f, v) for f, v in feature_impact if v < 0]
positives.sort(key=lambda x: x[1], reverse=True)
negatives.sort(key=lambda x: x[1])

print(f"\n{'=' * 55}")
print(f"  Simple Explanation  -  Student #{SAMPLE_IDX + 1}")
print(f"{'=' * 55}")
print(f"  Result : {label}")
print()

if negatives:
    top_neg = negatives[0]
    print(f"  Main reason for this result:")
    print(f"    -> {top_neg[0]} was the biggest factor")
    print(f"       (impact: {top_neg[1]:.3f}  i.e. lowered placement chance)")

if positives:
    top_pos = positives[0]
    print(f"\n  Biggest strength:")
    print(f"    -> {top_pos[0]} helped the most")
    print(f"       (impact: +{top_pos[1]:.3f}  i.e. raised placement chance)")

print(f"\n  Feature ranking (most to least important):")
for i, (fname, fval) in enumerate(
    sorted(feature_impact, key=lambda x: abs(x[1]), reverse=True)[:5], 1
):
    sign = "(+)" if fval > 0 else "(-)"
    print(f"    {i}. {fname:<35} {sign}  {fval:+.4f}")

# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("   SHAP Analysis Complete!")
print("=" * 55)
print(f"\n   Charts saved in : {CHARTS_DIR}")
print("   -> shap_waterfall.png   (technical waterfall chart)")
print("   -> shap_summary.png     (all students, feature ranking)")
print("   -> shap_simple.png      (simple bar chart - easy to read)")
