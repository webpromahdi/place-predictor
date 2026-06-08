"""
SVM Placement Prediction — Placement_Data_Full_Class.csv
=========================================================
Requirements covered:
  1. Data cleaning & leakage prevention (drop sl_no, salary; encode target)
  2. ColumnTransformer preprocessing pipeline (StandardScaler + OneHotEncoder)
  3. imbalanced-learn Pipeline with SMOTE + SVC
  4. Model evaluation (Accuracy, Precision, Recall, F1-Score)
  5. EDA bar charts saved as PNG files
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend → safe for saving files
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from imblearn.pipeline import Pipeline          # ← imbalanced-learn Pipeline (NOT sklearn)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# 0. CONFIGURATION
# ─────────────────────────────────────────────
DATA_PATH   = "Placement_Data_Full_Class.csv"
TEST_SIZE   = 0.20
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# 1. LOAD & CLEAN DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("  STUDENT PLACEMENT — SVM CLASSIFICATION")
print("=" * 60)

df = pd.read_csv(DATA_PATH)
print(f"\n[INFO] Dataset loaded  →  {df.shape[0]} rows × {df.shape[1]} cols")

# Drop ID and post-event leakage column
df.drop(columns=["sl_no", "salary"], inplace=True)
print("[INFO] Dropped 'sl_no' (ID) and 'salary' (data leakage) columns")

# Handle missing values
missing_before = df.isnull().sum().sum()
df.dropna(inplace=True)
print(f"[INFO] Missing values removed  →  {missing_before} NaN(s) dropped "
      f"({df.shape[0]} rows remaining)")

# Encode target: Placed → 1, Not Placed → 0
le = LabelEncoder()
df["status"] = le.fit_transform(df["status"])   # Not Placed=0, Placed=1
print(f"[INFO] Target encoded  →  classes: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Class distribution
counts = df["status"].value_counts()
print(f"\n[INFO] Class distribution before SMOTE:")
print(f"       Placed (1)     : {counts.get(1, 0)}")
print(f"       Not Placed (0) : {counts.get(0, 0)}")

# ─────────────────────────────────────────────
# 2. FEATURE / TARGET SPLIT
# ─────────────────────────────────────────────
X = df.drop(columns=["status"])
y = df["status"]

numerical_cols   = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

print(f"\n[INFO] Numerical features  ({len(numerical_cols)}): {numerical_cols}")
print(f"[INFO] Categorical features({len(categorical_cols)}): {categorical_cols}")

# Train / Test split — stratified to preserve class ratio in test set
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)
print(f"\n[INFO] Train size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 3. PREPROCESSING PIPELINE
# ─────────────────────────────────────────────
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(),                          numerical_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore",
                          sparse_output=False),        categorical_cols),
], remainder="drop")

# ─────────────────────────────────────────────
# 4. IMBLEARN PIPELINE  →  SMOTE → SVM
#    SMOTE is fitted only on training data because
#    it is inside the Pipeline (sklearn convention).
# ─────────────────────────────────────────────
pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("smote",        SMOTE(random_state=RANDOM_SEED)),   # ← balances TRAINING set only
    ("classifier",   SVC(kernel="rbf",
                         C=1.0,
                         gamma="scale",
                         random_state=RANDOM_SEED,
                         probability=False)),
])

print("\n[INFO] Pipeline architecture:")
for name, step in pipeline.steps:
    print(f"       ├─ {name:<15} → {step.__class__.__name__}")

# ─────────────────────────────────────────────
# 5. TRAIN
# ─────────────────────────────────────────────
print("\n[INFO] Fitting pipeline on training data …")
pipeline.fit(X_train, y_train)
print("[INFO] Training complete ✓")

# ─────────────────────────────────────────────
# 6. EVALUATE
# ─────────────────────────────────────────────
y_pred = pipeline.predict(X_test)

accuracy  = accuracy_score (y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score   (y_test, y_pred, zero_division=0)
f1        = f1_score       (y_test, y_pred, zero_division=0)

print("\n" + "─" * 40)
print("  MODEL EVALUATION ON TEST SET")
print("─" * 40)
print(f"  Accuracy  : {accuracy :.4f}  ({accuracy *100:.2f}%)")
print(f"  Precision : {precision:.4f}")
print(f"  Recall    : {recall   :.4f}")
print(f"  F1-Score  : {f1       :.4f}")
print("─" * 40)

# ─────────────────────────────────────────────
# 7. EDA VISUALISATIONS  (saved as PNG)
# ─────────────────────────────────────────────
# Use the original df (status back to readable labels for plots)
df_plot = df.copy()
df_plot["status_label"] = df_plot["status"].map({1: "Placed", 0: "Not Placed"})

# Colour palette
PALETTE = {"Placed": "#2ECC71", "Not Placed": "#E74C3C"}

def plot_placement_rate(df_src: pd.DataFrame,
                        group_col: str,
                        title: str,
                        filename: str) -> None:
    """Compute placement rate per category and save a bar chart."""

    # Placement rate = Placed / Total per group
    summary = (
        df_src.groupby([group_col, "status_label"])
              .size()
              .reset_index(name="count")
    )
    totals = summary.groupby(group_col)["count"].transform("sum")
    summary["rate"] = summary["count"] / totals * 100

    placed = summary[summary["status_label"] == "Placed"].copy()
    placed = placed.sort_values("rate", ascending=False)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)

    # LEFT: stacked count bar chart
    pivot_counts = (
        df_src.groupby([group_col, "status_label"])
              .size()
              .unstack(fill_value=0)
    )
    # Ensure both columns exist
    for col in ["Placed", "Not Placed"]:
        if col not in pivot_counts.columns:
            pivot_counts[col] = 0

    pivot_counts[["Not Placed", "Placed"]].plot(
        kind="bar",
        stacked=True,
        ax=axes[0],
        color=["#E74C3C", "#2ECC71"],
        edgecolor="white",
        width=0.55,
    )
    axes[0].set_title("Headcount by Status", fontsize=12)
    axes[0].set_xlabel(group_col.replace("_", " ").title(), fontsize=11)
    axes[0].set_ylabel("Number of Students", fontsize=11)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(title="Status", fontsize=9)
    axes[0].yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # RIGHT: placement-rate bar chart
    bars = axes[1].bar(
        placed[group_col],
        placed["rate"],
        color=[PALETTE["Placed"]] * len(placed),
        edgecolor="white",
        width=0.45,
    )
    for bar, val in zip(bars, placed["rate"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.0,
            f"{val:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold"
        )
    axes[1].set_title("Placement Rate (%)", fontsize=12)
    axes[1].set_xlabel(group_col.replace("_", " ").title(), fontsize=11)
    axes[1].set_ylabel("Placement Rate (%)", fontsize=11)
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].set_ylim(0, 110)
    axes[1].axhline(y=df_src["status"].mean() * 100, color="grey",
                    linestyle="--", linewidth=1.2, label="Overall avg")
    axes[1].legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved EDA chart → {filename}")


# Chart 1 — Gender
plot_placement_rate(df_plot, "gender",
                    "Placement Rate by Gender",
                    "eda_placement_by_gender.png")

# Chart 2 — Work Experience
plot_placement_rate(df_plot, "workex",
                    "Placement Rate by Work Experience",
                    "eda_placement_by_workex.png")

# Chart 3 — Specialisation
plot_placement_rate(df_plot, "specialisation",
                    "Placement Rate by MBA Specialisation",
                    "eda_placement_by_specialisation.png")

print("\n[DONE] All tasks complete.")
print("       EDA charts : eda_placement_by_gender.png")
print("                    eda_placement_by_workex.png")
print("                    eda_placement_by_specialisation.png")
