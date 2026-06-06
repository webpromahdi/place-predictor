import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

import joblib
import os

print("All imports successful.")

#load data---->

DATA_PATH = "../data/raw/Placement_Data_Full_Class.csv"

df = pd.read_csv(DATA_PATH)

print(f"\n Dataset shape: {df.shape}")
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())

#data cleaning and leakage prevention

df = df.drop(columns=["sl_no", "salary"])

print("\n Dropped 'sl_no' and 'salary' (leakage prevention).")
print(f"Dataset shape after dropping: {df.shape}")

