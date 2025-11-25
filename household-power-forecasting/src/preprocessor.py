# -----------------------------
# ANDRO BIERT (MOHAMED OUADDANE BEST REGARDS)
# -----------------------------

import pandas as pd
import numpy as np
from scipy.stats import zscore
from src.data_loader import load_data, inspect_data, explain, describe_data, save_processed_dataset
from summarytools import dfSummary, display

# -----------------------------
# 1. Loading & Inspecting Data
# -----------------------------
def load_and_inspect():
    df = load_data()
    print("=== Inspect Data ===")
    inspect_data(df)
    print("=== Explain Data ===")
    explain(df)
    print("=== Describe Data ===")
    describe_data(df)
    print("=== Summary Data ===")
    display(dfSummary(df))
    return df


# -----------------------------
# 2. Preprocessing
# -----------------------------
def lowercase_columns(df):
    df.columns = df.columns.str.lower()
    return df


def handle_missing_values(df):
    """
    Fill missing numeric values with column mean.
    """
    df = df.fillna(df.mean(numeric_only=True))
    return df


def missing_data_report(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = round(df.isnull().sum()/df.shape[0]*100, 2).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', '%'])
    return missing_data


# -----------------------------
# 3. Outliers Detection
# -----------------------------
def detect_outliers_zscore(df, cols=None, threshold=5):
    if cols is None:
        cols = df.columns
    outliers = {}
    for col in cols:
        z_scores = zscore(df[col].dropna())
        outliers[col] = df[col][abs(z_scores) > threshold]
    return outliers


def detect_outliers_iqr(df, cols=None):
    if cols is None:
        cols = df.columns
    outliers = {}
    for col in cols:
        Q1 = df[col].quantile(0.01)
        Q3 = df[col].quantile(0.99)
        IQR = Q3 - Q1
        lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        outliers[col] = df[(df[col] < lower) | (df[col] > upper)][col]
    return outliers


# -----------------------------
# 4. Save Processed Data
# -----------------------------
def save_preprocessed(df, filepath='../data/processed/preprocessed_data.csv'):
    df.to_csv(filepath, index=True)
    print(f"Data saved to {filepath}")


# -----------------------------
# 5. Example Workflow
# -----------------------------
if __name__ == "__main__":
    df = load_and_inspect()
    df = lowercase_columns(df)
    df = handle_missing_values(df)
    print("=== Missing Data After Fill ===")
    print(missing_data_report(df))
    print("=== Detect Outliers (Z-score) ===")
    z_outliers = detect_outliers_zscore(df)
    print(z_outliers)
    print("=== Detect Outliers (IQR) ===")
    iqr_outliers = detect_outliers_iqr(df)
    print(iqr_outliers)
    save_preprocessed(df)
