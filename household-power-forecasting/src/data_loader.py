# -----------------------------
# ANDRO BIERT (MOHAMED OUADDANE BEST REGARDS)
# -----------------------------

import pandas as pd
import os
import sys, os


project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)

from config import RAW_DATA_PATH

def load_data():
    """
    Load the raw CSV file, parse datetime, and handle missing values.
    Returns: pd.DataFrame with datetime index.
    """
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"Raw data not found at {RAW_DATA_PATH}. Place household_power_consumption.txt in data/raw/")
    
    df = pd.read_csv(
        RAW_DATA_PATH,
        sep=';',
        parse_dates={'dt': ['Date', 'Time']},
        infer_datetime_format=True,
        low_memory=False,
        na_values=['nan', '?'],
        index_col='dt'
    )
    return df

def inspect_data(df):
    """
    Basic inspection: head, tail, info.
    """
    print("Dataset shape:", df.shape)
    print("\nHead:\n", df.head())
    print("\nTail:\n", df.tail())
    print("\nMissing values:\n", df.isnull().sum())
    return df

def info_data(df):
    """
    Print detailed info about the DataFrame.
    """
    print("\nData Info:")
    print(df.info(show_counts=True))
    return df

def describe_data(df):
    """
    Print statistical description of the DataFrame.
    """
    print("\nStatistical Description:\n", df.describe())
    return df

def explain(df):
    """
    Return a clear text description of the energy consumption dataset.
    """
    # Dataset characteristics
    n_rows, n_cols = df.shape
    missing_percent = df.isnull().mean().mean() * 100

    text = []
    text.append(f"The dataset contains {n_rows:,} rows and {n_cols} columns.")
    text.append(f"Missing values: approximately {missing_percent:.2f}% of the data.")

    text.append("\nMain Features:")
    text.append("- date: Date of measurement (dd/mm/yyyy).")
    text.append("- time: Time of measurement (hh:mm:ss).")
    text.append("- global_active_power (kW): Actual power consumed by appliances.")
    text.append("- global_reactive_power (kW): Reactive (non-consumed) power.")
    text.append("- voltage (V): Average voltage per minute.")
    text.append("- global_intensity (A): Current intensity per minute.")
    text.append("- sub_metering_1 (Wh): Kitchen appliances (dishwasher, oven, microwave).")
    text.append("- sub_metering_2 (Wh): Laundry appliances (washing machine, dryer, fridge, lights).")
    text.append("- sub_metering_3 (Wh): Water heater and air conditioner.")

    text.append(
        "\nNote: Unmeasured household consumption can be approximated by:\n"
        "(global_active_power*1000/60) – (sub_metering_1 + sub_metering_2 + sub_metering_3)"
    )

    return "\n".join(text)



if __name__ == "__main__":
    df = load_data()
    inspect_data(df)
    explain(df)



def save_processed_dataset(df, filename="processed_data.csv"):
    """
    Save the processed dataset into the data/processed directory.
    """
    # Build path relative to project structure
    processed_dir = [os.path.abspath(os.path.join(os.getcwd(), "..")), "data", "processed"]
    # os.makedirs(processed_dir, exist_ok=True)  
    
    # Full file path
    file_path = os.path.join(processed_dir, filename)
    
    # Save as CSV
    df.to_csv(file_path, index=True)
    print(f"✅ Processed dataset saved at: {file_path}")
    
from config import PLOTS_DIR
def save_fig(plt,file_name):

    SAVE_FIG_PATH = os.path.join(PLOTS_DIR,file_name)
    plt.savefig(SAVE_FIG_PATH)
    print(f"✅ Plot saved: {file_name}")
