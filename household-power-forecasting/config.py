
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_PATH = os.path.join(DATA_DIR, 'raw', 'household_power_consumption.txt')
PROCESSED_DATA_PATH = os.path.join(DATA_DIR, 'processed', 'preprocessed_data.csv')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
FORECASTS_DIR = os.path.join(OUTPUTS_DIR, 'forecasts')
PLOTS_DIR = os.path.join("../outputs/plots")
# Data params
RESAMPLE_FREQ = 'D'  # 'D' for daily, 'W' for weekly, etc.
FILL_METHOD = 'mean'  # Forward fill for missing values
TARGET_COL = 'Global_active_power'  # Main target for forecasting

# Model params (example for LSTM)
LSTM_UNITS = 50
EPOCHS = 100
BATCH_SIZE = 32

# # Create directories if they don't exist
# os.makedirs(DATA_DIR, exist_ok=True)
# os.makedirs(RAW_DATA_PATH.replace('household_power_consumption.txt', ''), exist_ok=True)
# os.makedirs(PROCESSED_DATA_PATH.replace('daily_power.csv', ''), exist_ok=True)
# os.makedirs(OUTPUTS_DIR, exist_ok=True)
# os.makedirs(PLOTS_DIR, exist_ok=True)
# os.makedirs(FORECASTS_DIR, exist_ok=True)