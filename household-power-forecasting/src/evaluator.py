# -----------------------------
# ANDRO BIERT (MOHAMED OUADDANE BEST REGARDS)
# -----------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# -----------------------------
# 1. Error Metrics
# -----------------------------
def calculate_metrics(y_true, y_pred):
    """
    Calculate common regression metrics.
    Returns a dictionary of metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100  # avoid div by 0
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "R2": r2
    }
    return metrics

# -----------------------------
# 2. Print Metrics
# -----------------------------
def print_metrics(metrics_dict):
    for k, v in metrics_dict.items():
        if k in ["MAPE", "R2"]:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v:.6f}")

# -----------------------------
# 3. Plot Predictions
# -----------------------------
def plot_predictions(y_true, y_pred, title="Predictions vs True Values", figsize=(12,6), save_path=None):
    plt.figure(figsize=figsize)
    plt.plot(y_true.index, y_true, label="Réel", color="black", alpha=0.7)
    plt.plot(y_true.index, y_pred, label="Prédiction", color="red", alpha=0.6)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Values")
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path)
    plt.show()

# -----------------------------
# 4. Compare Multiple Models
# -----------------------------
def compare_models(y_true, predictions_dict):
    """
    predictions_dict = {
        "SARIMA": y_pred_sarima,
        "XGBoost": y_pred_xgb,
        "LSTM": y_pred_lstm
    }
    """
    plt.figure(figsize=(12,6))
    plt.plot(y_true.index, y_true, label="Réel", color="black", alpha=0.7)
    for model_name, y_pred in predictions_dict.items():
        plt.plot(y_true.index, y_pred, label=model_name, alpha=0.6)
    plt.title("Comparaison des modèles")
    plt.xlabel("Time")
    plt.ylabel("Valeurs")
    plt.legend()
    plt.grid(True)
    plt.show()
