# -----------------------------
# ANDRO BIERT (MOHAMED OUADDANE BEST REGARDS)
# -----------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller, plot_acf, plot_pacf
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


# -----------------------------
# 1. Feature Engineering
# -----------------------------
def compute_power_factor(df):
    P = df["global_active_power"]
    Q = df["global_reactive_power"]
    S = np.sqrt(P**2 + Q**2)
    df["power_factor"] = round(P/S, 5)
    return df


def compute_peak_offpeak_ratio(df):
    daily = df["global_active_power"].resample("D").agg(["max", "min"])
    daily["Peak_OffPeak_Ratio"] = daily["max"] / daily["min"].replace(0, np.nan)
    return daily["Peak_OffPeak_Ratio"]


def create_lag_features(data, n_lags=7):
    df_lags = data.copy()
    for lag in range(1, n_lags+1):
        df_lags[f"lag_{lag}"] = df_lags['global_active_power'].shift(lag)
    return df_lags.dropna()


# -----------------------------
# 2. Stationarity Check
# -----------------------------
def check_stationarity(series):
    result = adfuller(series.dropna())
    print("ADF Statistic:", result[0])
    print("p-value:", result[1])
    if result[1] <= 0.05:
        print("The series is stationary")
    else:
        print("The series is NOT stationary")


# -----------------------------
# 3. SARIMA Modeling
# -----------------------------
def sarima_forecast(series, order, seasonal_order, params, last_obs=30):
    model = SARIMAX(series, order=order, seasonal_order=seasonal_order,
                    enforce_stationarity=False, enforce_invertibility=False)
    res = model.filter([params.get(k, 0) for k in model.param_names])
    forecast = res.get_forecast(steps=last_obs)
    forecast_mean = forecast.predicted_mean
    forecast_ci = forecast.conf_int()
    return forecast_mean, forecast_ci


# -----------------------------
# 4. XGBoost Modeling
# -----------------------------
def xgb_train_predict(X_train, y_train, X_test, y_test, best_params=None):
    if best_params is None:
        best_params = {
            'colsample_bytree': 0.8,
            'learning_rate': 0.01,
            'max_depth': 3,
            'n_estimators': 500,
            'subsample': 0.8
        }

    model = XGBRegressor(**best_params, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}")

    return y_pred, model


# -----------------------------
# 5. LSTM Modeling
# -----------------------------
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


def lstm_train_predict(X_tensor, y_tensor, last_n=30, epochs=50, batch_size=32, lr=0.001):
    dataset = TensorDataset(X_tensor, y_tensor)
    train_size = int(0.9 * len(dataset))
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset)-train_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_tensor.shape[1]
    model = LSTMModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}")

    # Predict last_n
    model.eval()
    X_test = X_tensor[-last_n:]
    y_test = y_tensor[-last_n:].numpy()
    with torch.no_grad():
        y_pred = model(X_test).numpy()

    return y_pred, y_test, model


# -----------------------------
# 6. Plotting
# -----------------------------
def plot_forecast(train, test, forecast_mean, forecast_ci=None, title="Forecast"):
    plt.figure(figsize=(12,6))
    plt.plot(train.index[-30:], train['global_active_power'].iloc[-30:], label="Train (Derniers points)", color="blue", alpha=0.6)
    plt.plot(test.index, test['global_active_power'], label="Test (Réel)", color="black")
    plt.plot(forecast_mean.index, forecast_mean, label="Forecast", color="red")
    if forecast_ci is not None:
        plt.fill_between(forecast_ci.index, forecast_ci.iloc[:,0], forecast_ci.iloc[:,1], color="pink", alpha=0.3)
    plt.legend()
    plt.title(title)
    plt.show()


def plot_lstm_forecast(index, y_true, y_pred, title="LSTM Forecast"):
    plt.figure(figsize=(12,6))
    plt.plot(index, y_true, label="Réel", color="black")
    plt.plot(index, y_pred, label="Prédiction", color="red")
    plt.title(title)
    plt.legend()
    plt.show()
