# Electricity Consumption Forecasting

## 📄 Project Overview
This project focuses on forecasting electricity consumption using **time series analysis** and **machine learning**.  
The main goals are:

1. Preprocess and clean the dataset (missing values, outliers, feature engineering).  
2. Explore and visualize the data.  
3. Forecast electricity consumption using:
   - SARIMA (seasonal ARIMA)
   - XGBoost (with lag features)
   - LSTM (Deep Learning with PyTorch)  
4. Compare models using metrics like MSE, RMSE, MAE.  


---

## 🗂 Project Structure


### ✅ Explanation

- **data/raw/** → Original datasets from source.  
- **data/processed/** → Cleaned and preprocessed datasets ready for modeling.  
- **src/data_loader.py** → Load data, inspect, handle missing values, outliers, feature engineering.  
- **src/modeling.py** → All forecasting models (SARIMA, XGBoost, LSTM).  
- **src/evaluator.py** → Functions to calculate MSE, RMSE, MAE, and plot predictions.  
- **outputs/** → Folder to save trained models.  
- **notebooks/** → Jupyter notebooks for experimentation and visualization.  
- **requirements.txt** → Python dependencies for reproducibility.  
- **config.py** → Store paths like `PROCESSED_DATA_PATH` and `PLOTS_DIR`.  

---


Do you want me to do that?



---

## 📦 Requirements

Install the packages from `requirements.txt`:

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt

