import os
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import math
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from helper_functions import make_sequences, build_features

if __name__ == '__main__':
    local_csv = "data/wig30_d.csv"
    spx_csv   = "data/spx_d.csv"
    fx_csv    = "data/usdpln_d.csv"
    cpi_csv   = "data/miesieczne_wskazniki_cen_towarow_i_uslug_konsumpcyjnych_od_1982roku.csv"
    rate_csv  = "data/stopy_ref.csv"
    outdir_scalers = "scalers/"
    outdir_models = "models/"
    
    mae_best = math.inf
    window_size_best = 0
    dropout_best = 0
    batch_size_best = 0
    
    market_name = os.path.splitext(os.path.basename(local_csv))[0]
    
    features = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv, False)
    features = features.iloc[1:-1].reset_index(drop=True)
    days_to_predict = 10
    window_size = 120
    dropout = 0.2
    batch_size = 64

    X = features.drop(columns=['Date'])
    y_true = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv, is_testing=True)['Close'][len(features):(len(features)+days_to_predict)].values
    y_t = features['Close']
    dates = features['Date']
    
    scaler_X_path = os.path.join(outdir_scalers, f"scaler_X_{market_name}.pkl")
    scaler_y_path = os.path.join(outdir_scalers, f"scaler_y_{market_name}.pkl")
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    X_scaled = scaler_X.transform(X)
    
    model_path = os.path.join(outdir_models, f"model_BS_{batch_size}_WS_{window_size}_D_{dropout}.h5")

    try:
        model = load_model(model_path, compile=False)
        print(f"Model found: {model_path}\n")
    except Exception as e:
        print(f"There was a problem with loading {model_path}\n{e}")
        continue

    last_window = X_scaled[-window_size:]
    last_window = np.expand_dims(last_window, axis=0)
    
    Y_pred_scaled = model.predict(last_window)
    Y_pred = scaler_y.inverse_transform(Y_pred_scaled)[0]

    plt.figure(figsize=(12, 6))
    plt.plot(dates[-200:], y_t[-200:], label='True history', color='blue')
    plt.plot(future_dates, Y_pred, label=f'Forecast (next {days_to_predict} days)', color='red')
    plt.plot(future_dates, y_true, label='True future', color='green', linestyle='--')
    plt.axvline(last_date, color='gray', linestyle='--', alpha=0.7)
    plt.title(f"{market_name} – WS={window_size}, D={dropout}, BS={batch_size}")
    plt.xlabel("Date")
    plt.ylabel("Close price")
    plt.legend()
    plt.grid(True)
    plot_name = f"forecast_WS_{window_size}_D_{dropout}_BS_{batch_size}.png"
    plt.savefig(plot_name)
    plt.close()