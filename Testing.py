import os
from sklearn.preprocessing import StandardScaler
from helper import build_features
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import math
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error
from helper import make_sequences
from helper import make_sequences

if __name__ == '__main__':
    local_csv = "data/wig30_d.csv"
    spx_csv   = "data/spx_d.csv"
    fx_csv    = "data/usdpln_d.csv"
    cpi_csv   = "data/miesieczne_wskazniki_cen_towarow_i_uslug_konsumpcyjnych_od_1982roku.csv"
    rate_csv  = "data/stopy_ref.csv"
    outdir = "runs/multi_run"
    
    mae_best = math.inf
    window_size_best = 0
    dropout_best = 0
    batch_size_best = 0
    
    market_name = os.path.splitext(os.path.basename(local_csv))[0]
    
    features = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv, False)
    features = features.iloc[1:-1].reset_index(drop=True)
    days_to_predict = 10

    X = features.drop(columns=['Date'])
    y_true = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv, is_testing=True)['Close'][len(features):(len(features)+days_to_predict)].values
    y_t = features['Close']
    dates = features['Date']
    
    scaler_X_path = os.path.join(outdir, f"scaler_X_{market_name}.pkl")
    scaler_y_path = os.path.join(outdir, f"scaler_y_{market_name}.pkl")
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    X_scaled = scaler_X.transform(X)
    
    results = []
    best_mae = math.inf
    
    for window_size in [30, 60, 90, 120]:
      for dropout in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for batch_size in [32, 64, 128, 256, 512, 1028]:
    
            model_path = os.path.join(outdir, f"model_BS_{batch_size}_WS_{window_size}_D_{dropout}.h5")

            try:
                model = load_model(model_path, compile=False)
                print(f"Model found: {model_path}\n")
            except Exception as e:
                print(f"There was a problem with loading {model_path}\n{e}")
                continue

            last_window = X_scaled[-window_size:]
            last_window = np.expand_dims(last_window, axis=0)
            
            Y_pred_scaled = model.predict(last_window)
            
            print("Y_pred_scaled shape:", Y_pred_scaled.shape)
            
            Y_pred = scaler_y.inverse_transform(Y_pred_scaled)[0]

            mae = mean_absolute_error(y_true, Y_pred)
            mse = mean_squared_error(y_true, Y_pred)
            rmse = math.sqrt(mse)

            results.append({
                "window_size": window_size,
                "dropout": dropout,
                "batch_size": batch_size,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse
            })

            print(f"MAE={mae:.6f}, MSE={mse:.6f}, RMSE={rmse:.6f}")

            if mae < best_mae:
                best_mae = mae
                best_model = (window_size, dropout, batch_size, model_path, mae, mse, rmse)

            last_date = pd.to_datetime(dates.iloc[-1])
            future_dates = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=days_to_predict)

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

    print("\nBest model:")
    print(f"Window Size: {best_model[0]}, Dropout: {best_model[1]}, Batch Size: {best_model[2]}")
    print(f"MAE: {best_model[4]:.6f}, MSE: {best_model[5]:.6f}, RMSE: {best_model[6]:.6f}")