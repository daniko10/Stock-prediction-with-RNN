import os
from sklearn.preprocessing import StandardScaler
from helper import build_features
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
    
    features = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv)
    features = features.iloc[1:-1].reset_index(drop=True)

    X = features.drop(columns=['Date','y_t'])
    y_t = features['y_t'].values
    date = features['Date'].values
    
    scaler_X_path = os.path.join(outdir, f"scaler_X_{market_name}.pkl")
    scaler_y_path = os.path.join(outdir, f"scaler_y_{market_name}.pkl")
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)
    X_scaled = scaler_X.transform(X)
    
    for window_size in [120, 90, 60, 30]:
      for dropout in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]:
        for batch_size in [1028, 512, 256, 128, 64]:
    
            model_path = os.path.join(outdir, f"model_BS_{batch_size}_WS_{window_size}_D_{dropout}.h5")

            try:
                model = load_model(model_path, compile=False)
                print(f"Model found: {model_path}\n")
            except Exception as e:
                print(f"There was a problem with loading {model_path}\n{e}")
                continue

            X_windows = []
            for i in range(window_size, len(X_scaled)):
                X_windows.append(X_scaled[i-window_size:i])

            X_windows = np.array(X_windows)

            preds_scaled = model.predict(X_windows, verbose=1) 
            preds_scaled = np.array(preds_scaled)
            preds = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).flatten()

            y_t_window = y_t[window_size:]

            dates = date[window_size:]
            plt.figure(figsize=(12,6))
            plt.plot(dates, y_t_window, label='True Close')
            plt.plot(dates, preds, label='Predicted Close')
            plt.title(f'{market_name}')
            plt.xlabel('Day')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig(f"wykres_BS_{batch_size}_WS_{window_size}_D_{dropout}.png", dpi=300)

            mae_local = mean_absolute_error(y_t_window, preds)
            if (mae_best > mae_local):
                mae_best = mae_local
                window_size_best = window_size
                dropout_best = dropout
                batch_size_best = batch_size
            
            print("MAE:", mae_local)
            print("RMSE:", math.sqrt(mean_squared_error(y_t_window, preds)))
    
    print(f"Best MAE: {mae_best} with Window Size: {window_size_best}, Dropout: {dropout_best}, Batch Size: {batch_size_best}")