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
    local_csv = "data/swig80_d.csv"
    spx_csv   = "data/spx_d.csv"
    fx_csv    = "data/usdpln_d.csv"
    cpi_csv   = "data/miesieczne_wskazniki_cen_towarow_i_uslug_konsumpcyjnych_od_1982roku.csv"
    rate_csv  = "data/stopy_ref.csv"

    outdir = "runs/multi_run"
    market_name = os.path.splitext(os.path.basename(local_csv))[0]

    mae_best = math.inf
    window_size_best = 0
    dropout_best = 0
    batch_size_best = 0
    
    features = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv)
    features = features.iloc[1:-1].reset_index(drop=True)
    # features = features[3000:]

    y_true = features['y_t'].values

    date = features['Date'].values
    X = features.drop(columns=['Date','y_t'])
    
    for window_size in [30, 60, 90]:
      for dropout in [0.3, 0.4, 0.5]:
        for batch_size in [64, 32, 16, 8]:
    
            model_path = os.path.join(outdir, f"model_BS_{batch_size}_WS_{window_size}_D_{dropout}.h5")

            try:
                model = load_model(model_path, compile=False)
                print(f"Model found: {model_path}\n")
            except Exception as e:
                print(f"There is not found: {model_path}\n{e}")
                continue
            
            scaler_X_path = os.path.join(outdir, f"scaler_X_{market_name}.pkl")
            scaler_y_path = os.path.join(outdir, f"scaler_y_{market_name}.pkl")

            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            scaler_X.fit(X)
            scaler_y.fit(y_true.reshape(-1,1))

            if os.path.exists(scaler_X_path) and os.path.exists(scaler_y_path):
                scaler_X = joblib.load(scaler_X_path)
                scaler_y = joblib.load(scaler_y_path)
            else:
                os.makedirs(outdir, exist_ok=True)
                joblib.dump(scaler_X, scaler_X_path)
                joblib.dump(scaler_y, scaler_y_path)

            X_scaled = scaler_X.transform(X)

            X_windows = []
            for i in range(window_size, len(X_scaled)):
                X_windows.append(X_scaled[i-window_size:i])

            X_windows = np.array(X_windows)

            preds_scaled = model.predict(X_windows, verbose=1) 

            preds_scaled = np.array(preds_scaled)
            preds = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).flatten()

            y_true_window = y_true[window_size:]

            dates = date[window_size:]
            plt.figure(figsize=(12,6))
            plt.plot(dates, y_true_window, label='True Close')
            plt.plot(dates, preds, label='Predicted Close')
            plt.title(f'{market_name}')
            plt.xlabel('Day')
            plt.ylabel('Price')
            plt.legend()

            plt.savefig(f"wykres_BS_{batch_size}_WS_{window_size}_D_{dropout}.png", dpi=300)

            # plt.show()
            
            if (mae_best > mean_absolute_error(y_true_window, preds)):
                mae_best = mean_absolute_error(y_true_window, preds)
                window_size_best = window_size
                dropout_best = dropout
                batch_size_best = batch_size
            
            print("MAE:", mean_absolute_error(y_true_window, preds))
            print("RMSE:", math.sqrt(mean_squared_error(y_true_window, preds)))
    
    print(f"Best MAE: {mae_best} with Window Size: {window_size_best}, Dropout: {dropout_best}, Batch Size: {batch_size_best}")