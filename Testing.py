import os
from sklearn.preprocessing import StandardScaler
from Project import build_features
from tensorflow.keras.models import load_model
import joblib
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

if __name__ == '__main__':
    local_csv = "data/wig20_d.csv"
    spx_csv   = "data/spx_d.csv"
    fx_csv    = "data/usdpln_d.csv"
    cpi_csv   = "data/miesieczne_wskazniki_cen_towarow_i_uslug_konsumpcyjnych_od_1982roku.csv"
    rate_csv  = "data/stopy_ref.csv"

    outdir = "runs/multi_run"
    market_name = os.path.splitext(os.path.basename(local_csv))[0]

    window_size = 30

    features = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv)
    features = features.iloc[1:-1].reset_index(drop=True)

    y_true = features['y_t'].values

    X = features.drop(columns=['Date','y_t'])

    model = load_model(os.path.join(outdir, "model.h5"), compile=False)
    scaler_X = joblib.load(os.path.join(outdir, f"scaler_X_{market_name}.pkl"))
    scaler_y = joblib.load(os.path.join(outdir, f"scaler_y_{market_name}.pkl"))

    X_scaled = scaler_X.transform(X)

    X_windows = []
    for i in range(window_size, len(X_scaled)):
        X_windows.append(X_scaled[i-window_size:i])

    X_windows = np.array(X_windows)

    preds_scaled = model.predict(X_windows, verbose=1) 

    preds_scaled = np.array(preds_scaled)
    preds = scaler_y.inverse_transform(preds_scaled.reshape(-1,1)).flatten()

    y_true_window = y_true[window_size:]

    plt.figure(figsize=(12,6))
    plt.plot(y_true_window, label='True Close')
    plt.plot(preds, label='Predicted Close')
    plt.title(f'{market_name}')
    plt.xlabel('Day')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    print("MAE:", mean_absolute_error(y_true_window, preds))
    print("RMSE:", math.sqrt(mean_squared_error(y_true_window, preds)))