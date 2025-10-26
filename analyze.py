import os
import math
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from helper import build_features

if __name__ == '__main__':
    local_csv = "data/wig30_d.csv"
    spx_csv   = "data/spx_d.csv"
    fx_csv    = "data/usdpln_d.csv"
    cpi_csv   = "data/miesieczne_wskazniki_cen_towarow_i_uslug_konsumpcyjnych_od_1982roku.csv"
    rate_csv  = "data/stopy_ref.csv"
    outdir    = "runs/multi_run"

    market_name = os.path.splitext(os.path.basename(local_csv))[0]
    days_to_predict = 10
    window_size = 120
    dropout = 0.2
    batch_size = 64

    features = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv, False)
    features = features.iloc[1:-1].reset_index(drop=True)
    dates = features['Date']
    y_t = features['Close']

    y_true_future = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv, is_testing=True)['Close']
    y_true = y_true_future[len(features):(len(features) + days_to_predict)].values

    scaler_X_path = os.path.join(outdir, f"scaler_X_{market_name}.pkl")
    scaler_y_path = os.path.join(outdir, f"scaler_y_{market_name}.pkl")
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    X = features.drop(columns=['Date'])
    X_scaled = scaler_X.transform(X)

    model_path = os.path.join(outdir, f"model_BS_{batch_size}_WS_{window_size}_D_{dropout}.h5")

    try:
        model = load_model(model_path, compile=False)
        print(f"\nModel załadowany: {model_path}")
    except Exception as e:
        print(f"\nProblem z wczytaniem modelu: {model_path}\n{e}")
        exit(1)

    last_window = X_scaled[-window_size:]
    last_window = np.expand_dims(last_window, axis=0)

    Y_pred_scaled = model.predict(last_window)
    Y_pred = scaler_y.inverse_transform(Y_pred_scaled)[0]

    mae = mean_absolute_error(y_true, Y_pred)
    mse = mean_squared_error(y_true, Y_pred)
    rmse = math.sqrt(mse)

    print(f"\nWyniki predykcji ({days_to_predict} dni):")
    print(f"MAE  = {mae:.4f}")
    print(f"MSE  = {mse:.4f}")
    print(f"RMSE = {rmse:.4f}")

    mae_per_day = np.abs(y_true - Y_pred)

    cumulative_mae = []
    for i in range(len(mae_per_day)):
        cumulative_mae.append(np.mean(mae_per_day[:i+1]))

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, days_to_predict + 1), mae_per_day, marker='o', label='MAE daily')
    plt.plot(range(1, days_to_predict + 1), cumulative_mae, marker='s', linestyle='--', label='MAE accumulated')
    plt.title(f"MAE for ({market_name})")
    plt.xlabel("Day")
    plt.ylabel("MAE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"mae_per_day_{market_name}.png")
    plt.close()

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

    print(f"\nZapisano wykresy: {plot_name} oraz mae_per_day_{market_name}.png\n")
