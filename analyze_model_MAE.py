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
    outdir_scalers = "scalers/"
    outdir_models = "models/"

    market_name = os.path.splitext(os.path.basename(local_csv))[0]
    days_to_predict = 10
    window_size = 120
    dropout = 0.2
    batch_size = 64

    scaler_X_path = os.path.join(outdir_scalers, f"scaler_X_{market_name}.pkl")
    scaler_y_path = os.path.join(outdir_scalers, f"scaler_y_{market_name}.pkl")
    scaler_X = joblib.load(scaler_X_path)
    scaler_y = joblib.load(scaler_y_path)

    features = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv, False)
    features_future = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv, True)

    features = features.iloc[1:-1].reset_index(drop=True)
    dates = features['Date']
    y_t = features['Close']

    y_true_future = features_future['Close']
    X = features_future.drop(columns=['Date'])
    X_scaled = scaler_X.transform(X)

    model_path = os.path.join(outdir_models, f"model_BS_{batch_size}_WS_{window_size}_D_{dropout}_{market_name}.h5")

    try:
        model = load_model(model_path, compile=False)
        print(f"\nModel załadowany: {model_path}")
    except Exception as e:
        print(f"\nProblem z wczytaniem modelu: {model_path}\n{e}")
        exit(1)

    ae_per_day_all = np.zeros((days_to_predict,))
    average_close = y_true_future[len(features):].mean()

    available_n_days = len(features_future) - len(features) - days_to_predict
    for i in range(available_n_days):
        last_window = X_scaled[len(features)-window_size+i:(len(features) + i)]
        last_window = np.expand_dims(last_window, axis=0)

        Y_pred_scaled = model.predict(last_window)
        Y_pred = scaler_y.inverse_transform(Y_pred_scaled)[0]

        y_true = y_true_future[len(features)+i:(len(features) + days_to_predict + i)].values
        
        for j in range(days_to_predict):
            ae_per_day_all[j] += abs(y_true[j] - Y_pred[j])
    
    avg_mae_per_day = ae_per_day_all / available_n_days
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, days_to_predict + 1), avg_mae_per_day, marker='o')
    plt.title(f"Średnie MAE dla dnia predykcji (1–{days_to_predict} dni) - {market_name}\nŚrednia cena zamknięcia styczeń-grudzień 2024: {average_close:.2f}")
    plt.xlabel("Dzień predykcji")
    plt.ylabel("Średnie MAE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"avg_mae_per_day_{market_name}.png")
    plt.show()

    print("Średnie MAE dla każdego dnia:")
    for i, v in enumerate(avg_mae_per_day, 1):
        print(f"Dzień {i}: {v:.4f}")
