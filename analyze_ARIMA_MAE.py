import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_files import read_stock_data
from pmdarima import auto_arima

days_to_predict = 10

if __name__ == "__main__":
    outdir = "ARIMA_results/"
    local_csv = "data/wig30_d.csv"
    train_data = read_stock_data(local_csv)
    test_data = read_stock_data(local_csv, is_testing=True)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    market_name = os.path.splitext(os.path.basename(local_csv))[0]

    stock_prices_future = test_data["Close"][len(train_data):]
    full_series = test_data["Close"]

    ae_per_day = np.zeros((days_to_predict,))

    available_n_days = len(stock_prices_future) - days_to_predict

    for i in range(available_n_days):
        history = full_series.iloc[:len(train_data) + i]

        model = auto_arima(
            history,
            stepwise=True,
            trace=True,
            trend="t"
        )

        forecast = model.predict(n_periods=days_to_predict)

        for d in range(days_to_predict):
            ae_per_day[d] += np.abs(forecast.iloc[d] - stock_prices_future.iloc[i + d])
        if i == 0: # rysuje dla pierwszych 10 dni styczniowych
            true_future = stock_prices_future.iloc[0 : days_to_predict].reset_index(drop=True)
            plt.figure(figsize=(12, 6))
            plt.plot(range(1, days_to_predict + 1), forecast, marker='o', label="Predykcja ARIMA")
            plt.plot(range(1, days_to_predict + 1), true_future, linestyle="--", color="tab:orange",
                     linewidth=2, label="Prawdziwa przyszłość")
            plt.title(f"ARIMA - Predykcja na 10 dni - {market_name}")
            plt.xlabel("Dni")
            plt.ylabel("Cena zamknięcia")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{outdir}/ARIMA_forecast_{market_name}.png", dpi=150)

    avg_mae_per_day = ae_per_day / available_n_days

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, days_to_predict + 1), avg_mae_per_day, marker='o')
    plt.title(f"ARIMA - MAE osobne dla danego dnia predykcji (1–{days_to_predict} dni) - {market_name}")
    plt.xlabel("Dzień predykcji")
    plt.ylabel("Średnie MAE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/MAE_{market_name}.png")
