import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_files import read_stock_data

np.random.seed(42)

business_day_in_year = 252
days_to_predict = 10

if __name__ == "__main__":
    outdir = "GBM_results/"
    local_csv = "data/wig30_d.csv"
    train_data = read_stock_data(local_csv)
    test_data = read_stock_data(local_csv, is_testing=True)

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    market_name = os.path.splitext(os.path.basename(local_csv))[0]

    stock_prices_train = train_data["Close"]
    log_returns = np.log(stock_prices_train / stock_prices_train.shift(1)).dropna()
    stock_prices = test_data["Close"][len(train_data)-1:]

    mu = log_returns.mean() * business_day_in_year
    sigma = log_returns.std() * np.sqrt(business_day_in_year)
    T = days_to_predict / business_day_in_year

    print(f"Oczekiwana roczna stopa zwrotu (mu): {mu:.4f}")
    print(f"Roczna zmienność (sigma): {sigma:.4f}")

    n_time_intervals = days_to_predict
    n_simulations = 1000

    dt = T / n_time_intervals

    average_close = stock_prices.mean()
    ae_per_day_all = np.zeros((days_to_predict,))
    S_fwd = np.zeros((n_time_intervals + 1, n_simulations))

    available_n_days = len(stock_prices) - days_to_predict
    for i in range(available_n_days):
        S_fwd[0] = stock_prices.iloc[i]

        for t in range(1, n_time_intervals + 1):
            Z = np.random.standard_normal(n_simulations)
            S_fwd[t] = S_fwd[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        for d in range(days_to_predict):
            ae_per_day_all[d] += np.abs(S_fwd[d + 1].mean() - stock_prices.iloc[i + d + 1])

    avg_mae_per_day = ae_per_day_all / available_n_days
    plt.figure(figsize=(10,5))
    plt.plot(range(1, days_to_predict + 1), avg_mae_per_day, marker='o')
    plt.title(f"Średnie MAE dla dnia predykcji (1–{days_to_predict} dni) - {market_name}\nŚrednia cena zamknięcia styczeń-grudzień 2024: {average_close:.2f}")
    plt.xlabel("Dzień prognozy")
    plt.ylabel("Średnie MAE")
    plt.savefig(f"{outdir}/avg_mae_per_day_GBM_{market_name}.png")