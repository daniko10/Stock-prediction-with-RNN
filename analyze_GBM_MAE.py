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
    n_simulations = 10

    dt = T / n_time_intervals

    ae_per_day = np.zeros((days_to_predict,))
    S_fwd = np.zeros((n_time_intervals + 1, n_simulations))

    available_n_days = len(stock_prices) - days_to_predict
    for i in range(available_n_days):
        S_fwd[0] = stock_prices.iloc[i]

        for t in range(1, n_time_intervals + 1):
            Z = np.random.standard_normal(n_simulations)
            S_fwd[t] = S_fwd[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        for d in range(days_to_predict): # Licze MAE tylko dla pierwszej symulacji
            ae_per_day[d] += np.abs(S_fwd[d + 1, 0] - stock_prices.iloc[i + d + 1])
        if i == 0: # Wyrysowanie wszystkikch dla pierwszej iteracji - 10 pierwszych dni styczniowych
            true_future = stock_prices.iloc[0 : days_to_predict+1].reset_index(drop=True)
            plt.figure(figsize=(12, 6))
            plt.plot(range(n_time_intervals), S_fwd[1:], lw=1)
            plt.plot(range(n_time_intervals), true_future[1:], linestyle="--", color="tab:orange",
                    linewidth=2, label="Prawdziwa przyszłość")
            plt.title(f"GBM - symulacja 10 możliwości - {market_name}")
            plt.xlabel("Dni")
            plt.ylabel("Cena zamknięcia")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{outdir}/GBM_simulations_all_{market_name}.png", dpi=150)
            plt.show()
            plt.clf()

            # Rysowanie jednej ścieżki wraz z prawdziwą przyszłością
            plt.figure(figsize=(12, 6))
            plt.plot(range(n_time_intervals), true_future[1:], linestyle="--", color="tab:orange",
                    linewidth=2, label="Prawdziwa przyszłość")
            plt.plot(range(n_time_intervals), S_fwd[1:, 0], color="tab:green",
                    linewidth=2, label="Predykcja (1. ścieżka)")
            plt.title(f"GBM - Predykcja jednej ścieżki razem z prawdziwą - {market_name}")
            plt.xlabel("Dni")
            plt.ylabel("Cena zamknięcia")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"{outdir}/GBM_forecast_{market_name}.png", dpi=150)
            plt.show()


    avg_mae_per_day = ae_per_day / available_n_days
    plt.figure(figsize=(10,5))
    plt.plot(range(1, days_to_predict + 1), avg_mae_per_day, marker='o')
    plt.title(f"GBM - MAE osobne dla danego dnia predykcji (1–{days_to_predict} dni) - {market_name}")
    plt.xlabel("Dzień predykcji")
    plt.ylabel("Średnie MAE")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{outdir}/MAE_{market_name}.png")