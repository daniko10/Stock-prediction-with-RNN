import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from read_files import read_stock_data

LICZBA_DNI_ROKU_BIZNESOWEGO = 252
LICZBA_DNI_DO_PRZODU = 10

if __name__ == "__main__":

    train_data = read_stock_data("data/wig20_d.csv")
    test_data = read_stock_data("data/wig20_d.csv", is_testing=True)

    print("Dlugosc danych treningowych:", len(train_data))
    print("Dlugosc danych testowych:", len(test_data))

    stock_prices_train = train_data["Close"]
    log_returns = np.log(stock_prices_train / stock_prices_train.shift(1)).dropna()
    stock_prices = test_data["Close"][len(train_data)-1:]

    mu = log_returns.mean() * LICZBA_DNI_ROKU_BIZNESOWEGO # oczekiwana roczna stopa zwrotu
    sigma = log_returns.std() * np.sqrt(LICZBA_DNI_ROKU_BIZNESOWEGO) # roczna zmienność
    T = LICZBA_DNI_DO_PRZODU / LICZBA_DNI_ROKU_BIZNESOWEGO # chce symulować na 10 dni do przodu

    print(f"Oczekiwana roczna stopa zwrotu (mu): {mu:.4f}")
    print(f"Roczna zmienność (sigma): {sigma:.4f}")

    n_time_intervals = LICZBA_DNI_DO_PRZODU # liczba kroków czasowych (dni)
    n_simulations = 100 # liczba symulacji

    dt = T / n_time_intervals # długość kroku czasowego

    available_n_days = len(stock_prices) - LICZBA_DNI_DO_PRZODU
    for i in range(available_n_days):
        S0 = stock_prices.iloc[i]  # ostatnia znana cena akcji
        S_fwd = np.zeros((n_time_intervals + 1, n_simulations))
        S_fwd[0] = S0

        for t in range(1, n_time_intervals + 1):
            Z = np.random.standard_normal(n_simulations)
            S_fwd[t] = S_fwd[t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

    plt.figure(figsize=(10,5))
    plt.plot(S_fwd[:,0:n_simulations])
    plt.title(f"Symulacje cen WIG20 – prognoza na {LICZBA_DNI_DO_PRZODU} dni\nOczekiwana roczna stopa zwrotu: {mu:.2%}, Roczna zmienność: {sigma:.2%}")
    plt.xlabel("Dzień symulacji")
    plt.ylabel("Cena indeksu")
    plt.grid()
    plt.savefig("GBM_simulations.png")