import os
import argparse
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

from read_files import read_stock_data, read_rate, read_cpi, read_exchange
from helper import resample, make_sequences
from model import build_lstm

from tensorflow.keras.callbacks import EarlyStopping

def build_features(local_csv: str,
                   spx_csv: str,
                   fx_csv: str,
                   cpi_csv: str,
                   rate_csv: str) -> pd.DataFrame:

    local = read_stock_data(local_csv)

    month = local['Date'].dt.month
    local['month_sin'] = np.sin(2*np.pi*month/12.0)
    local['month_cos'] = np.cos(2*np.pi*month/12.0)

    # S&P500
    spx = read_stock_data(spx_csv)
    spx['spx_change'] = spx['Close'].pct_change()
    spx = resample(spx, ['spx_change'])
    local = local.merge(spx[['Date','spx_change']], on='Date', how='left')
    local['spx_change'] = local['spx_change'].ffill()

    # USD
    fx = read_exchange(fx_csv)
    fx = resample(fx, ['usd_change'])
    local = local.merge(fx[['Date','usd_change']], on='Date', how='left')
    local['usd_change'] = local['usd_change'].ffill()
    
    # CPI
    cpi = read_cpi(cpi_csv)
    cpi = cpi.groupby('Date').nth(2).reset_index() # to jest miesiac z poprzedniego roku
    cpi = resample(cpi, ['Value'])
    cpi.rename(columns={'Value':'CPI'}, inplace=True)
    local = local.merge(cpi, on='Date', how='left')
    local['CPI'] = local['CPI'].ffill()

    # Rate
    rate = read_rate(rate_csv)
    rate.rename(columns={'Ref':'Rate'}, inplace=True)
    rate = resample(rate, ['Rate'])
    local = local.merge(rate, on='Date', how='left')
    local['Rate'] = local['Rate'].ffill()
    
    local = local.drop(columns=['index'])
    
    local['y_t'] = local['Close'].shift(-1)
    
    print(local)

    return local
    
def main():
    local_csv = "data/wig20_d.csv"
    spx_csv   = "data/spx_d.csv"
    fx_csv    = "data/usdpln_d.csv"
    cpi_csv   = "data/miesieczne_wskazniki_cen_towarow_i_uslug_konsumpcyjnych_od_1982roku.csv"
    rate_csv  = "data/stopy_ref.csv"
    
    market_name = os.path.splitext(os.path.basename(local_csv))[0]
    
    features = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv)
    features = features.iloc[1:-1].reset_index(drop=True)  # usuwam pierwszy bo ma NaN po pct_change i ostatni bo nie mam y_t
    
    print(features.isna().sum())
    
    outdir = f"runs/{market_name}_run"
    os.makedirs(outdir, exist_ok=True)
    
    X = features.drop(columns=['Date','y_t'])
    y = features['y_t']
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X = scaler_X.fit_transform(X)
    y = scaler_y.fit_transform(y.values.reshape(-1,1)).flatten()
    
    X_seq, y_seq = make_sequences(X, y, window=30)
    
    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]
    
    model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=50,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1, callbacks=[early_stop])

    model.save(os.path.join(outdir, "model.h5"))

    preds = model.predict(X_test).flatten()
    preds = scaler_y.inverse_transform(preds.reshape(-1,1)).flatten()
    y_test = scaler_y.inverse_transform(y_test.reshape(-1,1)).flatten()
    
    preds = np.clip(preds, np.percentile(preds, 1), np.percentile(preds, 99))
    
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", math.sqrt(mean_squared_error(y_test, preds)))

    plt.figure(figsize=(10,5))
    plt.plot(y_test, label='True')
    plt.plot(preds, label='Predicted')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()