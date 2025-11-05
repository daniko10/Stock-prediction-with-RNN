import xml.etree.ElementTree as ET
from sklearn.preprocessing import StandardScaler
import joblib
import os
import csv
import pandas as pd
import numpy as np
from typing import List, Tuple
from read_files import read_stock_data, read_rate, read_cpi, read_exchange

def parse_interest_rates(xml_path, out_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    with open(out_path, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=';')
        writer.writerow(["Date", "Ref"])

        for section in root.findall("pozycje"):
            date = section.attrib.get("obowiazuje_od")

            for pos in section.findall("pozycja"):
                if pos.attrib.get("id") == "ref":
                    rate = pos.attrib.get("oprocentowanie").replace(",", ".")
                    writer.writerow([date, rate])
                    break


parse_interest_rates("data/stopy_procentowe_archiwum.xml", "data/stopy_ref.csv")

def resample(df, cols):
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df = df.set_index('Date', drop=True)

    idx = pd.bdate_range(df.index.min(), df.index.max())
    df = df.reindex(idx)

    df.index.name = 'Date'

    for col in cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    return df.reset_index()

def make_sequences(X, y, window_size, days_to_predict):
    Xs, ys = [], []
    for i in range(len(X) - window_size - days_to_predict + 1):
        Xs.append(X[i:(i + window_size)])
        ys.append(y[(i + window_size):(i + window_size + days_to_predict)])
    return np.array(Xs), np.array(ys)

def build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv, is_testing=False):
    local = read_stock_data(local_csv, is_testing)

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
    cpi = cpi.groupby('Date').nth(2).reset_index()
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

    return local

def build_all_sequences(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv, outdir, window, days_to_predict):
    all_X_seq = []
    all_y_seq = []

    os.makedirs(outdir, exist_ok=True)

    market_name = os.path.splitext(os.path.basename(local_csv))[0]
    print(f"Extracting: {market_name}")

    features = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv)
    features = features.iloc[1:-1].reset_index(drop=True)

    X = features.drop(columns=['Date'])
    y = features['Close'].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).flatten()

    joblib.dump(scaler_X, os.path.join(outdir, f"scaler_X_{market_name}.pkl"))
    joblib.dump(scaler_y, os.path.join(outdir, f"scaler_y_{market_name}.pkl"))

    return make_sequences(X_scaled, y_scaled, window, days_to_predict)