import xml.etree.ElementTree as ET
import csv
import pandas as pd
import numpy as np
from typing import List, Tuple
from read_files import read_stock_data, read_rate, read_cpi, read_exchange

def parse_interest_rates(xml_path: str, out_path: str):
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

def resample(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    df = df.set_index('Date').sort_index()
    idx = pd.date_range(df.index.min(), df.index.max(), freq='B')
    df = df.reindex(idx)
    df.index.name = 'Date'
    df[cols] = df[cols].ffill()
    return df.reset_index()

def make_sequences(X: np.ndarray, y: np.ndarray, window: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    Xs, ys = [], []
    for i in range(len(X) - window):
        Xs.append(X[i:(i + window)])
        ys.append(y[i + window])
    return np.array(Xs), np.array(ys)

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

def build_all_sequences(csv_list, spx_csv, fx_csv, cpi_csv, rate_csv, outdir, window=30):
    all_X_seq = []
    all_y_seq = []
    scalers_X = {}
    scalers_y = {}

    os.makedirs(outdir, exist_ok=True)

    for local_csv in csv_list:
        market_name = os.path.splitext(os.path.basename(local_csv))[0]
        print(f"Extracting: {market_name}")

        features = build_features(local_csv, spx_csv, fx_csv, cpi_csv, rate_csv)
        features = features.iloc[1:-1].reset_index(drop=True)

        X = features.drop(columns=['Date','y_t'])
        y = features['y_t']

        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_scaled = scaler_X.fit_transform(X)
        y_scaled = scaler_y.fit_transform(y.values.reshape(-1,1)).flatten()

        joblib.dump(scaler_X, os.path.join(outdir, f"scaler_X_{market_name}.pkl"))
        joblib.dump(scaler_y, os.path.join(outdir, f"scaler_y_{market_name}.pkl"))
        scalers_X[market_name] = scaler_X
        scalers_y[market_name] = scaler_y

        X_seq, y_seq = make_sequences(X_scaled, y_scaled, window=window)

        all_X_seq.append(X_seq)
        all_y_seq.append(y_seq)

    X_seq_all = np.concatenate(all_X_seq, axis=0)
    y_seq_all = np.concatenate(all_y_seq, axis=0)

    return X_seq_all, y_seq_all, scalers_X, scalers_y