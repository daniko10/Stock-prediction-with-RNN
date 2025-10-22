import pandas as pd
import os

def read_stock_data(path: str, is_testing = False) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
        
    colmap = {
        'Data': 'Date',
        'Zamkniecie': 'Close',
        'Wolumen': 'Volume'
    }
    
    df = pd.read_csv(path, sep=',')
    
    available = [c for c in colmap.keys() if c in df.columns]
        
    df = df[available]
    df.rename(columns={pl:en for pl,en in colmap.items() if pl in available}, inplace=True)
    
    if (not is_testing):
        df = df[(df['Date'] >= '2000-01-01') & (df['Date'] < '2024-01-01')]
    else:
        df = df[df['Date'] >= '2000-01-01']
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Close'] = pd.to_numeric(df['Close'])
    df['Volume'] = pd.to_numeric(df['Volume'])
    
    return df

def read_rate(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    df = pd.read_csv(path, sep=';')
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Ref'] = pd.to_numeric(df['Ref'])
    
    return df

def read_cpi(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    colmap = {
        'Rok': 'Year',
        'Miesiac': 'Month',
        'Wartosc': 'Value'
    }
    
    df = pd.read_csv(path, sep=';', usecols=colmap.keys(), decimal=',')
    df.rename(columns=colmap, inplace=True)
    
    df['Year'] = pd.to_numeric(df['Year'])
    df['Month'] = pd.to_numeric(df['Month'])
    df['Value'] = pd.to_numeric(df['Value']) - 100.0 # jak mam np 104 to znaczy ze wzrost o 4% w stosunku do roku poprzedniego
    
    df['Date'] = pd.to_datetime(df[['Year','Month']].assign(DAY=1))
    
    df = df[['Date','Value']].sort_values('Date').reset_index(drop=True)
    
    return df

def read_exchange(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    
    colmap = {
        'Data': 'Date',
        'Zamkniecie': 'Exchange Rate $'
    }
    
    df = pd.read_csv(path, sep=',', usecols=colmap.keys())
    df.rename(columns=colmap, inplace=True)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df['Exchange Rate $'] = pd.to_numeric(df['Exchange Rate $'])
    df['usd_change'] = df['Exchange Rate $'].pct_change()
    
    df = df[['Date','usd_change']].sort_values('Date').reset_index(drop=True)
    
    return df