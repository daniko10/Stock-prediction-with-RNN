import xml.etree.ElementTree as ET
import csv
import pandas as pd
import numpy as np
from typing import List, Tuple

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