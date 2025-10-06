import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler

from read_files import read_stock_data, read_rate, read_cpi, read_exchange
from helper import resample, make_sequences, build_features, build_all_sequences
from model import build_lstm

from tensorflow.keras.callbacks import EarlyStopping

def main():
    spx_csv = "data/spx_d.csv"
    fx_csv = "data/usdpln_d.csv"
    cpi_csv = "data/miesieczne_wskazniki_cen_towarow_i_uslug_konsumpcyjnych_od_1982roku.csv"
    rate_csv = "data/stopy_ref.csv"

    instruments = [
        "data/wig20_d.csv",
        "data/wig_d.csv",
        "data/mwig40_d.csv"
    ]

    outdir = "runs/multi_run"
    window_size = 30

    X_seq, y_seq, scalers_X, scalers_y = build_all_sequences(
        instruments, spx_csv, fx_csv, cpi_csv, rate_csv,
        window=window_size, outdir=outdir
    )

    split = int(len(X_seq) * 0.8)
    X_train, X_test = X_seq[:split], X_seq[split:]
    y_train, y_test = y_seq[:split], y_seq[split:]

    model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]))

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

    model.fit(X_train, y_train, epochs=60, batch_size=16, validation_split=0.1, callbacks=[early_stop])

    model.save(os.path.join(outdir, "model.h5"))
    print("Model saved to ", os.path.join(outdir, "model.h5"))
    
if __name__ == '__main__':
    main()