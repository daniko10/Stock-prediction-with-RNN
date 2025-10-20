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
        "data/mwig40_d.csv",
        "data/wig30_d.csv"
    ]
    outdir = "runs/multi_run"

    for window_size in [30, 60, 90]:
      for dropout in [0.3, 0.4, 0.5]:
        for batch_size in [64, 32, 16, 8]:
          X_train, y_train = build_all_sequences(
              instruments, spx_csv, fx_csv, cpi_csv, rate_csv,
              window=window_size, outdir=outdir
          )

          model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]), dropout=dropout)

          early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)

          model.fit(X_train, y_train, epochs=60, batch_size=batch_size, validation_split=0.1, shuffle=False, callbacks=[early_stop])

          model.save(os.path.join(outdir, f"model_BS_{batch_size}_WS_{window_size}_D_{dropout}.h5"))
    
if __name__ == '__main__':
    main()