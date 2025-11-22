import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from constants import local_csv

from read_files import read_stock_data, read_rate, read_cpi, read_exchange
from helper_functions import resample, make_sequences, build_features, build_all_sequences
from model_structure import build_lstm

from tensorflow.keras.callbacks import EarlyStopping
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_and_save_models(X_train, y_train, outdir, window_size, days_to_predict, instrument_name):
    for dropout in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for batch_size in [1028, 512, 256, 128, 64, 32]:
            logging.info(f"Training model with window_size={window_size}, dropout={dropout}, batch_size={batch_size}")

            early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True, verbose=1)

            model = build_lstm(input_shape=(X_train.shape[1], X_train.shape[2]), dropout=dropout, days_to_predict=days_to_predict)
            model.fit(
                X_train, y_train, epochs=500, batch_size=batch_size, 
                validation_split=0.1, shuffle=False, callbacks=[early_stop]
            )

            model_path = os.path.join(outdir, f"model_BS_{batch_size}_WS_{window_size}_D_{dropout}_{instrument_name}.h5")
            model.save(model_path)
            logging.info(f"Model saved to {model_path}")

def main():
    spx_csv = "data/spx_d.csv"
    fx_csv = "data/usdpln_d.csv"
    cpi_csv = "data/miesieczne_wskazniki_cen_towarow_i_uslug_konsumpcyjnych_od_1982roku.csv"
    rate_csv = "data/stopy_ref.csv"
    outdir_scalers = "scalers/"
    outdir_models = "models/"
    
    days_to_predict = 10

    for window_size in [120, 90, 60, 30]:
        logging.info(f"Building sequences for window_size={window_size}")
        X_train, y_train = build_all_sequences(
            local_csv, spx_csv, fx_csv, cpi_csv, rate_csv,
            outdir_scalers, window_size, days_to_predict
        )
        train_and_save_models(X_train, y_train, outdir_models, window_size, days_to_predict, os.path.splitext(os.path.basename(local_csv))[0])

if __name__ == '__main__':
    main()