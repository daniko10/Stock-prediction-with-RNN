import os

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers, Input

def build_lstm(input_shape, dropout, days_to_predict):
    model = models.Sequential([
        Input(shape=input_shape),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(dropout),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(dropout),
        layers.Dense(16, activation='relu'),
        layers.Dense(days_to_predict)
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model