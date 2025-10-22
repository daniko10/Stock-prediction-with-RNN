import os

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers, Input

def build_lstm(input_shape, dropout, days_to_predict):
    model = Sequential([
        LSTM(64, return_sequences=False, input_shape=input_shape),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dense(days_to_predict)
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model