import os

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers, regularizers, Input

def build_lstm(input_shape, dropout=0.3):
    model = models.Sequential([
        Input(shape=input_shape), # (window, features)
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)), # musze zwrócić sekwencje moich window_size kroków [h1, h2, .., h_ws] dla kolejnej warstwy
        layers.Dropout(dropout),
        layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
        layers.Dropout(dropout),
        layers.Bidirectional(layers.LSTM(32, return_sequences=False)), # zwracam tylko ostatni krok [h_ws]
        layers.Dropout(dropout),
        layers.Dense(16, activation='relu'),
        layers.Dense(1)
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model