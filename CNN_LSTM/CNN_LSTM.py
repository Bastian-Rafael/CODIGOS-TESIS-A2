#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Programa CNN-1D + LSTM más profundo y denso
Entrenable en datos normalizados con mayor capacidad
"""
import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv1D, MaxPooling1D,
    LSTM, Dropout,
    Dense, BatchNormalization
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (                    # ← asegúrate de importar
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau      # ← ReduceLROnPlateau ya está aquí
)

# 1) Configuración GPU (opcional)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# 2) Cargar datos desde ./simulations
csv_dir = os.path.join(os.getcwd(), 'simulations')
Y  = np.loadtxt(os.path.join(csv_dir, 'Y_train.csv'),
                delimiter=',').astype('float32')
X2 = np.loadtxt(os.path.join(csv_dir, 'X_train_2.csv'),
                delimiter=',').astype('float32')
X3 = np.loadtxt(os.path.join(csv_dir, 'X_train_3.csv'),
                delimiter=',').astype('float32')

# 3) Preparar entradas: dos canales + canal cuadrático
Xt2 = X2**2
X   = np.stack([X2, X3, Xt2], axis=-1)  # (n_samples, timesteps, 3)

# 4) División train/validation
X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42, shuffle=True
)

# 5) Parámetros de entrenamiento
batch_size = 32
epochs     = 120
lr         = 1e-4

# 6) Pipelines tf.data
def make_ds(features, targets, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((features, targets))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(features))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_ds(X_train, Y_train, shuffle=True)
val_ds   = make_ds(X_val,   Y_val,   shuffle=False)

# 7) Definir modelo más profundo y denso
timesteps, channels = X_train.shape[1], X_train.shape[2]
n_outputs = Y_train.shape[1]

model = Sequential([
    Conv1D(512, kernel_size=3, padding='same', activation='relu',
           input_shape=(timesteps, channels)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(512, kernel_size=6, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(512, kernel_size=12, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(512, kernel_size=12, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(512, kernel_size=6, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(512, kernel_size=6, padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    LSTM(256, return_sequences=True, unroll=True),
    Dropout(0.4),
    LSTM(128, return_sequences=False, unroll=True),
    Dropout(0.4),

    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.4),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.35),

    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.35),

    Dense(n_outputs, activation='linear', dtype='float32')
])

# 8) Compilar con clipping
opt = Adam(learning_rate=lr, clipnorm=1.0)
model.compile(
    optimizer=opt,
    loss='mean_squared_error',
    metrics=['mse']
)

# 9) Mostrar arquitectura
model.summary()

# 10) Callbacks
checkpoint_cb = ModelCheckpoint(
    'deep_cnn_lstm_best.keras',
    monitor='val_loss', save_best_only=True, verbose=1
)
early_stop_cb = EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True, verbose=1
)
reduce_lr_cb = ReduceLROnPlateau(                        # ← callback añadido
    monitor='val_loss',       # observa la métrica de validación
    factor=0.5,               # reduce LR multiplicando por 0.5
    patience=3,               # tras 3 épocas sin mejora
    min_lr=1e-6,              # no bajar de este LR mínimo
    verbose=1
)

# 11) Entrenamiento con ReduceLROnPlateau
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],  # ← incluido
    verbose=1
)

# 12) Guardar modelo final
model.save('deep_cnn_lstm_final.keras')

