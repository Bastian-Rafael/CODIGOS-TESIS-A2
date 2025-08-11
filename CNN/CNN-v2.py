# -*- coding: utf-8 -*-
"""
Optimized CNN training pipeline con validación y parada temprana
 - batch_size ajustado a 64
 - mixed precision
 - tf.data pipeline para entrenamiento y validación
 - callbacks: ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
 - arquitectura de modelo SIN cambios
"""
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import mixed_precision
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# 1) Configurar crecimiento dinámico de GPU y mixed precision
#gpus = tf.config.list_physical_devices('GPU')
#if gpus:
#    for gpu in gpus:
#        tf.config.experimental.set_memory_growth(gpu, True)

mixed_precision.set_global_policy('mixed_float16')
Y = np.loadtxt('simulations/Y_train.csv', delimiter=',').astype('float32')
#X_train_1 = np.loadtxt('simulatons/X_train_1.csv'), delimiter=',').astype('float32')
X2 = np.loadtxt('simulations/X_train_2.csv', delimiter=',').astype('float32')
X3 = np.loadtxt('simulations/X_train_3.csv', delimiter=',').astype('float32')

Xt2 = X2**2

X = np.stack((X2, X3, Xt2), axis=2)  # (n_samples, timesteps, 2 canales)

# 3) Parámetros de entrenamiento
epochs = 120
batch_size = 64
validation_split = 0.2  # 20% para validación

# 4) Definir y entrenar modelo
def ML_CNN(X, Y):
    # 4.1) Separar train/validation
    X_train, X_val, Y_train, Y_val = train_test_split(
        X, Y, test_size=validation_split, random_state=42, shuffle=True
    )

    # 4.2) Construir pipelines tf.data
    def make_ds(features, targets, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((features, targets))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(features))
        return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(X_train, Y_train, shuffle=True)
    val_ds   = make_ds(X_val,   Y_val,   shuffle=False)

    # 4.3) Definir arquitectura (idéntica a la original)
    model = Sequential([
        Conv1D(1024, 2, padding='same', activation='relu', input_shape=(X.shape[1], 3)),
        MaxPooling1D(3),
        Conv1D(1024, 4, padding='same', activation='relu'),
        MaxPooling1D(),
        Conv1D(1024, 12, padding='same', activation='relu'),
        MaxPooling1D(2),
        Conv1D(1024, 12, padding='same', activation='relu'),
        MaxPooling1D(2),
        Conv1D(1024, 6, padding='same', activation='relu'),
        MaxPooling1D(2),
        Conv1D(1024, 6, padding='same', activation='relu'),
        MaxPooling1D(2),
        Conv1D(1024, 6, padding='same', activation='relu'),
        Flatten(),
        Dropout(0.25),
        Dense(128, activation='relu'),
        *[Dense(256, activation='relu') for _ in range(6)],
        Dropout(0.25),
        Dense(4, activation='linear', dtype='float32')
    ])

    # 4.4) Compilar
    model.compile(
        loss='mean_squared_error',
        optimizer=Adam(),
        metrics=['mse']
    )

    # 4.5) Callbacks que monitorean val_loss
    checkpoint_cb = ModelCheckpoint(
        'CNN_best.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1
    )
    early_stop_cb = EarlyStopping(
        monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1
    )
    callbacks = [checkpoint_cb, early_stop_cb, reduce_lr_cb]

    # 4.6) Ajuste
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )

    # 4.7) Guardar modelo final
    model.save('CNN_model.keras')
    return model, history

# Ejecutar entrenamiento
model, history = ML_CNN(X, Y)

# Función de predicción
def ML_predict(X_new, model):
    return model.predict(X_new, batch_size=1)
