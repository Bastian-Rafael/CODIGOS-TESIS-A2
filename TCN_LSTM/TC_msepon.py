#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 12 06:00:06 2025
@author: basti

CNN–LSTM con TCN causal amplio y pérdida ponderada,
pero sin cuello de botella en el pipeline de datos:
– Reemplazamos el generator Python por un tf.data puro.
– Cálculo del canal cuadrático dentro de map().
– Paralelismo y prefetch automáticos.
Listo para ejecutar en Docker montando `/workspace`.
"""

import os, gc
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import mixed_precision
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, SpatialDropout1D, Add,
    LSTM, Dropout, Dense, Softmax, Multiply, Lambda
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
)

# 1) Mixed precision
mixed_precision.set_global_policy('mixed_float16')

# 2) Carga de datos (CSV → NPY → memmap)
csv_dir = 'simulations'
for name in ('X_train_2','X_train_3'):
    npy = os.path.join(csv_dir, f'{name}.npy')
    if not os.path.exists(npy):
        arr = np.loadtxt(os.path.join(csv_dir, f'{name}.csv'),
                         delimiter=',', dtype='float32')
        np.save(npy, arr.astype('float16'))

X2 = np.load(os.path.join(csv_dir, 'X_train_2.npy'), mmap_mode='r')  # float16
X3 = np.load(os.path.join(csv_dir, 'X_train_3.npy'), mmap_mode='r')  # float16
Y  = np.loadtxt(os.path.join(csv_dir, 'Y_train.csv'),
                delimiter=',').astype('float32')

n_params = Y.shape[1]
indices  = np.arange(Y.shape[0])
train_idx, val_idx = train_test_split(indices,
                                      test_size=0.2,
                                      random_state=42,
                                      shuffle=True)

timesteps   = X2.shape[1]
channels    = 3
batch_size  = 32       # batch más pequeño ayuda a la memoria
buffer_size = 10_000

# 3) tf.data pipeline sin generator Python
def pack_features(ch2, ch3, y):
    # calcula el canal cuadrático dentro del graph
    sq = tf.math.square(ch2)
    x  = tf.stack([ch2, ch3, sq], axis=-1)
    return x, y

# separar numpy con memmap por índices
X2_train, X2_val = X2[train_idx], X2[val_idx]
X3_train, X3_val = X3[train_idx], X3[val_idx]
Y_train,  Y_val  = Y[train_idx],  Y[val_idx]

# Dataset de entrenamiento
train_ds = tf.data.Dataset.from_tensor_slices(
    (X2_train, X3_train, Y_train)
).map(pack_features, num_parallel_calls=tf.data.AUTOTUNE
).shuffle(buffer_size
).batch(batch_size
).prefetch(tf.data.AUTOTUNE)

# Dataset de validación
val_ds = tf.data.Dataset.from_tensor_slices(
    (X2_val, X3_val, Y_val)
).map(pack_features, num_parallel_calls=tf.data.AUTOTUNE
).batch(batch_size
).prefetch(tf.data.AUTOTUNE)

steps_per_epoch  = len(train_idx)//batch_size
validation_steps = len(val_idx)  //batch_size

# 4) Bloque TCN causal
def temporal_block(x, filters, kernel_size, dilation):
    prev = x
    x = Conv1D(filters, kernel_size,
               padding='causal',
               dilation_rate=dilation,
               activation='relu')(x)
    x = BatchNormalization()(x)
    x = SpatialDropout1D(0.2)(x)
    x = Conv1D(filters, kernel_size,
               padding='causal',
               dilation_rate=dilation,
               activation='relu')(x)
    x = BatchNormalization()(x)
    if prev.shape[-1] != filters:
        prev = Conv1D(filters, 1, padding='same')(prev)
    return Add()([prev, x])

# 5) Atención temporal
def temporal_attention(inputs):
    score   = Dense(1)(inputs)
    weights = Softmax(axis=1)(score)
    context = Multiply()([inputs, weights])
    return Lambda(lambda z: K.sum(z, axis=1))(context)

# 6) Construcción del modelo
inp = Input(shape=(timesteps, channels), name='input_series')
x = inp

for d in (1, 2, 4, 8, 16, 32):
    x = temporal_block(x, filters=512, kernel_size=3, dilation=d)

# stack de LSTM
x = LSTM(512, return_sequences=True, unroll=True)(x)
x = Dropout(0.3)(x)
x = LSTM(512, return_sequences=True, unroll=True)(x)
x = Dropout(0.3)(x)

# atención y densas finales
x = temporal_attention(x)
for units, drop in [(512,0.4),(256,0.3),(128,0.3),(64,0.3)]:
    x = Dense(units, activation='relu', dtype='float32')(x)
    x = BatchNormalization()(x)
    x = Dropout(drop)(x)

out = Dense(n_params, activation='linear',
            dtype='float32', name='params')(x)
model = Model(inp, out, name='CNN_LSTM_TCN_expanded')

# 7) Pérdida ponderada y compilación
def weighted_mse(y_true, y_pred):
    weights = K.constant([1.0,1.0,5.0,1.0])
    return K.mean(weights * K.square(y_true - y_pred), axis=-1)

model.compile(
    optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
    loss=weighted_mse,
    metrics=['mse']
)

model.summary()

# 8) Callbacks
checkpoint_cb = ModelCheckpoint('best_model_TC_LSTM.keras',
                                monitor='val_loss',
                                save_best_only=True,
                                verbose=1)
early_stop_cb = EarlyStopping(monitor='val_loss',
                              patience=10,
                              restore_best_weights=True,
                              verbose=1)
reduce_lr_cb = ReduceLROnPlateau(monitor='val_loss',
                                 factor=0.5,
                                 patience=5,
                                 min_lr=1e-6,
                                 verbose=1)

# 9) Entrenamiento
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=250,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[checkpoint_cb, early_stop_cb, reduce_lr_cb],
    verbose=1
)

# 10) Limpieza
del train_ds, val_ds
gc.collect()
K.clear_session()
