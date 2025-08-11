#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN–LSTM con TCN causal reducido para no agotar GPU
- batch_size = 32
- sólo 4 bloques TCN en vez de 8
- filtros TCN = 256 en vez de 512
- LSTM de 256 unidades en vez de 512
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

def main():
    # 1) GPU + mixed precision
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    mixed_precision.set_global_policy('mixed_float16')

    # 2) Datos en /workspace/simulations
    csv_dir = os.path.join(os.getcwd(), 'simulations')
    # Convertir CSV→NPY (si hace falta)
    for name in ['X_train_2','X_train_3']:
        p = os.path.join(csv_dir, f'{name}.npy')
        if not os.path.exists(p):
            arr = np.loadtxt(os.path.join(csv_dir, f'{name}.csv'),
                             delimiter=',', dtype='float32')
            np.save(p, arr.astype('float16'))

    X2 = np.load(os.path.join(csv_dir, 'X_train_2.npy'), mmap_mode='r')
    X3 = np.load(os.path.join(csv_dir, 'X_train_3.npy'), mmap_mode='r')
    Y  = np.loadtxt(os.path.join(csv_dir, 'Y_train.csv'),
                    delimiter=',').astype('float32')

    # 3) Índices y parámetros
    n_params = Y.shape[1] if Y.ndim>1 else 1
    idx = np.arange(Y.shape[0])
    train_idx, val_idx = train_test_split(idx, test_size=0.2,
                                          random_state=42, shuffle=True)
    timesteps = X2.shape[1]
    channels  = 3

    # 4) Generador
    def data_gen(ids):
        for i in ids:
            ch2 = X2[i]; ch3 = X3[i]; sq = ch2**2
            yield np.stack([ch2, ch3, sq], axis=-1), Y[i]

    sig = (
      tf.TensorSpec((timesteps,channels), dtype=tf.float16),
      tf.TensorSpec((n_params,),      dtype=tf.float32)
    )

    # 5) Dataset con batch reducido
    batch_size = 32  
    train_ds = tf.data.Dataset.from_generator(lambda: data_gen(train_idx),
               output_signature=sig
             ).shuffle(5000).repeat().batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = tf.data.Dataset.from_generator(lambda: data_gen(val_idx),
             output_signature=sig
           ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    steps_per_epoch  = len(train_idx)//batch_size
    val_steps        = len(val_idx)//batch_size

    # 6) TCN causal aligerado
    def temporal_block(x, filters, kernel, dilation):
        prev = x
        x = Conv1D(filters, kernel, padding='causal', dilation_rate=dilation, activation='relu')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout1D(0.2)(x)
        x = Conv1D(filters, kernel, padding='causal',
                   dilation_rate=dilation, activation='relu')(x)
        x = BatchNormalization()(x)
        if prev.shape[-1]!=filters:
            prev = Conv1D(filters,1,padding='same')(prev)
        return Add()([prev,x])

    # 7) Construir modelo
    inp = Input((timesteps,channels), name='input_series')
    x = inp

    # Sólo 4 dilations y filtros=256
    for d in [1,2,4,8]:
        x = temporal_block(x, filters=256, kernel=3, dilation=d)

    # LSTM reducido
    x = LSTM(256, return_sequences=True, unroll=True)(x)
    x = Dropout(0.3)(x)
    x = LSTM(256, return_sequences=False, unroll=True)(x)
    x = Dropout(0.3)(x)

    # Fully-connected final
    for units, drop in [(256,0.4),(128,0.3),(64,0.3)]:
        x = Dense(units, activation='relu', dtype='float32')(x)
        x = BatchNormalization()(x)
        x = Dropout(drop)(x)

    out = Dense(n_params, activation='linear',
                dtype='float32', name='params')(x)
    model = Model(inp,out,name='small_CNN_LSTM_TCN')

    # 8) Compilar
    def weighted_mse(y_true,y_pred):
        w = K.constant([1.0,1.0,5.0,1.0])
        return K.mean(w*K.square(y_true-y_pred),axis=-1)

    model.compile(
        optimizer=Adam(learning_rate=1e-4, clipnorm=1.0),
        loss=weighted_mse,
        metrics=['mse']
    )

    model.summary()

    # 9) Callbacks (incluye ReduceLROnPlateau)
    checkpoint_cb = ModelCheckpoint('best_model.keras', save_best_only=True,
                                    monitor='val_loss', verbose=1)
    early_cb      = EarlyStopping(patience=10, monitor='val_loss',
                                  restore_best_weights=True, verbose=1)
    reduce_lr_cb  = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                      patience=5, min_lr=1e-6, verbose=1)

    # 10) Entrenar
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=250,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=[checkpoint_cb, early_cb, reduce_lr_cb],
        verbose=1
    )

    # 11) Limpieza
    del train_ds, val_ds; gc.collect(); K.clear_session()

if __name__=='__main__':
    main()
