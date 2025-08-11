#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 00:12:56 2025
@author: basti

Inferencia con CNN + ABC-Rejection
Usa GPU (si está disponible), mixed‐precision y memoria limitada.
Carga datos desde ./simulations y modelo CNN_best.keras
"""

import os
# (Opcional) forzar uso de la GPU número 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.models import load_model

# ─── 1) Configuración GPU ────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=6000)]
    )

# ─── 2) Mixed precision ─────────────────────────────────────────
mixed_precision.set_global_policy('mixed_float16')

# ─── 3) Cargo el modelo y los datos de test ─────────────────────
model = load_model("CNN_best.keras")

data_dir = "simulations"
Y  = np.loadtxt(os.path.join(data_dir, "Y_test.csv"),   delimiter=",").astype("float32")
X2 = np.loadtxt(os.path.join(data_dir, "X_test_2.csv"), delimiter=",").astype("float32")
X3 = np.loadtxt(os.path.join(data_dir, "X_test_3.csv"), delimiter=",").astype("float32")

# Apilo los dos canales y su cuadrado
Xt2 = X2**2
X   = np.stack((X2, X3, Xt2), axis=2)  # (n_samples, timesteps, 3)

# ─── 4) Funciones auxiliares ────────────────────────────────────
def generate_parameters():
    """Genera un tuple (omega, alpha, beta, delta) válido."""
    while True:
        omega = np.random.uniform(0, 1)
        alpha = np.random.uniform(0, 0.6)
        beta  = np.random.uniform(0, 0.6)
        delta = np.random.uniform(0, 0.6)
        if (alpha + beta < 1 and
            alpha + beta + delta < 1 and
            (3*alpha**2 + 2*alpha*beta + beta**2 < 1)):
            return omega, alpha, beta, delta

def observed_sample(n, omega, alpha, beta, delta):
    """
    Simula una serie de longitud `n` para los parámetros indicados.
    Devuelve un array de forma (n, 3) de tipo float32.
    """
    noise = np.random.normal(0, 1, size=(2, n)).astype("float32")
    Z, e = noise[0], noise[1]

    X = np.empty(n, dtype="float32")
    S = np.empty(n, dtype="float32")
    # Inicializamos el primer elemento sin usar .astype() sobre un float
    X[0] = np.random.rand()         # se castea automáticamente a float32 al asignar
    S[0] = np.random.rand()**2      # idem

    for i in range(1, n):
        x2, s2, z2 = X[i-1]**2, S[i-1]**2, Z[i-1]**2
        S[i] = np.sqrt(omega + alpha*x2 + beta*s2 + delta*z2)
        X[i] = S[i] * e[i]

    out = np.empty((n, 3), dtype="float32")
    out[:, 0], out[:, 1], out[:, 2] = X, Z, X**2
    return out


def ABC_rejection(S_test, S_obs, Y_test, acceptance_ratio=0.01):
    """ABC por distancia Euclídea."""
    distances = np.linalg.norm(S_test - S_obs, axis=1)
    n_accept  = max(1, int(len(distances)*acceptance_ratio))
    idx       = np.argsort(distances)[:n_accept]
    return pd.DataFrame(Y_test[idx], columns=["omega","alpha","beta","delta"])

def ABC_rejection_fast(S_train, S_obs, Y_train, acceptance_ratio=0.01):
    """ABC rápido con NearestNeighbors."""
    n_accept = max(1, int(len(S_train)*acceptance_ratio))
    nn       = NearestNeighbors(n_neighbors=n_accept).fit(S_train)
    _, idx   = nn.kneighbors(S_obs.reshape(1, -1), return_distance=True)
    return pd.DataFrame(Y_train[idx[0]], columns=["omega","alpha","beta","delta"])

# ─── 5) Predicción de estadísticas de resumen para todos los tests ───
#     no expandimos dims en el eje de muestras, pasamos X de forma (n_samples, timesteps, 3)
S_x_test = model.predict(X, batch_size=32, verbose=1)  # (n_samples, n_outputs)

# ─── 6) Configuro NearestNeighbors sobre la salida del test ──────────
I        = 10**3
df       = pd.DataFrame(columns=[
    "omega","omega_pos",
    "alpha","alpha_pos",
    "beta","beta_pos",
    "delta","delta_pos"
])
# Número de vecinos a aceptar
n_accept = max(1, int(len(S_x_test)*0.01))
nn       = NearestNeighbors(n_neighbors=n_accept).fit(S_x_test)

# ─── 7) Muestreo ABC ────────────────────────────────────────────────
for i in range(1, I+1):
    if i <= 5 or i % 100 == 0:
        print(f"iteración {i} de {I}", flush=True)

    omega, alpha, beta, delta = generate_parameters()
    X_obs = observed_sample(300, omega, alpha, beta, delta)       # (300,3)
    S_x_obs = model.predict(                                   
        X_obs[np.newaxis, ...],  # pasa (1,300,3)
        batch_size=1,
        verbose=0
    )[0]  # obtenemos (n_outputs,)

    # buscamos los vecinos más cercanos en S_x_test
    _, idx = nn.kneighbors(S_x_obs.reshape(1, -1))
    post_mean = Y[idx[0]].mean(axis=0)
    df.loc[len(df)] = [
        omega,      post_mean[0],
        alpha,      post_mean[1],
        beta,       post_mean[2],
        delta,      post_mean[3]
    ]

# ─── 8) Resultado final ────────────────────────────────────────────
df.to_csv('ABC_CNN.csv')
