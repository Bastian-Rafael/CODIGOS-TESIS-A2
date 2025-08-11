# -*- coding: utf-8 -*-
"""
Created on Tue Jul 29 00:39:31 2025

@author: basti
"""

"""
        Funciones
        1. Simulación de datos  
"""
import numpy as np 
import time
import os 
# --- Parameter Generation ---
def generate_parameters():
    while True:
        omega = np.random.uniform(0,1)
        alpha = np.random.uniform(0, 0.6)
        beta = np.random.uniform(0, 0.6)
        delta = np.random.uniform(0, 0.6)

        if alpha + beta < 1 and alpha + beta + delta < 1 and (3 * alpha**2 + 2 * alpha * beta + beta**2 < 1):
            return omega, alpha, beta, delta

# --- GARCH-X Sample Generation --- #
def observed_sample(n, omega, alpha, beta, delta):
    Z = np.random.normal(0, 1, n)
    e = np.random.normal(0, 1, n)
    X = np.zeros(n)
    S = np.zeros(n)
    X[0] = np.random.uniform(0, 1)
    S[0] = np.random.uniform(0, 1)**2
    for i in range(1, n):
        S[i] = np.sqrt(omega + alpha * X[i-1]**2 + beta * S[i-1]**2 + delta * Z[i-1]**2)
        X[i] = S[i] * e[i]

    return np.column_stack((np.arange(1, n+1), X, Z))


# --- Batch Simulation --- #
def simulate_tensor(N, n):
    X_path = []
    Y_params = []
        
    i = 0
    for i in range(N):
        omega, alpha, beta, delta = generate_parameters()
        path = observed_sample(n, omega, alpha, beta, delta)
        X_path.append(path)
        Y_params.append([omega, alpha, beta, delta])
        print(f"Simulación número {i+1} de {N}")
        
    X_path = np.array(X_path, dtype='float32')
    Y_params = np.array(Y_params, dtype='float32')
    
    N = len(X_path)
    train_size = N - int(N*0.3)
    #test_size = N - train_size
    Xp_train, Xp_test = X_path[0:train_size,:,:], X_path[train_size+1:N,:,:]
    Yp_train, Yp_test = Y_params[0:train_size,:], Y_params[train_size+1:N,:]
    
    #print("Training sample size:", train_size, "-- Test sample size:", test_size)
    
    return [Xp_train, Xp_test, Yp_train, Yp_test]

# Simulación de datos y parámetros
N_train = 10**6
n = 300 

#Medir el tiempo que demora la simulación 
start_time = time.time()
X_train, X_test , Y_train , Y_test = simulate_tensor(N_train, n)
end_time = time.time()
tiempo_proceso_1 = end_time-start_time
print(f"El proceso tardó {tiempo_proceso_1/60:.4f} minutos.")

output_path = r'F:/simulations'
os.makedirs(output_path, exist_ok=True)

def save_csv(array, filename):
    full_path = os.path.join(output_path, filename)
    np.savetxt(full_path, array, delimiter=',')
    
# Entrenamiento
save_csv(X_train[:, :, 0], 'X_train_1.csv')
save_csv(X_train[:, :, 1], 'X_train_2.csv')
save_csv(X_train[:, :, 2], 'X_train_3.csv')
save_csv(Y_train, 'Y_train.csv')

# Validación
save_csv(X_test[:, :, 0], 'X_test_1.csv')
save_csv(X_test[:, :, 1], 'X_test_2.csv')
save_csv(X_test[:, :, 2], 'X_test_3.csv')
save_csv(Y_test, 'Y_test.csv')