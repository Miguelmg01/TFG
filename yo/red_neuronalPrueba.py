# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:01:14 2024

@author: Usuario
"""

#REDES NEURONALES
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import sys, os
import pandas as pd


#importamos el modelo
sys.path.insert(0, os.pardir)
from model.model import OsteoporosisModel, ParameterSet
from model.test_treatments import medseq
from model.tools import param_file, results_path, nft, plot_settings, colours, format_label


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split


P = ParameterSet(param_file)
test_admins = {
    'alendronate': medseq(a_dur=1, t_int=7, dose=70),
    'romosozumab': medseq(a_dur=1, t_int=30, dose=140),
    'denosumab': medseq(a_dur=1, t_int=180, dose=60)
}    


#salidas esperadas del modelo
#estrategia medicacion,maximo cambio de valor de BMD, cambio de valor de BMD en 10 años
resultados_BMD = [['A -> R -> D', 0.051196154029266694, -0.07697014207212571],
 ['A -> D -> R', 0.04020642409769959, -0.0839894581241607],
 ['R -> A -> D', 0.0522584760579361, -0.07051776029252954],
 ['R -> D -> A', 0.04678137271416216, -0.057056276445954124],
 ['D -> A -> R', 0.033396967148129786, -0.07191260807387878],
 ['D -> R -> A', 0.033528449873000854, -0.05965596525608419]]


num_samples = 6

parametros = P.params
dt = {}
for key,values in parametros.items():
    if type(values)==float:
        dt[key]= values
    else:
        if values=='True':
            dt[key]=1 
        else:
            dt[key]=0

# Crear el DataFrame de entrada
data = {
    'E_alendronate': np.full(num_samples, dt['E_alendronate']),
    'E_blosozumab': np.full(num_samples, dt['E_blosozumab']),
    'E_denosumab': np.full(num_samples, dt['E_denosumab']),
    'E_romosozumab': np.full(num_samples, dt['E_romosozumab']),
    'E_teriparatide': np.full(num_samples, dt['E_teriparatide']),
    'T_alendronate': np.full(num_samples, dt['T_alendronate']),
    'T_blosozumab': np.full(num_samples, dt['T_blosozumab']),
    'T_denosumab': np.full(num_samples, dt['T_denosumab']),
    'T_romosozumab': np.full(num_samples, dt['T_romosozumab']),
    'T_teriparatide': np.full(num_samples, dt['T_teriparatide']),
    'a_e': np.full(num_samples, dt['a_e']),
    'beta_B_pth': np.full(num_samples, dt['beta_B_pth']),
    'beta_b_rAb': np.full(num_samples, dt['beta_b_rAb']),
    'beta_pC_pth': np.full(num_samples, dt['beta_pC_pth']),
    'beta_pC_rAb': np.full(num_samples, dt['beta_pC_rAb']),
    'c_0': np.full(num_samples, dt['c_0']),
    'delta_s': np.full(num_samples, dt['delta_s']),
    'e_C': np.full(num_samples, dt['e_C']),
    'e_pC': np.full(num_samples, dt['e_pC']),
    'e_s': np.full(num_samples, dt['e_s']),
    'estrogen_decline': np.full(num_samples, dt['estrogen_decline']),
    'eta_B': np.full(num_samples, dt['eta_B']),
    'eta_C': np.full(num_samples, dt['eta_C']),
    'eta_C_bp': np.full(num_samples, dt['eta_C_bp']),
    'eta_Y': np.full(num_samples, dt['eta_Y']),
    'gamma': np.full(num_samples, dt['gamma']),
    'init_age': np.full(num_samples, dt['init_age']),
    'kappa_s': np.full(num_samples, dt['kappa_s']),
    'lambda_B': np.full(num_samples, dt['lambda_B']),
    'lambda_C': np.full(num_samples, dt['lambda_C']),
    'n': np.full(num_samples, dt['n']),
    'nu_C': np.full(num_samples, dt['nu_C']),
    'nu_Omega': np.full(num_samples, dt['nu_Omega']),
    'omega_B': np.full(num_samples, dt['omega_B']),
    'omega_pB': np.full(num_samples, dt['omega_pB']),
    'omega_pC': np.full(num_samples, dt['omega_pC']),
    'q_BSAP': np.full(num_samples, dt['q_BSAP']),
    'q_CTX': np.full(num_samples, dt['q_CTX']),
    'q_P1NP': np.full(num_samples, dt['q_P1NP']),
    'r_C': np.full(num_samples, dt['r_C']),
    'r_Omega': np.full(num_samples, dt['r_Omega']),
    'r_pB': np.full(num_samples, dt['r_pB']),
    's_Omega': np.full(num_samples, dt['s_Omega']),
    's_pB': np.full(num_samples, dt['s_pB']),
    's_pC': np.full(num_samples, dt['s_pC']),
    'sclerostin_increase': np.full(num_samples, dt['sclerostin_increase']),
    'tau_e': np.full(num_samples, dt['tau_e'])
}

input_df = pd.DataFrame(data)

#Esto serian los datos de salida de la red ,pero no se si plantearlo asi
output_df = pd.DataFrame(resultados_BMD, columns=['Estrategia_Medicacion', 'BMD_Max', 'BMD_Change'])



X = input_df.values
y = output_df[['BMD_Max', 'BMD_Change']].values

# Crear el modelo de la red neuronal
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation=tf.keras.activations.sigmoid, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(32, activation=tf.keras.activations.sigmoid, kernel_regularizer=l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(16, activation=tf.keras.activations.sigmoid, kernel_regularizer=l2(0.01)))
model.add(Dense(2))  # Salida con 2 neuronas para BMD_Max y BMD_Change



# Compilar el modelo
model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mean_squared_error',
              metrics =[ 'mean_squared_error', 'mean_absolute_error'])

# Entrenar el modelo
# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print('Inicio de entrenamiento...')
historial = model.fit(X_train, y_train, epochs=500,verbose = False, batch_size=10,validation_split=0.2)
print('modelo entrenado')


plt.xlabel('# epoca')
plt.ylabel('Magnitud de perdida')
plt.plot(historial.history['loss'])
plt.show()
plt.close()


# Evaluar el modelo 
loss = model.evaluate(X_test, y_test)
print('Pérdida del modelo:', loss)

y_pred = model.predict(X_test)
plt.scatter(y_test, y_pred)
plt.title("Actual vs. Predicted")
plt.xlabel("Actual")
plt.ylabel("Predicted")

def modelo_predict(model,entrada_dato):
    sol = model.predict(entrada_dato)
    return sol