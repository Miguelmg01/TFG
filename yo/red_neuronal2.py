# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:39:07 2024

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
from model.data import ClinicalTrialData
from model.model import OsteoporosisModel, ParameterSet
from model.test_treatments import medseq
from model.tools import param_file, results_path, nft, plot_settings, colours, format_label


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split


params = ParameterSet(param_file)
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

parametros = {
    'E_alendronate': 2.97e-05,
    'E_blosozumab': 0.013104958,
    'E_denosumab': 4339.630194,
    'E_romosozumab': 0.013104958,
    'E_teriparatide': 0.265529542,
    'T_alendronate': 152.710497,
    'T_blosozumab': 7.0,
    'T_denosumab': 10.0,
    'T_romosozumab': 7.0,
    'T_teriparatide': 0.0417,
    'a_e': 50.0,
    'beta_B_pth': 1.305332076,
    'beta_b_rAb': 0.015764123,
    'beta_pC_pth': 4.282514877,
    'beta_pC_rAb': 0.873373525,
    'c_0': 0.8,
    'delta_s': 0.05,
    'e_C': 0.990728213,
    'e_pC': 0.937654795,
    'e_s': 9.595406602,
    'estrogen_decline': True,
    'eta_B': 0.00867806,
    'eta_C': 0.023815993,
    'eta_C_bp': 1.000665052,
    'eta_Y': 0.000109589,
    'gamma': 0.006654662,
    'init_age': 0.0,
    'kappa_s': 0.05,
    'lambda_B': 1.29e-06,
    'lambda_C': 3.82e-06,
    'n': 1.0,
    'nu_C': 0.000123164,
    'nu_Omega': 107.5332737,
    'omega_B': 0.000624353,
    'omega_pB': 0.319241069,
    'omega_pC': 0.930890444,
    'q_BSAP': 0.923890022,
    'q_CTX': 1.160526444,
    'q_P1NP': 1.453273548,
    'r_C': 10.12488334,
    'r_Omega': 1024.4198,
    'r_pB': 111.9483271,
    's_Omega': 3039.645954,
    's_pB': 163.2824648,
    's_pC': 8603687.044,
    'sclerostin_increase': False,
    'tau_e': 2.6
}

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

output_df = pd.DataFrame(resultados_BMD, columns=['Estrategia_Medicacion', 'BMD_Max', 'BMD_Change'])


# Convertir los DataFrames a arrays de NumPy
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


#EPOCHS= 500
#validation=0.3

print('Inicio de entrenamiento...')
historial = model.fit(X_train, y_train, epochs=500,verbose = False, batch_size=10,validation_split=0.2)

#early_stop = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error',
#                                              patience =10 , min_delta =0.08 , mode ='min')

#historial = modelo.fit(input_df,output_df,epochs =EPOCHS,callbacks=[early_stop],
                    #validation_split =validation , verbose =1 ,batch_size=1024)
print('modelo entrenado')


plt.xlabel('# epoca')
plt.ylabel('Magnitud de perdida')
plt.plot(historial.history['loss'])


# Evaluar el modelo (ejemplo de evaluación, necesitarías un conjunto de datos de prueba real)
loss = model.evaluate(X_test, y_test)
print('Pérdida del modelo:', loss)

def modelo_predict(model,entrada_dato):
    sol = model.predict(entrada_dato)
    return sol