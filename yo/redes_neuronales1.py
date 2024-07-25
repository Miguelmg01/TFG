# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 10:27:32 2024

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

params = ParameterSet(param_file)
test_admins = {
    'alendronate': medseq(a_dur=1, t_int=7, dose=70),
    'romosozumab': medseq(a_dur=1, t_int=30, dose=140),
    'denosumab': medseq(a_dur=1, t_int=180, dose=60)
}    
ages = [67, 68, 69]
num_pacientes = 6

#salidas esperadas del modelo
#estrategia medicacion,maximo cambio de valor de BMD, cambio de valor de BMD en 10 años
resultados_BMD = [['A -> R -> D', 0.051196154029266694, -0.07697014207212571],
 ['A -> D -> R', 0.04020642409769959, -0.0839894581241607],
 ['R -> A -> D', 0.0522584760579361, -0.07051776029252954],
 ['R -> D -> A', 0.04678137271416216, -0.057056276445954124],
 ['D -> A -> R', 0.033396967148129786, -0.07191260807387878],
 ['D -> R -> A', 0.033528449873000854, -0.05965596525608419]]




#dosis_alendronato = np.full(num_pacientes,70*152)
#dosis_romosozumab = np.full(num_pacientes,140*12)
#dosis_denosumab = np.full(num_pacientes,60*2)

#data_dosis = {
#    'Alendronate_Dosis': dosis_alendronato,
#    'Romosozumab_Dosis': dosis_romosozumab,
 #   'Denosumab_Dosis': dosis_denosumab
#}
#input_df = pd.DataFrame(data_dosis)

#bmd_max = np.array([])
#bmd_change = np.array([])
#for i in range(len(resultados_BMD)):
#    bmd_max= np.append(bmd_max,resultados_BMD[i][1])
#    bmd_change = np.append(bmd_change,resultados_BMD[i][2])
    
#output_df = pd.DataFrame({
#    'BMD_Max': bmd_max,
#    'BMD_Change': bmd_change
#})




#aqui en los array[tiempo de administracion,dosis]
#70 mg por semana durante un año de alendronato
#140 mg por mes durante un año de romosozumab
#60mg cada 6 meses por un año de denosumab
tratamientos = {'A -> R -> D': {'alendronate': np.array([[24455.,    70.],
         [24462.,    70.],         [24469.,    70.],         [24476.,    70.],
         [24483.,    70.],         [24490.,    70.],         [24497.,    70.],
         [24504.,    70.],         [24511.,    70.],         [24518.,    70.],
         [24525.,    70.],         [24532.,    70.],         [24539.,    70.],
         [24546.,    70.],         [24553.,    70.],         [24560.,    70.],
         [24567.,    70.],         [24574.,    70.],         [24581.,    70.],
         [24588.,    70.],         [24595.,    70.],         [24602.,    70.],
         [24609.,    70.],         [24616.,    70.],         [24623.,    70.],
         [24630.,    70.],         [24637.,    70.],         [24644.,    70.],
         [24651.,    70.],         [24658.,    70.],         [24665.,    70.],
         [24672.,    70.],         [24679.,    70.],         [24686.,    70.],
         [24693.,    70.],         [24700.,    70.],         [24707.,    70.],
         [24714.,    70.],         [24721.,    70.],         [24728.,    70.],
         [24735.,    70.],         [24742.,    70.],         [24749.,    70.],
         [24756.,    70.],         [24763.,    70.],         [24770.,    70.],
         [24777.,    70.],         [24784.,    70.],         [24791.,    70.],
         [24798.,    70.],         [24805.,    70.],
         [24812.,    70.],
         [24819.,    70.]]),
  'romosozumab': np.array([[24820.,   140.],
         [24850.,   140.],         [24880.,   140.],         [24910.,   140.],
         [24940.,   140.],         [24970.,   140.],         [25000.,   140.],
         [25030.,   140.],         [25060.,   140.],         [25090.,   140.],
         [25120.,   140.],         [25150.,   140.],         [25180.,   140.]]),
  'denosumab': np.array([[25185.,    60.],
         [25365.,    60.],         [25545.,    60.]])},
 'A -> D -> R': {'alendronate': np.array([[24455.,    70.],
         [24462.,    70.],         [24469.,    70.],         [24476.,    70.],
         [24483.,    70.],         [24490.,    70.],         [24497.,    70.],
         [24504.,    70.],         [24511.,    70.],         [24518.,    70.],
         [24525.,    70.],         [24532.,    70.],         [24539.,    70.],
         [24546.,    70.],         [24553.,    70.],         [24560.,    70.],
         [24567.,    70.],         [24574.,    70.],         [24581.,    70.],
         [24588.,    70.],         [24595.,    70.],         [24602.,    70.],
         [24609.,    70.],         [24616.,    70.],         [24623.,    70.],
         [24630.,    70.],         [24637.,    70.],         [24644.,    70.],
         [24651.,    70.],         [24658.,    70.],         [24665.,    70.],
         [24672.,    70.],         [24679.,    70.],         [24686.,    70.],
         [24693.,    70.],         [24700.,    70.],         [24707.,    70.],
         [24714.,    70.],         [24721.,    70.],         [24728.,    70.],
         [24735.,    70.],         [24742.,    70.],         [24749.,    70.],
         [24756.,    70.],         [24763.,    70.],         [24770.,    70.],
         [24777.,    70.],         [24784.,    70.],         [24791.,    70.],
         [24798.,    70.],         [24805.,    70.],         [24812.,    70.],
         [24819.,    70.]]),
  'romosozumab': np.array([[25185.,   140.],
         [25215.,   140.],         [25245.,   140.],         [25275.,   140.],
         [25305.,   140.],         [25335.,   140.],         [25365.,   140.],
         [25395.,   140.],         [25425.,   140.],         [25455.,   140.],
         [25485.,   140.],         [25515.,   140.],         [25545.,   140.]]),
  'denosumab': np.array([[24820.,    60.],
         [25000.,    60.],         [25180.,    60.]])},
 'R -> A -> D': {'alendronate': np.array([[24820.,    70.],
         [24827.,    70.],         [24834.,    70.],         [24841.,    70.],
         [24848.,    70.],         [24855.,    70.],         [24862.,    70.],
         [24869.,    70.],         [24876.,    70.],         [24883.,    70.],
         [24890.,    70.],         [24897.,    70.],         [24904.,    70.],
         [24911.,    70.],         [24918.,    70.],         [24925.,    70.],
         [24932.,    70.],         [24939.,    70.],         [24946.,    70.],
         [24953.,    70.],         [24960.,    70.],         [24967.,    70.],
         [24974.,    70.],         [24981.,    70.],         [24988.,    70.],
         [24995.,    70.],         [25002.,    70.],         [25009.,    70.],
         [25016.,    70.],         [25023.,    70.],         [25030.,    70.],
         [25037.,    70.],         [25044.,    70.],         [25051.,    70.],
         [25058.,    70.],         [25065.,    70.],         [25072.,    70.],
         [25079.,    70.],         [25086.,    70.],         [25093.,    70.],
         [25100.,    70.],         [25107.,    70.],         [25114.,    70.],
         [25121.,    70.],         [25128.,    70.],         [25135.,    70.],
         [25142.,    70.],         [25149.,    70.],         [25156.,    70.],
         [25163.,    70.],         [25170.,    70.],         [25177.,    70.],
         [25184.,    70.]]),
  'romosozumab': np.array([[24455.,   140.],
         [24485.,   140.],         [24515.,   140.],         [24545.,   140.],
         [24575.,   140.],         [24605.,   140.],         [24635.,   140.],
         [24665.,   140.],         [24695.,   140.],         [24725.,   140.],
         [24755.,   140.],         [24785.,   140.],         [24815.,   140.]]),
  'denosumab': np.array([[25185.,    60.],
         [25365.,    60.],         [25545.,    60.]])},
 'R -> D -> A': {'alendronate': np.array([[25185.,    70.],
         [25192.,    70.],         [25199.,    70.],         [25206.,    70.],
         [25213.,    70.],         [25220.,    70.],         [25227.,    70.],
         [25234.,    70.],         [25241.,    70.],         [25248.,    70.],
         [25255.,    70.],         [25262.,    70.],         [25269.,    70.],
         [25276.,    70.],         [25283.,    70.],         [25290.,    70.],
         [25297.,    70.],         [25304.,    70.],         [25311.,    70.],
         [25318.,    70.],         [25325.,    70.],         [25332.,    70.],
         [25339.,    70.],         [25346.,    70.],         [25353.,    70.],
         [25360.,    70.],         [25367.,    70.],         [25374.,    70.],
         [25381.,    70.],         [25388.,    70.],         [25395.,    70.],
         [25402.,    70.],         [25409.,    70.],         [25416.,    70.],
         [25423.,    70.],         [25430.,    70.],         [25437.,    70.],
         [25444.,    70.],         [25451.,    70.],         [25458.,    70.],
         [25465.,    70.],         [25472.,    70.],         [25479.,    70.],
         [25486.,    70.],         [25493.,    70.],         [25500.,    70.],
         [25507.,    70.],         [25514.,    70.],         [25521.,    70.],
         [25528.,    70.],         [25535.,    70.],         [25542.,    70.],
         [25549.,    70.]]),
  'romosozumab': np.array([[24455.,   140.],
         [24485.,   140.],         [24515.,   140.],         [24545.,   140.],
         [24575.,   140.],         [24605.,   140.],         [24635.,   140.],
         [24665.,   140.],         [24695.,   140.],         [24725.,   140.],
         [24755.,   140.],         [24785.,   140.],         [24815.,   140.]]),
  'denosumab': np.array([[24820.,    60.],
         [25000.,    60.],         [25180.,    60.]])},
 'D -> A -> R': {'alendronate': np.array([[24820.,    70.],
         [24827.,    70.],         [24834.,    70.],         [24841.,    70.],
         [24848.,    70.],         [24855.,    70.],         [24862.,    70.],
         [24869.,    70.],         [24876.,    70.],         [24883.,    70.],
         [24890.,    70.],         [24897.,    70.],         [24904.,    70.],
         [24911.,    70.],         [24918.,    70.],         [24925.,    70.],
         [24932.,    70.],         [24939.,    70.],         [24946.,    70.],
         [24953.,    70.],         [24960.,    70.],         [24967.,    70.],
         [24974.,    70.],         [24981.,    70.],         [24988.,    70.],
         [24995.,    70.],         [25002.,    70.],         [25009.,    70.],
         [25016.,    70.],         [25023.,    70.],         [25030.,    70.],
         [25037.,    70.],         [25044.,    70.],         [25051.,    70.],
         [25058.,    70.],         [25065.,    70.],         [25072.,    70.],
         [25079.,    70.],         [25086.,    70.],         [25093.,    70.],
         [25100.,    70.],         [25107.,    70.],         [25114.,    70.],
         [25121.,    70.],         [25128.,    70.],         [25135.,    70.],
         [25142.,    70.],         [25149.,    70.],         [25156.,    70.],
         [25163.,    70.],         [25170.,    70.],         [25177.,    70.],
         [25184.,    70.]]),
  'romosozumab': np.array([[25185.,   140.],
         [25215.,   140.],         [25245.,   140.],         [25275.,   140.],
         [25305.,   140.],         [25335.,   140.],         [25365.,   140.],
         [25395.,   140.],         [25425.,   140.],         [25455.,   140.],
         [25485.,   140.],         [25515.,   140.],         [25545.,   140.]]),
  'denosumab': np.array([[24455.,    60.],
         [24635.,    60.],         [24815.,    60.]])},
 'D -> R -> A': {'alendronate': np.array([[25185.,    70.],
         [25192.,    70.],         [25199.,    70.],         [25206.,    70.],
         [25213.,    70.],         [25220.,    70.],         [25227.,    70.],
         [25234.,    70.],         [25241.,    70.],         [25248.,    70.],
         [25255.,    70.],         [25262.,    70.],         [25269.,    70.],
         [25276.,    70.],         [25283.,    70.],         [25290.,    70.],
         [25297.,    70.],         [25304.,    70.],         [25311.,    70.],
         [25318.,    70.],         [25325.,    70.],         [25332.,    70.],
         [25339.,    70.],         [25346.,    70.],         [25353.,    70.],
         [25360.,    70.],         [25367.,    70.],         [25374.,    70.],
         [25381.,    70.],         [25388.,    70.],         [25395.,    70.],
         [25402.,    70.],         [25409.,    70.],         [25416.,    70.],
         [25423.,    70.],         [25430.,    70.],         [25437.,    70.],
         [25444.,    70.],         [25451.,    70.],         [25458.,    70.],
         [25465.,    70.],         [25472.,    70.],         [25479.,    70.],
         [25486.,    70.],         [25493.,    70.],         [25500.,    70.],
         [25507.,    70.],         [25514.,    70.],         [25521.,    70.],
         [25528.,    70.],         [25535.,    70.],         [25542.,    70.],
         [25549.,    70.]]),
  'romosozumab': np.array([[24820.,   140.],
         [24850.,   140.],         [24880.,   140.],         [24910.,   140.],
         [24940.,   140.],         [24970.,   140.],         [25000.,   140.],
         [25030.,   140.],         [25060.,   140.],         [25090.,   140.],
         [25120.,   140.],         [25150.,   140.],         [25180.,   140.]]),
  'denosumab': np.array([[24455.,    60.],
         [24635.,    60.],         [24815.,    60.]])}}




#ChatGpt
num_samples = 6

# Generar datos de ejemplo aleatorios para las características de entrada
np.random.seed(0)

# Pre-osteoblastos: número de células precursoras por unidad de volumen (valor de ejemplo)
pre_osteoblastos = np.random.rand(num_samples) * 100

# Pre-osteoclastos: número de células precursoras por unidad de volumen (valor de ejemplo)
pre_osteoclastos = np.random.rand(num_samples) * 100

# Osteoclastos: número de células por unidad de volumen (valor de ejemplo)
osteoclastos = np.random.rand(num_samples) * 100

# Osteoblastos: número de células por unidad de volumen (valor de ejemplo)
osteoblastos = np.random.rand(num_samples) * 100

# Osteocitos: número de células por unidad de volumen (valor de ejemplo)
osteocitos = np.random.rand(num_samples) * 100

# Esclerostina: concentración de esclerostina en sangre (valor de ejemplo en ng/mL)
esclerostina = np.random.rand(num_samples) * 10

# Densidad ósea total (BMD): densidad mineral ósea en g/cm^2 (valor de ejemplo)
densidad_osea_total = np.random.rand(num_samples) * 1 + 1  # valores entre 1 y 2 g/cm^2

# Contenido mineral óseo (BMC): contenido mineral óseo en gramos (valor de ejemplo)
contenido_mineral_oseo = np.random.rand(num_samples) * 2 + 1  # valores entre 1 y 3 g

# Tiempo total en años (considerando 10 años de seguimiento)
tiempo = np.full(num_samples, 10)

# Eficacia de los medicamentos y media vida (valores de ejemplo)
# Estos valores deberían ser obtenidos de estudios clínicos y literatura científica

# Eficacia del alendronato (valor de ejemplo)
eficacia_alendronate = np.random.uniform(0.7, 0.9, num_samples)  # valor entre 0.7 y 0.9

# Media vida del alendronato en años (valor de ejemplo)
media_vida_alendronate = np.random.uniform(7, 11, num_samples)  # valor entre 7 y 11 años

# Eficacia del romosozumab (valor de ejemplo)
eficacia_romosozumab = np.random.uniform(0.6, 0.8, num_samples)  # valor entre 0.6 y 0.8

# Media vida del romosozumab en años (valor de ejemplo)
media_vida_romosozumab = np.random.uniform(4, 6, num_samples)  # valor entre 4 y 6 años

# Eficacia del denosumab (valor de ejemplo)
eficacia_denosumab = np.random.uniform(0.8, 1.0, num_samples)  # valor entre 0.8 y 1.0

# Media vida del denosumab en años (valor de ejemplo)
media_vida_denosumab = np.random.uniform(5, 7, num_samples)  # valor entre 5 y 7 años

# Crear el DataFrame de entrada
data = {
    'Pre_Osteoblastos': pre_osteoblastos,
    'Pre_Osteoclastos': pre_osteoclastos,
    'Osteoclastos': osteoclastos,
    'Osteoblastos': osteoblastos,
    'Osteocitos': osteocitos,
    'Esclerostina': esclerostina,
    'Densidad_Osea_Total': densidad_osea_total,
    'Contenido_Mineral_Oseo': contenido_mineral_oseo,
    'Tiempo': tiempo,
    'Eficacia_Alendronate': eficacia_alendronate,
    'Media_Vida_Alendronate': media_vida_alendronate,
    'Eficacia_Romosozumab': eficacia_romosozumab,
    'Media_Vida_Romosozumab': media_vida_romosozumab,
    'Eficacia_Denosumab': eficacia_denosumab,
    'Media_Vida_Denosumab': media_vida_denosumab
}

input_df = pd.DataFrame(data)

output_df = pd.DataFrame(resultados_BMD, columns=['Estrategia_Medicacion', 'BMD_Max', 'BMD_Change'])


#iteraciones = 200

#capa_entrada = tf.keras.layers.Dense(units=3,input_shape=[3])
#capa_salida = tf.keras.layers.Dense(units=2)

#modelo = tf.keras.Sequential([capa_entrada,capa_salida])

#modelo.compile(
#    optimizer=tf.keras.optimizers.Adam(0.1),
#    loss='mean_squared_error',
#    metrics =[ 'mean_squared_error', 'mean_absolute_error']
#)


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Convertir los DataFrames a arrays de NumPy
X = input_df.values
y = output_df[['BMD_Max', 'BMD_Change']].values

# Crear el modelo de la red neuronal
model = Sequential()
model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2))  # Salida con 2 neuronas para BMD_Max y BMD_Change

# Compilar el modelo
model.compile(optimizer='adam', loss='mse')

# Entrenar el modelo
historial = model.fit(X, y, epochs=100, batch_size=15)


#EPOCHS= 500
#validation=0.3

print('Inicio de entrenamiento...')
#historial = modelo.fit(input_df,output_df,epochs=100,verbose=False)
#early_stop = tf.keras.callbacks.EarlyStopping(monitor='mean_squared_error',
#                                              patience =10 , min_delta =0.08 , mode ='min')

#historial = modelo.fit(input_df,output_df,epochs =EPOCHS,callbacks=[early_stop],
                    #validation_split =validation , verbose =1 ,batch_size=1024)
print('modelo entrenado')


plt.xlabel('# epoca')
plt.ylabel('Magnitud de perdida')
plt.plot(historial.history['loss'])


# Evaluar el modelo (ejemplo de evaluación, necesitarías un conjunto de datos de prueba real)
loss = model.evaluate(X, y)
print('Pérdida del modelo:', loss)

def modelo_predict(model,entrada_dato):
    sol = model.predict(entrada_dato)
    return sol









