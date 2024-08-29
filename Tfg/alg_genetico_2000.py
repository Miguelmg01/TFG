# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 19:43:35 2024

@author: Miguel Madero
"""
import pickle
import numpy as np
import random
from datetime import datetime
import matplotlib.pyplot as plt


#random.seed(0)

# Definir la función de aptitud: el promedio de dmo
def calcular_aptitud(individuo):
    return np.mean(individuo['bmd'])

# Inicializar la población con los distintos esquemas de tratamiento
def inicializar_poblacion(data, esquemas, tamaño_poblacion):
    poblacion = []
    for _ in range(tamaño_poblacion):
        esquema = random.choice(esquemas)
        individuo = data[esquema]
        aptitud = calcular_aptitud(individuo)
        poblacion.append((esquema, aptitud))
    return poblacion

# Función de selección por torneo
def seleccion_torneo(poblacion, k=3):
    seleccionados = random.sample(poblacion, k)
    return max(seleccionados, key=lambda x: x[1])

# Función de cruzamiento
def cruzar(padre1, padre2, data):
    punto_cruce = random.randint(1, len(data[padre1]) - 1)
    hijo1_bmd = np.concatenate((data[padre1]['bmd'][:punto_cruce],\
                                data[padre2]['bmd'][punto_cruce:]))
    hijo2_bmd = np.concatenate((data[padre2]['bmd'][:punto_cruce],\
                                data[padre1]['bmd'][punto_cruce:]))
    
    hijo1 = {'bmd': hijo1_bmd}
    hijo2 = {'bmd': hijo2_bmd}
    
    return hijo1, hijo2

# Función de mutación
def mutar(individuo, tasa_mutacion=0.1):
    if random.random() < tasa_mutacion:
        indice = random.randint(0, len(individuo['bmd']) - 1)
        
        minm = np.min(individuo['bmd'])
        maxm = np.max(individuo['bmd'])
        individuo['bmd'][indice] = random.uniform(minm, maxm)
    return individuo

# Algoritmo Genético
def algoritmo_genetico(data, esquemas, generaciones, tamaño_poblacion):
    poblacion = inicializar_poblacion(data, esquemas, tamaño_poblacion)
    
    mejores_valores = []
    gen = []
    graficar =[]
    
    for i in range(generaciones):
        nueva_poblacion = []
        
        for _ in range(tamaño_poblacion // 2):
            # Selección
            padre1 = seleccion_torneo(poblacion)[0]
            padre2 = seleccion_torneo(poblacion)[0]
            
            # Cruzamiento
            hijo1, hijo2 = cruzar(padre1, padre2, data)
            
            # Mutación
            hijo1 = mutar(hijo1)
            hijo2 = mutar(hijo2)
            
            # Evaluación de la aptitud
            nueva_poblacion.append((padre1, calcular_aptitud(hijo1)))
            nueva_poblacion.append((padre2, calcular_aptitud(hijo2)))
        
        # Reemplazo de la población
        poblacion = sorted(nueva_poblacion + poblacion, \
                           key=lambda x: x[1], reverse=True)[:tamaño_poblacion]
        # reverse=True , nos ordena las tuplas por orden de mayor valor a menor
        #[:tamaño_poblacion] es para quedarnos con los que tienen mejor aptitud
        mejor_aptitud = max(poblacion, key=lambda x: x[1])[1]
        gen.append(i)
        mejores_valores.append(mejor_aptitud)
        
        #print(f"Generación {i+1} - Mejor Aptitud: {mejor_aptitud}")
        
    # Retornar el mejor individuo encontrado
    mejor_individuo = max(poblacion, key=lambda x: x[1]) #poblacion es una 
    #lista de tuplas y con key= lambda x: x[1] buscamos
    #el maximo de los valores de las tuplas ( secuencia tratamiento , su valor de aptitud)
    graficar.append(gen)
    graficar.append(mejores_valores)
    return mejor_individuo,graficar

# Datos simulados
with open('resultado.pkl','rb') as archivo:
    data = pickle.load(archivo)

esquemas = list(data.keys())
instanteInicial = datetime.now()
# Ejecutar el algoritmo genético
mejor_individuo,graficar = algoritmo_genetico(data, esquemas, generaciones=2000,tamaño_poblacion=1400)

plt.plot(graficar[0],graficar[1])
plt.xlabel('Numero de generaciones')
plt.ylabel('Valor de aptitud mejor individuo cada generación')
#plt.savefig('gráfica_valores_de_aptitud.png')
plt.show()
plt.close()

# al final de la partida
instanteFinal = datetime.now()
tiempo = instanteFinal - instanteInicial # Devuelve un objeto timedelta
segundos = tiempo.seconds
print(mejor_individuo)
print('numero de generaciones', 1000 )
print(segundos)