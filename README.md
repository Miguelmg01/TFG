## Optimización de terapias contra la osteoporosis 

## Descripción del TFG

Este trabajo tiene como objetivo optimizar los tratamientos contra la osteoporosis. Con ese objetivo, a partir de datos simulados de secuencias de tratamientos,
se busca maximizar los niveles de DMO (densidad mineral ósea) para medir la salud ósea con los tratamientos. 
En este repositorio se encuentra el código del modelo junto a los datos para la simulación de las secuencias de tratamiento así como los algoritmos 
genéticos utilizados para la obtención de sus resultados.

## Contenido del repositorio
- En 'main/tfg/' se encuentran el código de los algoritmos genéticos, variando el número de generaciones y tamaño de población. También, se encuentran los
  códigos con los que se obtienen las distintas gráficas del trabajo.
- En 'main/' los archivos '__init__.py', 'data.py' y 'model.py' son los archivos del artículo de Jörg D.J. *et al*, donde en 'data.py' se guardan los
  conjuntos de datos disponibles y en 'model.py' se realiza la simulación del modelo matemático de la osteoporosis.
- En 'main/data/' se encuentran todos los conjuntos de datos clínicos que se gan utilizado para la simulación de tratamientos.
- En 'main/parameters/' se encuentra el archivo 'fit.csv' donde están guardados los valores constantes del modelo.
- En 'main/run_scripts/' se ubican los archivos de validación del modelo realizado por el artículo de referencia. En esta ubicación,
  en 'main/run_scripts/results/' se hallan las gráficas resultantes de esta validación.
- En 'main/conda_env/' se encuentran dos archivos de explicación que paquetes son compatibles para la ejecución de la simulación del modelo.
