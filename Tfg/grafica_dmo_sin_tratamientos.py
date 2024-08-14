# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:21:16 2024

@author: Usuario
"""

import sys, os
from itertools import permutations
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

sys.path.insert(0, os.pardir)
from model.model import OsteoporosisModel, ParameterSet
from model.tools import param_file, results_path, nft, plot_settings, colours, format_label

params = ParameterSet(param_file)

O = OsteoporosisModel(params._params,init_state='equilibrium')

t_all = {}; y_all = {}

t_min, t_max = 66 * 365, 72 * 365
t_ref = 67 * 365
t_sim = 365. * 85
max_step = OsteoporosisModel.piecewise_max_step(t_min, t_max)
t_all['tratamiento'], y_all['tratamiento'] = O.propagate(t_sim, dt=1, max_step=max_step)


# Visualizar los resultados
fig1 = plt.figure()
ax = fig1.subplots()
ax.plot(y_all['tratamiento']['bmd'])
ax.set_xlabel('Evolución en dias')
ax.set_ylabel('Nivel de DMO')
fig1.savefig('grafica_dmo_sin_medicamentos.png')
plt.show()
plt.close()


