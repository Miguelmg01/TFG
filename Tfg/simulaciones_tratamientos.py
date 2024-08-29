# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 17:30:20 2024

@author: Miguel
"""

import sys, os
from itertools import permutations
from copy import deepcopy
import numpy as np
import pickle

sys.path.insert(0, os.pardir)
from model.model import OsteoporosisModel, ParameterSet
from model.tools import param_file

def medseq(a_dur, t_int, dose):
    '''
    Generates an array of constant dose administrations in fixed intervals.
    
    Args:
        a_dur (int): Duration of the medication episode (in years).
        t_int (int): Interval between consecutive administrations (in days).
        dose (float): Drug dose.
        
    Returns:
        Array containing dose administration data listed as (admin. time, dose).
    '''
    return np.array([[t, float(dose)] for t in np.arange(0, a_dur * 365, t_int)])


t_min, t_max = 66 * 365, 72 * 365
t_ref = 67 * 365
t_sim = 365. * 85

# drug-individual administration schemes
test_admins = {
    'alendronate': medseq(a_dur=1, t_int=7, dose=70),
    'romosozumab': medseq(a_dur=1, t_int=30, dose=140),
    'denosumab': medseq(a_dur=1, t_int=180, dose=60)
}    
ages = [67, 68, 69]

# generate permutations
treatments = {}
seq_abbr = {}
for drugs in permutations(test_admins.keys()): 
    admins = deepcopy(test_admins)
    for a, d in zip(ages, drugs):
        admins[d][:,0] += a * 365
    key = ' -> '.join([d[0].upper() for d in drugs])
    treatments[key] = admins
    seq_abbr[key] = ''.join([d[0].upper() for d in drugs])
    
# read parameters
params = ParameterSet(param_file)

# simulate
t_all = {}; y_all = {}
for d, admins in treatments.items():
    print('Simulating {}...'.format(d), end='')
    # equilibrate
    avatar = OsteoporosisModel(params, admins=admins, init_state='equilibrium')
    
    # max solver step 1 day within the treatment region
    max_step = OsteoporosisModel.piecewise_max_step(t_min, t_max)
    t_all[d], y_all[d] = avatar.propagate(t_sim, dt=1, max_step=max_step)
    print('done.')
print('')


with open('resultado.pkl', 'wb') as archivo:
    pickle.dump(y_all, archivo)