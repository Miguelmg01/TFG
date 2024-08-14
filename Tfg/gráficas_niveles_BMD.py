# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:36:25 2024

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



def main():
    # simulation time
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
    #params.update__params('nivel_ejercicio',OsteoporosisModel.nivel_ejercicio())
    
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

    # compute BMD changes
    tab = []
    bmd = []
    for label, data in treatments.items():
        t, y = t_all[label], y_all[label]
        # normalise BMD to reference time point , t_ref=67 years
        bmd_change = nft(t, y['bmd'], t_ref) - 1.
        bmd.append(bmd_change)
        # max. BMD change        
        max_bmd_change = np.max(bmd_change[t >= t_min])
        # BMD change 10 years after treatment end
        bmd_change_10yrs = bmd_change[t >= t_max + 10 * 365][0]
        # convert to percent and store
        tab.append([max_bmd_change, bmd_change_10yrs])
    
    
    
    # Visualizar los resultados
    fig1 = plt.figure()
    ax = fig1.subplots()
    for label in treatments.keys():
        ax.plot(y_all[label]['bmd'])
    ax.set_xlabel('Evolución en dias')
    ax.set_ylabel('Nivel de DMO')
    fig1.savefig('grafica_dmo.png')
    plt.show()
    plt.close()
    
    fig2 = plt.figure()
    ax = fig2.subplots()
    for tratamientos in treatments.keys():
        ax.plot(y_all[tratamientos]['bmd'][24090:], label=str(tratamientos))
        ax.legend()
    ax.set_xlabel('Evolución en dias a partir de los 66 años')
    ax.set_ylabel('Nivel de DMO')
    fig2.savefig('grafica_dmo2.png')
    plt.show()
    plt.close()



if __name__ == '__main__':
    main()