# -*- coding: utf-8 -*-
'''
Classes handling data and parameter management.

Author: David J. Joerg
'''

import pandas as pd
import numpy as np
import os
from tabulate import tabulate

Lista = ["continuous_dosing_looker1998_and_black2006_alendronate-alendronate.csv","continuous_dosing_looker1998_and_black2006_alendronate-placebo.csv","continuous_dosing_looker1998_and_leder2014_combination.csv","continuous_dosing_looker1998_and_leder2014_teriparatide.csv","continuous_dosing_looker1998_and_leder2015_combination_to_denosumab.csv","continuous_dosing_looker1998_and_leder2015_denosumab_to_teriparatide.csv","continuous_dosing_looker1998_and_leder2015_teriparatide_to_denosumab.csv","continuous_dosing_looker1998_and_saag2017_alendronate.csv","continuous_dosing_looker1998_and_saag2017_romosozumab_to_alendronate.csv","looker1998_and_black2006_alendronate-alendronate.csv","looker1998_and_black2006_alendronate-placebo.csv","looker1998_and_bone2011_denosumab.csv","looker1998_and_bone2011_placebo.csv","looker1998_and_cosman2016_placebo_to_denosumab.csv","looker1998_and_cosman2016_romosozumab_to_denosumab.csv","looker1998_and_leder2014_combination.csv","looker1998_and_leder2014_denosumab.csv","looker1998_and_leder2014_teriparatide.csv","looker1998_and_leder2015_combination_to_denosumab.csv","looker1998_and_leder2015_denosumab_to_teriparatide.csv","looker1998_and_leder2015_teriparatide_to_denosumab.csv","looker1998_and_lewiecki2018_placebo_to_denosumab.csv","looker1998_and_lewiecki2018_romosozumab_to_denosumab.csv","looker1998_and_mcclung2006_denosumab_6_q3m.csv","looker1998_and_mcclung2006_denosumab_14_q3m.csv","looker1998_and_mcclung2006_denosumab_14_q6m.csv","looker1998_and_mcclung2006_denosumab_30_q3m.csv","looker1998_and_mcclung2006_denosumab_60_q6m.csv","looker1998_and_mcclung2006_denosumab_100_q6m.csv","looker1998_and_mcclung2006_denosumab_210_q6m.csv","looker1998_and_mcclung2017_denosumab_8yrs.csv","looker1998_and_mcclung2017_placebo_4yrs_to_denosumab_4yrs.csv","looker1998_and_mcclung2018_alendronate_to_romosozumab_to_denosumab.csv","looker1998_and_mcclung2018_alendronate_to_romosozumab_to_placebo.csv","looker1998_and_mcclung2018_placebo_to_denosumab.csv","looker1998_and_mcclung2018_placebo_to_placebo.csv","looker1998_and_mcclung2018_romosozumab_to_denosumab.csv","looker1998_and_mcclung2018_romosozumab_to_placebo.csv","looker1998_and_recknor2015_blosozumab_180_q2w.csv","looker1998_and_recknor2015_blosozumab_180_q4w.csv","looker1998_and_recknor2015_blosozumab_270_q2w.csv","looker1998_and_recknor2015_placebo.csv","looker1998_and_saag2017_alendronate.csv","looker1998_and_saag2017_romosozumab_to_alendronate.csv"]

class ParameterSet(object):
    def __init__(self, input_data=None):
        '''
        Reads parameter sets from files and manages them.
        
        Args:
            input_data: If a list of tuples or a dictionary, interpreted as the
                parameter list and values. If string, interpreted as file name of a csv
                file containing the parameter list and values.
        '''
        self._params = {}
        self._info = {}
        
        if input_data is not None:
            if isinstance(input_data, ParameterSet):
                self.params = input_data.params.copy()
                self._info = input_data.info
            elif isinstance(input_data, dict):
                self.params = input_data.copy()
            elif isinstance(input_data, list):
                self.params = dict(input_data)
            elif isinstance(input_data, str):
                # if a string is given, it is interpreted as a filename
                self.read_csv(input_data)
            
    def __repr__(self):
        'Prints parameters as a formatted table.'
        table = [[key, val] for key, val in self._params.items()]
        table = sorted(table, key=lambda x: x[0])
        return tabulate(table, headers=['Parameter', 'Value'])
    
    def __getitem__(self, arg):
        'Returns a single parameter value.'
        return self._params[arg]
    
    def _convert(self, val):
        'Automatic datatype conversion for bools and floats.'
        if isinstance(val, list):
            return map(self._convert, val)
        if isinstance(val, np.ndarray):
            return map(self._convert, val.tolist())
        
        # check if bool and convert if necessary
        if isinstance(val, bool) or isinstance(val,bool):
            return val
        if isinstance(val, str):
            if val.lower() == 'true':
                return True
            elif val.lower() == 'false':
                return False

        # try to convert to float
        try:
            v = float(val)
            return v
        except:
            return val
        
    def read_csv(self, filename):
        '''
        Reads a csv parameter file and stores its contents in an internal dictionary.
        
        Args:
            filename (str): Path of the parameter file.
        '''
        df = pd.read_csv(filename)
        self.params = dict(zip(df['Parameter'].values, map(self._convert, df['Value'].values)))
        
        self._info = {}
        info_cols = list(set(df.columns) - set(['Parameter', 'Value']))
        for col in info_cols:
            self._info[col] = dict(zip(df['Parameter'].values, self._convert(df[col].values)))

    @property
    def names(self):
        return self._params.keys()
    
    @property
    def params(self):
        return self._params
    
    #lo he metido yo
    def update__params(self,nombre,valor):
        self._params[nombre] = valor
    
    @params.setter
    def params(self, value):
        self._params = value
        
    def update(self, value):
        '''
        Updates the internal parameter dictionary.
        
        Args:
            value (dict): Dictionary of new parameter values.
        '''
        if isinstance(value, dict):
            self._params.update(value)
        elif isinstance(value, ParameterSet):
            self._params.update(value.params)
        
    @property
    def info(self):
        return self._info
    
    def get_param_class(self, class_name):
        return [key for key, val in self._info[class_name].iteritems() if val]
    
    @property
    def dataframe(self):
        table = [[key, val] for key, val in self._params.iteritems()]
        table = sorted(table, key=lambda x: x[0])
        return pd.DataFrame(table, columns=['Parameter', 'Value'])
    
    
class ClinicalTrialData(object):
    def __init__(self, filename=None):
        '''
        Reads and manages hybrid aging/treatment datasets.

        Args:
            filename (str): Path to the data file.
        '''
        # supported clinical variables
        self._biomarkers = ['p1np', 'ctx', 'bsap']
        self._observables = {
            'bmd': 'BMD total hip',
            'p1np': 'P1NP',
            'bsap': 'BSAP',
            'ctx': 'CTX'
        }
        # relevant for discrete administration schemes
        self._meds = {
            'Denosumab (mg)':    'denosumab',
            'Blosozumab (mg)':   'blosozumab',
            'Romosozumab (mg)':  'romosozumab',
            'Alendronate (mg)':  'alendronate'
        }
        self._prefix_med = 'Medication: '
        self._col_time = 'Time (days)'
        self._col_doses = ['Dosing start (days)', 'Dosing end (days)', 'Dose per day']
        
        self._occurring_meds = []
        self._data = {}
        
        if filename is not None:
            self.read_csv(filename)
            
    def __repr__(self):
        return str(self._data)
        
    def _read_dosing_data(self, filename):
        df = pd.read_csv(filename)
        admins = {}
        for med, df_med in df.groupby('Type'):
            admins[med] = df_med[self._col_doses].values.astype(float)
        return admins
    
    def _extract_data(self, df, col):
        columns = [
            self._col_time,
            col,
            'Error low: {}'.format(col),
            'Error high: {}'.format(col)
        ]
        data = df[columns].dropna(subset=[col]).values
        return data[:,0:1+1], data[:,2:]
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, value):
        self._data = value

    @property
    def errors(self):
        return self._errors
    
    @errors.setter
    def errors(self, value):
        self._errors = value
        
    @property
    def admins(self):
        return self._admins
    
    @admins.setter
    def admins(self, value):
        self._admins = value

    @property
    def treatment_period(self):
        'Extracts and returns the first and last date on which drugs are administered.'
        t0, t1 = np.inf, -np.inf
        for m in self._occurring_meds:
            a = self._admins[m]
            if a.shape[1] == 2:
                # discrete dosing format
                t, _ = np.transpose(a)
                t0 = min(t0, np.min(t))
                t1 = max(t1, np.max(t))
            elif a.shape[1] == 3:
                # continuous dosing format
                t_start, t_end, _ = np.transpose(a)
                t0 = min(t0, np.min(t_start))
                t1 = max(t1, np.max(t_end))
        return t0, t1
    
    @property
    def observation_period(self):
        'Extracts and returns the first and last date on which data points are present.'
        t0, t1 = np.inf, -np.inf
        for data in self._data.values():
            t, _ = np.transpose(data)
            t0 = min(t0, np.min(t))
            t1 = max(t1, np.max(t))
        return t0, t1
        
    def read_csv(self, filename, overwrite_discrete=True):
        '''
        Reads a csv data file and stores its contents.
        
        Args:
            filename (str): Path of the data file.
            overwrite_discrete (bool): If True, discrete dosing data is replaced by
                continuous dosing data if both are present. Defaults to `True`.
        '''
        # read clinical data
        df = pd.read_csv(filename)
        
        # obtain list of medications occurring in the data file
        cols_med = [c for c in df.columns.tolist() if self._prefix_med in c]
        self._occurring_meds = [self._meds[key[len(self._prefix_med):]] for key in cols_med]
        
        # extract discrete drug administration data
        admins = {}
        for col in cols_med:
            key = self._meds[col[len(self._prefix_med):]]
            data_med = df[[self._col_time, col]].dropna().values.astype(float)
            admins[key] = data_med.copy()
            
        # extract continuous drug administration data from separate file
        dosing_file = os.path.join(
            os.path.dirname(filename),
            f'continuous_dosing_{os.path.basename(filename)}'
        )
        if os.path.isfile(dosing_file):
            # read dosing data
            cont_admins = self._read_dosing_data(dosing_file)
            
            # store continuous dosing (and replace any discrete dosing if desired)
            for key, val in cont_admins.items():
                if (key not in admins.keys()) or overwrite_discrete:
                    admins[key] = val
                    self._occurring_meds.append(key)
                
        # define fit data
        data = {}
        errors = {}
        
        observables = {key: val for key, val in self._observables.items()
            if val in df.columns.tolist()}
        
        for obs, name in observables.items():
            traj, err = self._extract_data(df, name)
            data[obs] = traj
            
            # convert error low/high into lengths of error bars
            errors[obs] = np.array([
                [traj[i,1] - err[i,0], err[i,1] - traj[i,1]]
                for i, _ in enumerate(err)
            ])
        
        self._data = data
        self._errors = errors
        self._admins = admins
        