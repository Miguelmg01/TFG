# -*- coding: utf-8 -*-
'''
Classes encapsulating the osteoporosis model.

Author: David J. Joerg
'''

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from collections import namedtuple
from data import ParameterSet
import data

class BoneTurnoverModel(object):
    def __init__(
        self,
        params,
        x0 = None,
        t0 = 0.,
        aux = {}
    ):
        '''
        Simulation class for the bone turnover model.
    
        Attributes:
            params (dict): Model parameters.
            x0 (list or ndarray, optional): Model initial conditions.
            t0 (float): Initial time. Default to 0.
            aux (dics, optional): Auxiliary functions. If not specified, a default set of
                functions is used.
        '''
        # set up dictionary of auxiliary functions
        self._default_aux = {
            'estrogen': lambda t: 1.,
            'romosozumab': lambda t: 0.,
            'blosozumab': lambda t: 0.,
            'denosumab': lambda t: 0.,
            'alendronate': lambda t: 0.,
            'teriparatide': lambda t: 0.,
            'sclerostin_source': lambda t: 1.
        }
        self._aux = self._default_aux.copy()
        for key, val in aux.items():
            self._aux[key] = val
        
        # list of all dynamic variables        
        self._vars = [
            'pre-osteoblasts',
            'osteoblasts',
            'osteocytes',
            'pre-osteoclasts',
            'osteoclasts',
            'sclerostin',
            'sclerostin-ab',
            'bone density',
            'bone mineral fraction'
        ]
        self._n_vars = len(self._vars)
        self._bd_idx = self._vars.index('bone density')
        
        # list of all model parameters and TeX names
        self._params_dict = {
            'n':        r'n',
            'omega_pB': r'\omega_{\mathrm{B}^*}',
            's_pB':     r's_{\mathrm{B}^*}',
            'r_pB':     r'r_{\mathrm{B}^*}',
            'eta_B':    r'\eta_\mathrm{B}',
            'omega_B':  r'\omega_\mathrm{B}',
            'eta_Y':    r'\eta_\mathrm{Y}',
            'e_pC':     r'e_{\mathrm{C}^*}',
            's_pC':     r's_{\mathrm{C}^*}',
            'omega_pC': r'\omega_{\mathrm{C}^*}',
            'eta_C':    r'\eta_\mathrm{C}',
            'nu_C':     r'\nu_\mathrm{C}',
            'e_C':      r'e_\mathrm{C}',
            'r_C':      r'r_\mathrm{C}',
            'e_s':      r'e_\mathrm{s}',
            'kappa_s':  r'\kappa_\mathrm{s}',
            'delta_s':  r'\delta_\mathrm{s}',
            'lambda_B': r'\lambda_\mathrm{B}',
            's_Omega':  r's_\Omega',
            'r_Omega':  r'r_\Omega',
            'nu_Omega': r'\nu_\Omega',
            'lambda_C': r'\lambda_\mathrm{C}',
            'gamma':    r'\gamma',
            'c_0':      r'c_0',
            
            'beta_pC_rAb':  r'\beta_{\mathrm{C}^*}^\mathrm{rAb}',
            'beta_b_rAb':   r'\beta_{\mathrm{b}}^\mathrm{rAb}',
            'eta_C_bp':     r'\eta_{\mathrm{C}^*}^\mathrm{bp}',
            'beta_pC_pth':  r'\beta_{\mathrm{C}^*}^\mathrm{pth}',
            'beta_B_pth':   r'\beta_{\mathrm{B}}^\mathrm{pth}',
        }

        # set parameters and initial conditions
        self.params = ParameterSet(params)
            
        self.t = t0
        self.x = x0 if (x0 is not None) else np.full(self._n_vars, 0.)
       
        # constants
        self._reg_nrm = 1. / (1. - np.tanh(-1.))
        self._reg_off = np.tanh(-1.)
        
    @property
    def variables(self):
        'Returns the list of model variables.'
        return self._vars
    
    @property
    def params_dict(self, tex_dict=False):
        'Returns the list of model parameters.'
        return self._params_dict
    
    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, value):
        self._params = value

    def update_params(self, params):
        self._params.update(params)
        
    @property
    def x(self):
        return self._x
    
    @x.setter
    def x(self, value):
        if len(value) != self._n_vars:
            raise ValueError('The length of the model state must correspond to the '
                'number of model variables ({}).'.format(self._n_vars))
        self._x = value
        
    @property
    def t(self):
        return self._t
    
    @t.setter
    def t(self, value):
        self._t = value

    @staticmethod
    def piecewise_max_step(t0, t1, ms_outside=np.inf, ms_inside=30, padding=0.1):
        '''
        Convenience function that generates an array of time interval-dependent
        max. integration steps with different values outside and inside a time window
        (e.g. a treatment time window). The result can be directly passed to the
        `max_step` keyword argument of any class-related function.
            
        Args:
            t0, t1 (float): Start and end time of the time window.
            ms_outside (float, optional): Max. integration step (in days) outside the
                time window. Defaults to infinity.
            ms_inside (float, optional): Max. integration step (in days) inside the
                time window. Defaults to 30.
            padding (float): Time padding around the time window in multiples of the
                time window length. Defaults to 0.1.
                
        Returns:
            max_step (list): List of tuples containing the start times of the time
                regions and the respective max. integration step.
        '''
        w = padding * (t1 - t0)
        return [(t0 - w, ms_outside), (t1 + w, ms_inside), (np.inf, ms_outside)]
    
    def _reg_down(self, x):
        'Inhibiting Hill function.'
        # take absolute value to prevent complex values if `n` is not integer
        # and numerical error produce small negative values for `x`
        return 1. / (1. + np.power(np.abs(x), self._params['n']))
        
    def _reg_up(self, x):
        'Activating Hill function.'
        # take absolute value to prevent complex values if `n` is not integer
        # and numerical error produce small negative values for `x`
        return np.power(np.abs(x), self._params['n']) / \
            (1. + np.power(np.abs(x), self._params['n']))
        
    def _turnover_rates(self, t, x):
        p = self._params
        rho_pB, rho_B, rho_Y, rho_pC, rho_C, s, s_bound, rho_bone, c_bone = x
        
        b_formation = p['lambda_B'] * self._reg_down(s / p['s_Omega']) \
             * (1. + p['nu_Omega'] * self._reg_up(rho_C / p['r_Omega'])) * rho_B
        b_resorption = p['lambda_C'] * rho_C
        return b_formation, b_resorption
        
    def _resorption_rate(self, t, x):
        p = self._params
        rho_pB, rho_B, rho_Y, rho_pC, rho_C, s, s_bound, rho_bone, c_bone = x
        r = p['lambda_C'] * rho_C
        return r

    def _derivative(self, t, x):
        '''
        Returns the time derivatives of the underlying ODE system for a
        given model state.
        
        Args:
            t (float): Time point.
            x (ndarray): Array of dynamic variables at time point `t`.
        
        Returns:
            D (ndarray): Array containing the time derivatives of the dynamic variables
                (see parameter `x`).
        '''
        # shorthand notation
        p = self._params
        down = self._reg_down
        up = self._reg_up
        
        # extract variables from array
        rho_pB, rho_B, rho_Y, rho_pC, rho_C, s, s_bound, rho_bone, c_bone = x
            
        # auxiliary variables
        e = self._aux['estrogen'](t)
        exo_pth = self._aux['teriparatide'](t)
        mab_sclerostin = self._aux['romosozumab'](t) + self._aux['blosozumab'](t)
        mab_rankl = self._aux['denosumab'](t)
        bisph = self._aux['alendronate'](t)

        # differentiation fluxes
        flux_pB_to_B = p['omega_pB'] * down(s / p['s_pB']) * rho_pB

        flux_B_to_Y = p['omega_B'] * rho_B
        flux_pC_to_C = p['omega_pC'] * down(e / p['e_pC']) * up(s / p['s_pC']) \
            * (1. - p['beta_pC_rAb'] * up(mab_rankl)) \
            * (1. + p['beta_pC_pth'] * up(exo_pth)) * rho_pC
        
        # loss rates
        loss_B = - p['eta_B'] * (1. - p['beta_B_pth'] * up(exo_pth)) * rho_B - flux_B_to_Y
        loss_C = - p['eta_C'] * (1. + p['nu_C'] * up(e / p['e_C']) \
            * up(rho_C / p['r_C'])) * rho_C - p['eta_C_bp'] * up(bisph) * rho_C
                    
        bp, bm = self._turnover_rates(t, x)
            
        # derivative
        D = [    
            # pre-osteoblasts
            1. - flux_pB_to_B,
            # osteoblasts
            flux_pB_to_B + loss_B,
            # osteocytes
            flux_B_to_Y - p['eta_Y'] * rho_Y,
            # pre-osteoclasts
            1. - flux_pC_to_C,
            # osteoclasts
            flux_pC_to_C + loss_C,
            
            # sclerostin
            down(e / p['e_s']) * rho_Y \
                - p['kappa_s'] * s - p['kappa_s'] * mab_sclerostin * s \
                + p['delta_s'] * s_bound,
            # antibody-bound sclerostin
            p['kappa_s'] * mab_sclerostin * s - p['delta_s'] * s_bound \
                - p['kappa_s'] * s_bound,
                
            # bone density
            bp - bm,
            # bone mineral fraction
            p['gamma'] * (p['c_0'] + p['beta_b_rAb'] * up(mab_rankl) - c_bone)
        ]
        return np.array(D)
    
    def derivatives(self, t, x):
        '''
        Returns the time derivatives of the underlying ODE system for a
        given model state.
        
        Args:
            t (float): Time point.
            x (ndarray): Array of dynamic variables at time point `t`.
        
        Returns:
            D (ndarray): Array containing the time derivatives of the dynamic variables
                (see parameter `x`).
        '''
        return self._derivative(t, x)
        
    def propagate(self, t_prop, dt, solver='BDF', max_step=np.inf):
        '''
        Propagates the model state in time.
            
        Args:
            t_prop (float): Total time for which the model is simulated.
            dt (float): Output time step.
            solver (str, optional): Solver type (as used by `scipy.integrate.solve_ivp`).
                Defaults to 'BDF'.
            max_step (float, optional): Maximum step size for the solver.
            
        Returns:
            result (namedtuple): Time series of model variables.
                (result.t: time, result.x: time series for each model variable,
                result.r: time series for bone formation and resorption rates,
                result.vars: variable names).
        '''

        # initial conditions
        t0 = self._t
        x0 = self._x
        # final time
        t1 = t0 + t_prop
        
        # obtain sections for piecewise solution
        if isinstance(max_step, list) or isinstance(max_step, np.ndarray):
            sections = list(max_step)
        else:
            sections = [(t1, max_step)]
        sections[-1] = (t1, sections[-1][1])  # overwrite last section with total end time
        
        # solve piecewise
        t_res = np.empty(0)
        x_res = np.empty((0, self._n_vars))
        r_res = np.empty((0, 2))
        for i, (tf, m) in enumerate(sections):
            # solve the system of coupled ODEs
            range_t = np.arange(t0, tf + dt, dt)
            
            res = solve_ivp(self._derivative, (np.min(range_t), np.max(range_t)),
                x0, t_eval=range_t, method=solver, max_step=m)
            r = [self._turnover_rates(t, x) for t, x in zip(res.t, np.transpose(res.y))]
                
            # store results
            t_res = np.concatenate([t_res, res.t])
            x_res = np.concatenate([x_res, np.transpose(res.y)])
            r_res = np.concatenate([r_res, r])
            
            # set new initial conditions
            t0 = res.t[-1]
            x0 = res.y[:,-1]
            
        x_res = np.transpose(x_res)
        r_res = np.transpose(r_res)
        
        # set internal state
        self._t = t_res[-1]
        self._x = x_res[:,-1]
        
        res_tuple = namedtuple('result', ['t', 'x', 'r', 'vars'])
        return res_tuple(t_res, x_res, r_res, self.variables)


class OsteoporosisModel(BoneTurnoverModel):
    def __init__(
        self,
        params = {},
        admins = {},
        aux = {},
        init_state = None
    ):
        '''
        Osteoporosis model class managing simulations, medication information and
        explicit age dependencies of the model.
        
        Args:
            params (dict): Model parameters.
            admins (dict, optional): Treatment-related dosing information (see Notes).
            init_state (str or array-like, optional): Initial conditions for the dynamic
                variables of the model. Can either be an array that provides initial
                values for all dynamic variables (in the order specified by 
                `OsteoporosisModel.get_variables()`) or 'equilibrium' to start with the
                steady state for the bone mineral metabolism and unity for the bone
                density.
            
        Examples:
            A series of administrations has to be provided as a list of tuples
            indicating the administration time and the dose, e.g.,
            
            >>> admins['romosozumab'] = [(30, 1), (60, 1), (90, 1.5), (120, 1)]
            >>> admins['alendronate'] = [(180, 1), (210, 1)]
        '''
        self._meds = [
            'romosozumab',
            'blosozumab',
            'denosumab',
            'alendronate',
            'teriparatide'
        ]
        
        # TeX names of parameters
        osteoporosis_params_dict = {
            'init_age':          r'a_0',
            'a_e':               r'a_\mathrm{e}',
            'tau_e':             r'\tau_\mathrm{e}',
            'q_P1NP':            r'q_\mathrm{P1NP}',
            'q_BSAP':            r'q_\mathrm{BSAP}',
            'q_CTX':             r'q_\mathrm{CTX}',
        }
        osteoporosis_params_dict.update({'T_{}'.format(m): 'T_\mathrm{' + m + '}'
            for m in self._meds})
        osteoporosis_params_dict.update({'E_{}'.format(m): 'E_\mathrm{' + m + '}'
            for m in self._meds})
        
        # use custom specifications where supplied, otherwise default
        all_params = ParameterSet()
        all_params.update(params)
        super(OsteoporosisModel, self).__init__(params=all_params, aux=aux)
        
        self._params_dict.update(osteoporosis_params_dict)
                
        # default treatment
        default_admins = {m: [] for m in self._meds}
        available_treatments = default_admins.keys()

        # administrations
        self._admins = default_admins
        for key, val in admins.items():
            self._admins[key] = val

        # create pharmacokinetic time series from dosing information
        self._scaled_admins = {}
        self._discrete_admins = {}
        for key in available_treatments:
            # is there an administration?
            if len(self._admins[key]) == 0:
                self._scaled_admins[key] = []
                self._discrete_admins[key] = True # fallback
            else:
                # scale doses by efficacy
                self._scaled_admins[key] = self._admins[key].copy()
                self._scaled_admins[key][:,-1] *= self._params['E_{}'.format(key)]
                
                self._discrete_admins[key] = (len(self._scaled_admins[key][0]) == 2)

        self._treatment = {}
        self._treatment['romosozumab'] = \
            self._injections(self._scaled_admins['romosozumab'],
                np.log(2.) / self._params['T_{}'.format('romosozumab')])
        self._treatment['blosozumab'] = \
            self._injections(self._scaled_admins['blosozumab'],
                np.log(2.) / self._params['T_{}'.format('blosozumab')])
        self._treatment['denosumab'] = \
            self._injections(self._scaled_admins['denosumab'],
                np.log(2.) / self._params['T_{}'.format('denosumab')])

        if self._discrete_admins['teriparatide'] == True:
            self._treatment['teriparatide'] = \
                self._injections(self._scaled_admins['teriparatide'],
                    np.log(2.) / self._params['T_{}'.format('teriparatide')])
        else:
            self._treatment['teriparatide'] = \
                self._injections_cont(self._scaled_admins['teriparatide'],
                    np.log(2.) / self._params['T_{}'.format('teriparatide')])
        if self._discrete_admins['alendronate'] == True:
            self._treatment['alendronate'] = \
                self._injections(self._scaled_admins['alendronate'],
                    np.log(2.) / self._params['T_{}'.format('alendronate')])
        else:
            self._treatment['alendronate'] = \
                self._injections_cont(self._scaled_admins['alendronate'],
                    np.log(2.) / self._params['T_{}'.format('alendronate')])

        # set used auxiliary functions
        self._aux.update(aux)
        self._aux.update(self._treatment)
        
        if self._params['estrogen_decline']:
            self._aux['estrogen'] = lambda t: self._estrogen_decline(t)
            
        # initialise current state
        self._t = self._params['init_age'] * 365.
        if init_state is None:
            self._y = np.full(self._n_vars, 1.)
            # set antibody-bound Sclerostin to zero
            self._y[self._vars.index('sclerostin-ab')] = 0.
        else:
            if isinstance(init_state, list) or isinstance(init_state, np.ndarray):
                # if `init_state` is an array, set initial conditions to array
                assert len(init_state) == self._n_vars
                self._y = np.array(init_state)
            elif init_state == 'equilibrium':
                # initial state is the equilibrium and unit for the bmd
                self._y = np.full(self._n_vars, 1.)
                self._y[self._vars.index('sclerostin-ab')] = 0.
                self.equilibrate()
              
        return
    
    @property
    def supported_meds(self):
        return self._meds
    
    
    def _estrogen_decline(self, a):
        'Describes the senescence-related time evolution of average estrogen levels.'
        age = a / 365.
        if age <= self._params['a_e']:
            e = 1.
        else:
            e = 1. / (1. + (age - self._params['a_e']) / self._params['tau_e'])
        return e

    def _injections(self, a, kappa):
        '''
        Returns a function that represents exponentially decaying pulses with a
        characteristic decay time.
        
        Args:
            a (ndarray): 2D Array of administrations with administration times in the
                first column and doses in the second column.
            kappa (float): Exponential decay rate.
            
        Returns:
            c (function): Real-valued function representing the systemic concentration.
        '''
        def c(t):
            x = 0
            for t_i, h_i in a:
                if t >= t_i:
                    x += h_i * np.exp(-kappa * (t - t_i))
            return x
        return c
    
    def _injections_cont(self, a, kappa, n=10., dt_int=14.):
        '''
        Returns a function that represents an approximation of the solution of the
        pharmacokinetic profile obtained from piecewise constant administrations at
        a given rate per day.
        
        Args:
            a (ndarray): 2D Array of administrations with the start and end times of 
                administration times in the first and second column and administration
                rate per day in the third column.
            kappa (float): Exponential decay rate.
            n (float, optional): Multiple of the decay time :math:`\kappa^{-1}` used to
                determine the maximum time range after the last administration on which
                the PK dynamics is solved. After this time window, concentrations are
                set to zero. Defaults to 10.
            dt_int (float, optional): Time interval for linear interpolation of the
                pharmacokinetic profile. Smaller values will lead to considerably longer
                computation times when evaluating the function. Defaults to 14.
            
        Returns:
            c (function): Real-valued function representing the concentration.
        '''
        # derivative function
        def ode(t, x):
            source = 0
            for t0, t1, h in a:
                if t0 <= t < t1:
                    source += h
            return [source - kappa * x]
        
        # determine time range for numerical integration
        t_min = np.min(a[:,0]) - dt_int
        t_max = np.ceil(np.max(a[:,1]) + n / kappa)
        range_t = np.arange(t_min, t_max, dt_int)
        
        # solve and create a piecewise function defined for all t > 0
        res = solve_ivp(ode, (t_min, t_max), [0.], t_eval=range_t, max_step=0.5)
        c = lambda t: interp1d(res.t, res.y[0])(t) \
            if t_min < t < (t_max - dt_int) else 0.
        return c
    
    def equilibrate(self, t_int=100*365, tol=1e-8):
        '''
        Computes the equilibrium of the model and sets the model to the equilibrated
        state while retaining the current time and the bone mineral density.
        
        Args:
            t_int (float, optional): Length of successive time intervals after which the
                criteria for equilibration are checked. Defaults to 10 * 365.
            tol (float, optional): Threshold below which the average of the derivatives
                of all variables has to drop to consider the system as equilibrated.
                Defaults to 1e-12.
        '''
        # use current state of the model as starting point
        t0 = self._t
        x0 = self._x.copy()
        dt = int(t_int * 0.1)
        
        # temporarily use time-independent neutral auxiliary functions
        freeze_t = self._t
        freeze_aux = self._aux
        freeze_bd = self._y[self._bd_idx]
        self._aux = self._default_aux
        
        # try to minimise derivative directly first
        f = lambda x: np.max(self._derivative(0, x))
        stat = minimize(f, x0, method='Nelder-Mead', tol=tol, options={'maxiter': 10000})
        x0 = stat.x

        # obtain indices of all variables except the bmd
        test_indices = [i for i in range(self._n_vars) if i != self._bd_idx]

        equilibrated = False        
        while not equilibrated:
            # use method of parent class to retain array structure
            res = super(OsteoporosisModel, self).propagate(t_int, dt)
            
            # equilibration criterion
            t0, x0 = res.t[-1], res.x[:,-1]
            drv = self.derivatives(t0, x0)
            distance = np.mean(drv[test_indices])
            equilibrated = (abs(distance) < tol)
        
        # restore original time and auxiliary functions
        self._t = freeze_t
        self._aux = freeze_aux
        self._x[self._bd_idx] = freeze_bd
        
    @staticmethod
    def _safe_power(vals, p):
        return np.sign(vals) * np.power(np.abs(vals), p)
    
    def propagate(self, t_prop, dt=1., solver='BDF', max_step=np.inf):
        '''
        Forward-simulates the model and returns the results of the
        propagated time period.
        
        Args:
            t_forward (float): Simulation time by which the model is propagated.
            dt (float, optional): Output time step. (Also determines how
                fine-grained the final state of the model can be determined.) 
                Defaults to 1.
            solver (str, optional): Solver type (as used by `scipy.integrate.solve_ivp`).
                Defaults to 'BDF'.
            max_step (float, optional): Maximum step size for the solver.
            
        Returns:
            result (namedtuple): Time series of model variables.
                (result.t: time, result.x: dictionary of time series for each model
                variable as well as derived quantities)
        '''
        model_res = super(OsteoporosisModel, self).propagate(t_prop, dt, solver,
            max_step)
        
        # store dynamic variables including bmd
        x_dict = {key: model_res.x[i] for i, key in enumerate(self._vars)}
        x_dict['bmd'] = x_dict['bone density'] * x_dict['bone mineral fraction']
        
        r = model_res.r
        # store formation and resorption rates
        x_dict['bone formation rate'] = r[0]
        x_dict['bone resorption rate'] = r[1]
        
        # to avoid that spurious small negative rates produce imaginary
        # numbers when raised to a non-integer power, take absolute values
        # and reinstate sign afterwards
        x_dict['p1np'] = self._safe_power(r[0], self._params['q_P1NP'])
        x_dict['bsap'] = self._safe_power(r[0], self._params['q_BSAP'])
        x_dict['ctx'] =  self._safe_power(r[1], self._params['q_CTX'])
            
        # add auxiliary variables
        x_dict.update({k: np.array([ v(s) for s in model_res.t ])
            for k, v in self._aux.items()})

        # collect
        res_tuple = namedtuple('result', ['t', 'x'])
        return res_tuple(model_res.t, x_dict)