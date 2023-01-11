import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
import ares

def dict_to_list(d): # converts dictionary to two lists (key and value)
    #d must be a dictionary containing the value of parameters and their names
    key = list(d.keys())
    value = list(d.values())
    return value, key

def list_to_dict(value, key): #converts two lists (key and value) to a dictionary
    #value is a list of parameters' values
    #key is a list of parameters' names
    return dict(zip(key, value))

def call_ares (params, redshifts): 
    #params should be a dictionary
    value, key = dict_to_list(params)
    
    value_denormalized = np.array(value, dtype='float64')
    value_denormalized[0] = 10** (value[0])
    value_denormalized[1] = 10** (value[1])
    value_denormalized[2] = 10** (value[2])
    params_denormalized = list_to_dict(value_denormalized, key)
    
    #running ares
    sim = ares.simulations.Global21cm(**params_denormalized, verbose=False, progress_bar=False)
    sim.run()
    z = sim.history['z'][::-1]
    dTb = sim.history['dTb'][::-1]
    z = z[z<50]
    dTb = dTb [:len(z)]
    spline = CubicSpline(z, dTb)
    
    return spline(redshifts) 

def check_limits_lm(m):
#'pop_rad_yield_1': upper limit: 96
#'pop_rad_yield_2': upper limit: 31
#'clumping factor': upper limit: 28

    m_new = m
    if m[1]>96:
        print('out of bound 1')
        m_new[1] = 96
        
    if m[2]>31:
        print('out of bound 2')
        m[2] = 31
        
    if m[3] > 28:
        print('out of bound 3')
        m[3] = 28
        
    return m_new

def func_ares (m, z, d = 4*int(1E5)): 
    #you can further change this function to include the best dx
    #m is the list of params 
    #z is the redshift range
    #y is the brightness temp
    m = np.array(m)
    T = call_ares (list_to_dict(m, key), z)
    derivs = np.zeros([len(z), len(m)])
    dpars = np.zeros(len(m))
    dpars = m/d 
    for i in range(len(m)):
        pars_plus = np.array(m, copy=True, dtype = 'float64')
        pars_plus[i] = pars_plus[i] + dpars[i]
        
        pars_minus = np.array(m, copy=True, dtype = 'float64')
        pars_minus[i] = pars_minus[i] - dpars[i]
        
        try:
            A_plus = call_ares (list_to_dict(pars_plus, key), z)
        except:
            pars_plus = check_limits_lm(pars_plus)
            A_plus = call_ares (list_to_dict(pars_plus, key), z)
            
        try:
            A_minus = call_ares (list_to_dict(pars_minus, key), z)
        except:
            pars_minus = check_limits_lm(pars_minus)
            A_minus = call_ares (list_to_dict(pars_minus, key), z)
            
        A_m = (A_plus - A_minus)/(pars_plus[i]-pars_minus[i])
        derivs[:, i] = A_m    
    return T, derivs

z = np.linspace(5, 40, 100)
params = {'pop_rad_yield_0_': 4.03, 'pop_rad_yield_1_': 97, 'pop_rad_yield_2_': 5, 'clumping_factor': 0.71}
value, key = dict_to_list(params)

T, derivs = func_ares(value, z)
plt.plot(z, derivs[:, 1])
plt.savefig('test.png')
