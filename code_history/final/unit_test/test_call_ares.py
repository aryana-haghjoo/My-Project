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

#'pop_rad_yield_1': upper limit: 96
#'pop_rad_yield_2': upper limit: 31
#'clumping factor': upper limit: 28
params = {'pop_rad_yield_0_': 4.03, 'pop_rad_yield_1_': 96, 'pop_rad_yield_2_': 5, 'clumping_factor': 0.71} 
z = np.linspace(5, 40, 100)
T = call_ares(params, z)
plt.plot(z, T)
plt.savefig('test.png')
