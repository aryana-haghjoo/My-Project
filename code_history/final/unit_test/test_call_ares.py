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
    
    """
    value, key = dict_to_list(params)
    value_denormalized = np.array(value, dtype='float64')
    value_denormalized[0] = 10** (value[0])
    value_denormalized[1] = 10** (value[1])
    value_denormalized[2] = 10** (value[2])
    params_denormalized = list_to_dict(value_denormalized, key)
    """
    
    #running ares
    sim1 = ares.simulations.Global21cm(verbose=False, progress_bar=False)    
    sim1.run()  
    z1 = sim1.history['z'][::-1]
    dTb1 = sim1.history['dTb'][::-1]
    
    #z1 = z1[z1<50]
    #dTb1 = dTb1 [:len(z1)]
    #spline1 = CubicSpline(z1, dTb1)
    
    #return spline1(redshifts)
    return z1

params = {'pop_rad_yield_0_': 2}

z = np.linspace(5, 40, 100)
z_a = call_ares(params, z)
plt.plot(z_a)
plt.ylabel('z')
plt.savefig('test.png')
