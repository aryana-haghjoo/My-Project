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
    sim2 = ares.simulations.Global21cm(**params, verbose=False, progress_bar=False)
    
    sim1.run()
    sim2.run()
    
    z1 = sim1.history['z'][::-1]
    dTb1 = sim1.history['dTb'][::-1]
    
    z2 = sim2.history['z'][::-1]
    dTb2 = sim2.history['dTb'][::-1]
    
    z1 = z1[z1<50]
    dTb1 = dTb1 [:len(z1)]
    spline1 = CubicSpline(z1, dTb1)
    
    z2 = z2[z2<50]
    dTb2 = dTb2 [:len(z2)]
    spline2 = CubicSpline(z2, dTb2)
    
    return spline1(redshifts), spline2(redshifts)

#clumping_factor: 0-40
#pop_rad_yield_0_: 1E2 - 1E10 
#pop_rade_yield_1_: 0 - 1E41
#pop_rade_yield_2_: 0 - 1E6 

#params = {'pop_rad_yield_0_': , 'pop_rad_yield_1_': , 'pop_rad_yield_2_': , 'clumping_factor': 1} 
#params = {'clumping_factor': 40}
params = {'pop_rad_yield_0_': 2}

z = np.linspace(5, 40, 100)
default, perturb= call_ares(params, z)
plt.plot(z, default, label= "default")
plt.plot(z, perturb, label= "perturbed")
plt.legend()
plt.savefig('test.png')
