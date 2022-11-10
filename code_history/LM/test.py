import ares
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numpy.random import normal
from math import ceil
from scipy.interpolate import CubicSpline
import cProfile
import re
#cProfile.run('re.compile("foo|bar")')

data_1 = pd.read_csv('data_1.csv')

freq_e = data_1.iloc[:,0] #frequency, MHz

T_e = data_1.iloc[:,-1] #21cm brighness temperature, k

model_e = data_1.iloc[:, -2] #the model represented in the paper, k

#temporary mfactor
model_e = model_e/2

#Changing the data from frequency to redshift
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

#converting the Data temperatures from k to mK 
k = 1000
model_e = k * model_e

def dict_to_list(d): # converts dictionary to two lists (key and value)
    #d must be a dictionary containing the value of parameters and their names
    key = list(d.keys())
    value = list(d.values())
    return value, key

def list_to_dict(value, key): #converts two lists (key and value) to a dictionary
    #value is a list of parameters' values
    #key is a list parameters' names
    return dict(zip(key, value))

"""
def call_ares(input_dict, redshifts): #5params 

    tmp = ares.util.ParameterBundle('mirocha2017:base')
    tmp.update(input_dict)
    tmp.update({"pop_binaries_0_":True})
    params = tmp
    sim = ares.simulations.Global21cm(**params, verbose = False, progress_bar = False)
    sim.run()
    z = sim.history['z'][::-1]
    sorted_idx = np.argsort(z, kind="stable")
    dTb = sim.history['dTb'][::-1]
    z = z[sorted_idx]
    dTb = dTb[sorted_idx]
    spline_dTb = CubicSpline(z, dTb)
    return spline_dTb(redshifts)
"""
def call_ares (params, redshifts): #4params
    '''
    Returns the temperature fluctuation vs redshift curve.
    INPUTS:
        ares_params: Specify which the values of the desired paramamters.
        redshift: Specify the redshift over which to graph the curve.
    OUTPUTS:
        A cubic spline of the temperature fluctuation vs redshift curve produced by ares.
    '''    
    
    #to denormalize the values
    value, key = dict_to_list(params)
    value_denormalized = np.array(value, dtype='float64')
    value_denormalized[0] = 10** (value[0])
    value_denormalized[1] = 10** (value[1])
    value_denormalized[2] = 10** (value[2])
    params_denormalized = list_to_dict(value_denormalized, key_guess)
    
    #running ares
    sim = ares.simulations.Global21cm(**params_denormalized, verbose=False, progress_bar=False)
    sim.run()
    z = sim.history['z'][::-1]
    dTb = sim.history['dTb'][::-1]
    sorted_idx = np.argsort(z,kind="stable")
    z = z[sorted_idx]
    dTb = dTb[sorted_idx]
    spline = CubicSpline(z, dTb)	
    return spline(redshifts)   

def chisquare (model, data):
    return np.sum((model-data)**2)

start_guess = {'pop_rad_yield_0_': 4, 'pop_rad_yield_1_': 38, 'pop_rad_yield_2_': 4, 'clumping_factor': 1E-5} #4param version, normalized
#["pop_rad_yield_0_", 1E3, 1E6, "log"], ["pop_rad_yield_1_", 1E37, 5E40, "log"], ["pop_rad_yield_2_", 1E3, 5E6, "log"], ["clumping_factor", 0.05, 2, "lin"]
#2-7
#30-45
#2-7
#0.01-4

#converting start guess to two lists
value_guess, key_guess = dict_to_list(start_guess)

T = call_ares (start_guess, z_e)
print('chi-square: ', chisquare(model_e, T))
      
plt.plot(z_e, T, label='ARES')
plt.plot(z_e, model_e, label='EDGES')
plt.legend()
plt.savefig('test.png')

#cProfile.run('re.compile("foo|bar")')