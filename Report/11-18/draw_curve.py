import ares
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

#loading the EDGES data (the e subscipt is related to EDGES)-----------------------------------------------------------------
data_1 = pd.read_csv('data_1.csv')
freq_e = data_1.iloc[:,0] #frequency, MHz

#Changing the axis from frequency to redshift---------------------------------------------------------------------------------
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

def dict_to_list(d): # converts dictionary to two lists (key and value)
    #d must be a dictionary containing the value of parameters and their names
    key = list(d.keys())
    value = list(d.values())
    return value, key

def list_to_dict(value, key): #converts two lists (key and value) to a dictionary
    #value is a list of parameters' values
    #key is a list parameters' names
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
    sorted_idx = np.argsort(z, kind="stable")
    z = z[sorted_idx]
    dTb = dTb[sorted_idx]
    spline = CubicSpline(z, dTb)
    return spline(redshifts) 
low_bound = np.array([-100, 1E30, 1E3, 0.05]) 
up_bound = np.array([100, 1E40, 1E6, 2])

#'pop_rad_yield_1': upper limit: 96
#'pop_rad_yield_2': upper limit: 31
#'clumping factor': upper limit: 28

start_guess = {'pop_rad_yield_0_': 4, 'pop_rad_yield_1_': 97, 'pop_rad_yield_2_': 4, 'clumping_factor': 0.71} 
T = call_ares(start_guess, z_e)

fig1 = plt.figure()
plt.plot(z_e , T, label = 'ARES')
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.savefig('LM_result.png')
