import numpy as np
import pandas as pd
import ares
from scipy.interpolate import CubicSpline

#loading the data
params= pd.read_csv('one_sigma_params_mock.gz', sep = " ", header= None)
params = params.values

#loading the EDGES data (the e subscipt is related to EDGES)
data_1 = pd.read_csv('/home/o/oscarh/aryanah/My-Project/data/data_1.csv')
freq_e = data_1.iloc[:,0] #frequency, MHz

#Changing the axis from frequency to redshift
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

#functions
def dict_to_list(d): #converts dictionary to two lists (key and value)
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
    sim = ares.simulations.Global21cm(**params, verbose=False, progress_bar=False)
    sim.run()
    z = sim.history['z'][::-1]
    dTb = sim.history['dTb'][::-1]
    z = z[z<50]
    dTb = dTb[:len(z)]
    spline = CubicSpline(z, dTb)
    
    return spline(redshifts) 

dict_edges = {'pop_rad_yield_0_': 4.54933009e+03, 'pop_rad_yield_2_': 2.47592394e+03, 'fesc': 3.70100011e-01, 'fX': 1.36397790e-01}
value, key = dict_to_list(dict_edges)
curves = np.empty(np.shape(params)[0], len(z_e))
for i in range(np.shape(params)[0]):
    curves[i, :] = call_ares(list_to_dict(params[i, :], key), z_e)

np.savetxt('/scratch/o/oscarh/aryanah/curves_one_sigma_mock.gz' , curves)

