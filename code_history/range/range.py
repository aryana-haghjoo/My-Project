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

def call_ares (value, redshifts):  
    params = list_to_dict(value, key)

    #running ares
    sim = ares.simulations.Global21cm(**params, verbose=False, progress_bar=False)
    sim.run()
    z = sim.history['z'][::-1]
    dTb = sim.history['dTb'][::-1]
    z = z[z<50]
    dTb = dTb[:len(z)]
    spline = CubicSpline(z, dTb)
    
    return spline(redshifts) 

#----------------------------------------------------------------------------------------------------------------------------------
#pop_rad_yield_0_: 0 - 1E20 but it can go upper
#pop_rade_yield_1_: 0 - 1E100 
#pop_rade_yield_2_: 0 - 1E8 but it can go upper
#clumping_factor: 0-76

#pop_rad_yield_0_: 1E2 - 1E10 
#pop_rade_yield_1_: 0 - 1E41
#pop_rade_yield_2_: 0 - 1E5
#clumping_factor: 0-12

#dict_true = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_2_': 1E4, 'clumping_factor': 1.7, 'fX': 0.2} 
dict_true = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_2_': 1E3, 'fesc': 0.1, 'fX': 0.1}
value, key = dict_to_list(dict_true)
z = np.linspace(5, 40, 100)
n = 2
rad_0 = np.linspace(1E1, 1E20, n)
rad_2 = np.linspace(1E1, 1E6, n)
fesc = np.linspace(0.1, 1, n)
fX = np.linspace(1E-10, 1, n)
#-----------------------------------------------------------------------------------------------------------------------------------

accepted = []
for i in range (0, len(rad_0)):
    for j in range(0, len(rad_2)):
        for k in range(0, len(fesc)):
            for l in range(0, len(fX)):
                try:
                    params = [rad_0[i], rad_2[j], fesc[k], fX[l]]
                    T = call_ares(params, z)
                except:
                    pass
                else:
                    accepted.append(params)
#np.savetxt('accepted.gz', accepted)                    
np.savetxt('/scratch/o/oscarh/aryanah/range/accepted.gz', accepted)