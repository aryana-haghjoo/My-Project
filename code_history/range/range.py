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
    
    value_denormalized = np.array(value, dtype='float64')
    value_denormalized[0] = 10** (value[0])
    value_denormalized[1] = 10** (value[1])
    value_denormalized[2] = 10** (value[2])
    
    key = ['pop_rad_yield_0_', 'pop_rad_yield_1_', 'pop_rad_yield_2_', 'clumping_factor']    
    params_denormalized = list_to_dict(value_denormalized, key)

    #running ares
    sim = ares.simulations.Global21cm(**params_denormalized, verbose=False, progress_bar=False)
    sim.run()
    z = sim.history['z'][::-1]
    dTb = sim.history['dTb'][::-1]
    z = z[z<50]
    dTb = dTb[:len(z)]
    spline = CubicSpline(z, dTb)
    
    return spline(redshifts) 

#----------------------------------------------------------------------------------------------------------------------------------
z = np.linspace(5, 40, 100)

#pop_rad_yield_0_: 0 - 1E20 but it can go upper
#pop_rade_yield_1_: 0 - 1E100 
#pop_rade_yield_2_: 0 - 1E8 but it can go upper
#clumping_factor: 0-76

#pop_rad_yield_0_: 1E2 - 1E10 
#pop_rade_yield_1_: 0 - 1E41
#pop_rade_yield_2_: 0 - 1E6 
#clumping_factor: 0-15

rad_0 = np.linspace(2, 10, 10)
rad_1 = np.linspace(0, 41, 10)
rad_2 = np.linspace(0, 6, 10)
clf = np.linspace(0, 15, 10)
#-----------------------------------------------------------------------------------------------------------------------------------

accepted = []
for i in range (0, len(rad_0)):
    for j in range(0, len(rad_1)):
        for k in range(0, len(rad_2)):
            for l in range(0, len(clf)):
                try:
                    params = [rad_0[i], rad_1[j], rad_2[k], clf[l]]
                    T = call_ares(params, z)
                except:
                    pass
                else:
                    accepted.append(params)
#np.savetxt('accepted.gz', accepted)                    
np.savetxt('/scratch/o/oscarh/aryanah/range/accepted.gz', accepted)
