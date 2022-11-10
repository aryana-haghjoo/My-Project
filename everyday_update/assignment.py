import pandas as pd
import ares
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
N = 20
z = np.linspace(5,35,100)
data = pd.read_csv('mcmc_params', sep=" ", header=None)
#print(data)
data = data.drop(0,1)
#print(data)

def call_ares (ares_params, redshifts):
        '''
        Returns the temperature fluctuation vs redshift curve.
        INPUTS:
        ares_params: Specify which the values of the desired paramamters.
        redshift: Specify the redshift over which to graph the curve.
        OUTPUTS:
        A cubic spline of the temperature fluctuation vs redshift curve produced by ares.
        '''    
        
        sim = ares.simulations.Global21cm(**ares_params, verbose=False, progress_bar=False)
        sim.run()
        z = sim.history['z'][::-1]
        dTb = sim.history['dTb'][::-1]
        sorted_idx = np.argsort(z,kind="stable")
        z = z[sorted_idx]
        dTb = dTb[sorted_idx]
        spline = CubicSpline(z, dTb)    
        return spline(redshifts)

def list_to_dict(value,key):
    return dict(zip(key,value))

all_curves = np.zeros((N,len(z)))
features = np.zeros((N,3))
for i in range(N):
    print(i)
    paramters = ["fstar","fX","fesc","clumping_factor","cX"]
    values = data.iloc[i*1000,:]
    #print(values)
    dict_combined = list_to_dict(values,paramters)
    #print(dict_combined)
    temp = dict_combined["cX"]
    dict_combined["cX"] = temp*(1E39)
    curve = call_ares(dict_combined, z)
    min_index = np.argmin(curve)
    min_z = z[min_index]
    min_t = curve[min_index]
    depth = 0 - min_t
    half = -depth/2.0
    j = min_index
    k = min_index
    while (half > curve[j]):
        j= j-1
    while (half > curve[k]):
        k = k+1
    z_left = z[j-1]
    z_right = z[k+1]
    width = z_right-z_left

    all_curves[i,:] = curve
    features[i,0] = min_z
    features[i,1] = depth
    features[i,2] = width

a = np.zeros((N,N))    
#print(features)
for i in range(N):
    for j in range(N):
        if j < i:
            a[i, j] = np.sqrt((features[i,0]-features[j,0])**2+(features[i,1]-features[j,1])**2+(features[i,2]-features[j,2])**2)
            
#print(a)
np.savetxt('score.gz' , a)

plt.hist(a, bins= 10)
plt.savefig('histogram.png')
plt.xlabel('score')
plt.ylabel('population')

                                                                            
