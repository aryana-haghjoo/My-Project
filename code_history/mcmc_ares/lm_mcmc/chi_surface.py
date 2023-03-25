import numpy as np
import matplotlib.pyplot as plt
import ares
import pandas as pd
from scipy.interpolate import CubicSpline

def draw_samples(N,n):
    m = N.shape[0]
    mat = np.random.randn(m, n)
    L = np.zeros((m, m))
    w, v = np.linalg.eig(N)   
    for i in range(m):
        L [:, i] = np.real(np.sqrt(w [i])) * np.real(v [:, i])
    return np.real(L@mat).T

def list_to_dict(value, key): #converts two lists (key and value) to a dictionary
    #value is a list of parameters' values
    #key is a list parameters' names
    return dict(zip(key, value))

def dict_to_list(d): # converts dictionary to two lists (key and value)
    #d must be a dictionary containing the value of parameters and their names
    key = list(d.keys())
    value = list(d.values())
    return value, key

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
    plt.plot(z)
    plt.savefig('z.png')
    spline = CubicSpline(z, dTb)
    return spline(redshifts) 

def chisquare (pars, data, err): #returns the chi-square of two values - err can be a number/array
    pred = call_ares(list_to_dict(pars, key), z_e)
    chisq = np.sum((pred-data)**2/err**2)
    return chisq

data_1 = pd.read_csv('data_1.csv')
freq_e = data_1.iloc[:,0]
v_0 = 1420 
z_e = (v_0/freq_e)-1 
key = ['pop_rad_yield_0_', 'pop_rad_yield_1_', 'pop_rad_yield_2_', 'clumping_factor']

y = np.array([ -15.01515024,  -16.84145935,  -18.77110764,  -20.22221429,
        -22.45892996,  -24.92668574,  -27.4569482 ,  -30.16009176,
        -33.14570341,  -35.90719096,  -39.35886273,  -42.7084619 ,
        -46.45981957,  -50.02247979,  -53.62148476,  -57.38059519,
        -61.60571445,  -65.40789622,  -69.65601365,  -73.62466391,
        -77.85367294,  -81.76027216,  -85.88585149,  -89.67473374,
        -93.70984126,  -97.21883665, -101.22944581, -104.77555981,
       -108.30722337, -111.4692896 , -114.82529734, -117.93533454,
       -121.07572429, -123.74458313, -126.59479175, -128.90720126,
       -131.53282089, -133.66075546, -135.76372386, -137.69470258,
       -139.51487387, -141.15599036, -142.63762296, -144.00168469,
       -145.16460019, -146.24047035, -147.07843366, -148.08038161,
       -148.60746876, -148.95395788, -149.30748223, -149.59917876,
       -149.8461424 , -149.79451493, -149.54526114, -149.38411675,
       -148.89118404, -148.38293436, -147.68790673, -146.87748306,
       -146.08688412, -145.07351893, -144.07943246, -142.88170717,
       -141.62607414, -140.05985137, -138.53795472, -136.90127412,
       -135.40254926, -133.37441187, -131.66916543, -129.90392171,
       -127.91385762, -125.93476921, -123.72851767, -121.72061588,
       -119.44775108, -117.22481702, -114.68768131, -112.46896673,
       -110.08153712, -107.84688908, -105.28788221, -103.01773433,
       -100.39905528,  -97.95283991,  -95.4634135 ,  -93.01358828,
        -90.69898449,  -88.32766571,  -85.8971963 ,  -83.30970476,
        -81.00722727,  -78.60200979,  -76.32344037,  -74.08740127,
        -71.78482362,  -69.43900924,  -67.49753043,  -65.05117731,
        -63.09251706,  -61.17942106,  -59.232535  ,  -57.13064682,
        -55.35899786,  -53.31597605,  -51.61647355,  -49.89423654,
        -48.1687536 ,  -46.33064753,  -44.97746281,  -43.47618508,
        -42.04148099,  -40.5624824 ,  -39.14418224,  -37.8602809 ,
        -36.51488024,  -35.31971617,  -34.06509307,  -32.97336636,
        -31.94487115,  -30.98943127,  -29.94639699,  -28.88685784,
        -28.18234364,  -27.32240604,  -26.29958885,  -25.55006604])
    
cov = np.array([[ 2.66281232e+03, -3.27340955e+00, -1.08789224e+03,
        -1.46249702e+03],
       [-3.27340955e+00,  1.95932021e-01,  4.02689891e+01,
         1.99594510e+01],
       [-1.08789224e+03,  4.02689891e+01,  9.30015891e+03,
         5.26476716e+03],
       [-1.46249702e+03,  1.99594510e+01,  5.26476716e+03,
         3.97195997e+03]])

samples = pd.read_csv('samples.gz', sep=" ", header=None)
chi_square = np.zeros(1000)

for i in range(len(chi_square)):
    print(i)
    chi_square[i] = chisquare(samples.iloc[i, :], y, 1)
    
np.savetxt('chisquare.txt' , chi_square)
plt.plot(chi_square)
plt.savefig('chi.png')
