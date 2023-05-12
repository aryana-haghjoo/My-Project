import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import ares

#loading the EDGES data (the e subscipt is related to EDGES)
data_1 = pd.read_csv('/home/o/oscarh/aryanah/My-Project/data/data_1.csv')
freq_e = data_1.iloc[:,0] #frequency, MHz

#Changing the axis from frequency to redshift
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

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

def ares_deriv (m, z, d = 1E-2): 
    #I can further change this function to include the best dx - 4*int(1E5) is the best number I found so far
    #m is the list of params 
    #z is the redshift range
    #y is the brightness temp
    m = np.array(m)
    T = call_ares(list_to_dict(m, key), z)
    derivs = np.zeros([len(z), len(m)])
    dpars = d * m
    dpars = np.array(dpars, copy=True, dtype = 'float64')
    
    for i in range(len(m)):        
        pars_plus = np.array(m, copy=True, dtype = 'float64')
        pars_plus[i] = pars_plus[i] + dpars[i]
        
        pars_minus = np.array(m, copy=True, dtype = 'float64')
        pars_minus[i] = pars_minus[i] - dpars[i]
        
        A_plus = call_ares (list_to_dict(pars_plus, key), z)
        A_minus = call_ares (list_to_dict(pars_minus, key), z)
        A_m = (A_plus - A_minus)/(2*dpars[i])
        derivs[:, i] = A_m    
    return T, derivs

def get_matrices(m, fun, x, y, Ninv):
    model, derivs = fun(m, x)
    r = y-model
    lhs = derivs.T@Ninv@derivs
    rhs = derivs.T@(Ninv@r)
    chisq = r.T@Ninv@r
    return chisq, lhs, rhs

#dict_true = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_2_': 1E5, 'clumping_factor': 2.5, 'fX': 0.1} 
dict_true = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_2_': 1E3, 'fesc': 0.1, 'fX': 0.1}
m_true, key = dict_to_list(dict_true)
m_true = np.array(m_true, copy=True, dtype = 'float64')
y_true = call_ares(list_to_dict(m_true, key), z_e)
err = 1E-3
Ninv = ((err)**(-2))*np.eye(len(z_e))

fig1 = plt.plot()
plt.plot(z_e, y_true)
plt.title('ARES Curve for Starting Set of Parameters')
plt.savefig('curve_start.png')

chisq_f, lhs_f, rhs_f = get_matrices(m_true, ares_deriv, z_e, y_true, Ninv)
mycov = lhs_f
mycovinv= np.linalg.inv(mycov)

txt = open('cov_mat.txt','w')
txt.write('Starting Point: ' + repr(m_true) + '\n')
txt.write('Chi-Square at the starting point: ' +repr(chisq_f) + '\n')
txt.write('Inverse of Covariance Matrix: ' + repr(mycovinv) + '\n')
txt.close()