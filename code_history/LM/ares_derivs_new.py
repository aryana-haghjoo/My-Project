import ares
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

#loading the EDGES data (the e subscipt is related to EDGES)-----------------------------------------------------------------
data_1 = pd.read_csv('data_1.csv')
freq_e = data_1.iloc[:,0] #frequency, MHz
model_e = data_1.iloc[:, -2] #the model represented in the paper, k
model_e = 1000 * model_e #converting the Data temperatures from k to mK 
model_e = model_e/2 #temporary mfactor

#Changing the axis from frequency to redshift---------------------------------------------------------------------------------
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

#defining functions-----------------------------------------------------------------------------------------------------------
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
    params_denormalized = list_to_dict(value_denormalized, key_guess)
    
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

def func_ares (m, z, d = 1000): 
    #m is the list of params 
    #z is the redshift range
    #y is the brightness temp
    m = np.array(m)
    T = call_ares (list_to_dict(m, key_guess), z)
    derivs = np.zeros([len(z), len(m)])
    dpars = np.zeros(len(m))
    dpars = m/d 
    for i in range(len(m)):        
        pars_plus = np.array(m, copy=True, dtype = 'float64')
        pars_plus[i] = pars_plus[i] + dpars[i]
        
        pars_minus = np.array(m, copy=True, dtype = 'float64')
        pars_minus[i] = pars_minus[i] - dpars[i]
        
        A_plus = call_ares (list_to_dict(pars_plus, key_guess), z)
        A_minus = call_ares (list_to_dict(pars_minus, key_guess), z)
        A_m = (A_plus - A_minus)/(2*dpars[i])
        derivs[:, i] = A_m    
    return T, derivs

def d(func, x, dx, params):
    return (func(x+dx, params) - func(x-dx, params))/(2*dx)

def d3(func, x, dx, params):
    return (-0.5*func(x-2*dx, params) + func(x-dx, params) - func(x+dx, params) + 0.5*func(x+2*dx, params))/(dx**3)

def ideal_dx(func, x, params, dx0 = 1E-3):
    third_deriv = d3(func, x, dx0, params)
    with np.errstate(divide = 'ignore'):
        dx1 = np.cbrt(3*func(x, params)*1E-16/(d3(func, x, dx0, params)))
    dx1[np.where(np.isnan(dx1))] = dx0
    dx1[np.where(np.isinf(dx1))] = dx0
    return np.abs(dx1)

def ndiff(fun , x, params, dx0 = 1E-3):
    # Estimate ideal dx, we use 1E-3, as a starting guess for dx
    dx = ideal_dx(fun, x, params, dx0)
    # Calculated the derivative with given dx
    df = d(fun, x, dx, params)
    """
    # Return if all we want is the derivative
    if not full:
        return df
    else: # Estimate error for full output
        error = np.abs(fun(x) * np.finfo(np.double).eps / dx + d3(fun,x,dx)*(dx**2))
    """
    #return df,dx,error
    return df

def func_ares (m, z, dx0 = 1E-3): 
    #m is the list of params 
    #z is the redshift range
    #y is the brightness temp
    m = np.array(m)
    T = call_ares (list_to_dict(m, key_guess), z)
    derivs = np.zeros([len(z), len(m)])
    dpars = np.zeros(len(m))
    dpars = m/dx0
    
    for i in range(len(m)):   
        pars_plus_1 = np.array(m, copy=True, dtype = 'float64')
        pars_plus_1[i] = pars_plus_1[i] + dpars[i]
        
        pars_plus_2 = np.array(m, copy=True, dtype = 'float64')
        pars_plus_2[i] = pars_plus_1[i] + 2* dpars[i]
        
        pars_minus_1 = np.array(m, copy=True, dtype = 'float64')
        pars_minus_1[i] = pars_plus_1[i] - dpars[i]
        
        pars_plus_2 = np.array(m, copy=True, dtype = 'float64')
        pars_plus_2[i] = pars_plus_1[i] - 2* dpars[i]
        
        #calculating the ideal dx
        third_deriv = (-0.5*call_ares(pars_minus_2, z) + call_ares(pars_minus_1, z) - call_ares(pars_plus_1, z) + 0.5*call_ares(pars_plus_2, z))/(dpars[i]**3)
        
        with np.errstate(divide = 'ignore'):
            dx = np.cbrt(3*call_ares(m, z)*1E-16/(third_deriv))
        dx[np.where(np.isnan(dx1))] = dx0
        dx[np.where(np.isinf(dx1))] = dx0
        
        #calculating the derivative
        df = (func(x+dx, params) - func(x-dx, params))/(2*dx)

    df = d(fun, x, dx, params)
    """
    # Return if all we want is the derivative
    if not full:
        return df
    else: # Estimate error for full output
        error = np.abs(fun(x) * np.finfo(np.double).eps / dx + d3(fun,x,dx)*(dx**2))
    """
    #return df,dx,error
    return df
    
    for i in range(len(m)):        
        derivs[:, i] = A_m    
        
    return T, derivs

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#start_guess = {'pop_rad_yield_0_': 10**(4.03), 'pop_rad_yield_1_': 10**(35), 'pop_rad_yield_2_': 10**(4), 'clumping_factor': 0.71}
start_guess = {'pop_rad_yield_0_': 4.03, 'pop_rad_yield_1_': 35, 'pop_rad_yield_2_': 4, 'clumping_factor': 0.71} 

value_guess, key_guess = dict_to_list(start_guess) #converting start guess to two lists
curve, derivs = func_ares(value_guess, z_e)

fig1 = plt.figure()
plt.plot(z_e , model_e, label = 'EDGES Model')
plt.plot(z_e , curve, label = 'Model')

#plt.plot(z_e , derivs [:, 0], label = 'Derivative With Respect to parameter 0')
#plt.plot(z_e , derivs [:, 1], label = 'Derivative With Respect to parameter 1')
#plt.plot(z_e , derivs [:, 2], label = 'Derivative With Respect to parameter 2')
#plt.plot(z_e , derivs [:, 3], label = 'Derivative With Respect to parameter 3')

plt.title('Result of Levenberg–Marquardt', fontsize=12)
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
plt.savefig('derivs.png')