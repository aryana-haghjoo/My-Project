import ares
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numpy.random import normal
from math import ceil

#loading the EDGES data (the e subscipt is related to EDGES)-----------------------------------------------------------------
data_1 = pd.read_csv('data_1.csv')

freq_e = data_1.iloc[:,0] #frequency

T_e = data_1.iloc[:,-1] #21cm brighness temperature

model_e = data_1.iloc[:, -2] #the model represented in the paper

#Changing the data from frequency to redshift-------------------------------------------------------------------------------
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

#converting the Data temperatures from k to mK 
k = 1000
T_e = k * T_e 
model_e = k * model_e

#Defining fuctions----------------------------------------------------------------------------------------------------------
def chisquare (fobs, fexp): #returns the chi-square of two values
    return np.sum((fobs-fexp)**2)

def dict_to_list(d): # converts dictionary to two lists (key and value)
    #d must be a dictionary containing the value of parameters and their names
    key = list(d.keys())
    value = list(d.values())
    return value, key

def list_to_dict(value, key): #converts two lists (key and value) to a dictionary
    #value is a list of parameters' values
    #key is a list parameters' names
    return dict(zip(key, value))

def pert(low, p, up): #the PERT distribution function (to be used for random choise of parameters in MCMC)
    # low: lower bound
    #p: peak
    #up: upper bound
    
    lamb=4
    r = up - low
    alpha = 1 + lamb * (p-low) / r
    beta = 1 + lamb * (up - p) / r
    d = np.random.beta(alpha, beta, size=1)
    d = d * r
    return low + d

def pert_array(low, p, up): #returns an array of random numbers from the pert distribution with the following characteristics
    #p is an array of peak values for pert distribution
    #low: array of lower-bound values for pert distribution
    #up: array of upper-bound values for pert distribution
    # b is an array of pert outputs
    
    b = np.empty((len(p)))
    for i in range(len(p)):
        b[i]= pert(low[i], p[i], up[i])                 
    return b

def remove_element(d, key): #removes an element from a dictionary by specifing it's key
    #d: dictionary
    #key: key to the element that should be deleted
    r = dict(d)
    del r[key]
    return r

def interpolate(z_raw, T_raw): #interpolates ARES output to find the dT_b for the exact redshift points in the data
    f = interp1d(z_raw, T_raw)
    T = f(z_e) #with regard to EDGES redshift range
    return T
    
def run_ares(param_dict_m_factor): #function to run the ARES with a dictionary of parameters which the multiplication factor is removed from
    #start_guess: dictionary
    m_factor = start_guess['multiplication_factor'] #mulplication factor is >1 and will be multplied to ARES results to match the data
    new_dict = remove_element(param_dict_m_factor, 'multiplication_factor')
    sim = ares.simulations.Global21cm(**new_dict)
    sim.run()
    
    z_raw = sim.history['z'] #the raw output of ARES
    T_raw = sim.history['dTb']
    T = interpolate(z_raw, T_raw) #interpolating the raw output of ARES to match the redshift range of EDGES
    T = T * m_factor #multiply the model with m_factor
   
    return T

#MCMC inputs (the guess subscipt is related to the original guess)------------------------------------------------------------------
start_guess = {'multiplication_factor': 5, 'fstar': 0.02, 'fX': 0.9, 'fesc':0.1, 'clumping_factor':0.1, 'cX' : 48.9E38} #Do not change the key to multplication factor or change the run_ares function too

param_length = len(start_guess)
                 
#converting start guess to two lists
value_guess, key_guess = dict_to_list(start_guess)
                 
nstep = 30 #number of steps to be taken in the MCMC

#Input guess from ARES---------------------------------------------------------------------------------------------------------------
T_guess = run_ares(start_guess)

#Plotting data, paper model and original guess---------------------------------------------------------------------------------
fig1 = plt.figure()
plt.plot(z_e, T_e, label='EDGES data')
plt.plot(z_e, model_e, label='EDGES Model')
plt.plot(z_e, T_guess, label='Original Guess')
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
plt.savefig('guess_model')
plt.show()