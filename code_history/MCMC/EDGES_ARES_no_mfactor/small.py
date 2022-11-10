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
scaling_factor = 1/5
A = 1000 * scaling_factor # *1000 to convert K to mK and divide by 2 (scaling factor)
#A = 1000 #only *1000 to convert K to mK, no scaling facotr
T_e = A* T_e
model_e = A* model_e

#Changing the data from frequency to redshift-------------------------------------------------------------------------------
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

#Defining fuctions----------------------------------------------------------------------------------------------------------
def chisquare (fobs, fexp): #the chi-square
    return np.sum((fobs-fexp)**2)

def dict_to_list(d): # converts dictionary to two lists
    #d must be a dictionary containing the value of parameters and their names
    key = list(d.keys())
    value = list(d.values())
    return value, key

def list_to_dict(value, key): #converts two lists to a dictionary
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

def pert_array(low, p, up):
    #p is an array of peak values for pert distribution
    #low: array of lower-bound values for pert distribution
    #up: array of upper-bound values for pert distribution
    # b is an array of pert outputs
    
    b = np.empty((len(p)))
    for i in range(len(p)):
        b[i]= pert(low[i], p[i], up[i])                 
    return b
        
#MCMC inputs (the guess subscipt is related to the original guess)------------------------------------------------------------------
#start_guess = {'fstar': 0.02, 'fX': 0.9, 'fesc':0.1, 'clumping_factor':0.1, 'cX' : 48.9E38}
               #'rad_yield': 2.6E39, 'lya_nmax': 23, Z = 0.02, Nlw=9690, Nion= 4000
start_guess = {'fstar': 0.02, 'fX': 0.9, 'fesc':0.1, 'clumping_factor':0.1, 'cX' : 48.9E38}
param_length = len(start_guess)
                 
#converting start guess to two lists
value_guess, key_guess = dict_to_list(start_guess)
                 
nstep = 30 #number of steps to be taken in the MCMC

# Idea for adding scaling factor
#------------------------------
#def intermediate (start_guess):
#    new_dict = start_guess - mult_factor
#    sim_guess = ares.simulations.Global21cm(**new_dict)
#    return sim_guess * start_guess['mult_factor']
#-------------------------------

#Input guess from ARES---------------------------------------------------------------------------------------------------------------
sim_guess = ares.simulations.Global21cm(**start_guess)
#sim_guess = ares.simulations.Global21cm() # default ares without change of params
sim_guess.run()

z_guess_raw= sim_guess.history['z'] #the raw output of ARES
T_guess_raw= sim_guess.history['dTb']

#interpolation
def interpolate(z_raw, T_raw):
    f = interp1d(z_raw, T_raw)
    T = f(z_e) #with regard to EDGES redshift range
    return T

T_guess = interpolate(z_guess_raw, T_guess_raw) #interpolating the raw output of ARES to match the redshift range of EDGES

#printing the minimums---------------------------------------------------------------------------------------------------------
m = np.array(model_e)
T = np.array(T_guess)
print('min model:' , m.min())
print('min ARES:' , T.min())
#Plotting data, paper model and original guess---------------------------------------------------------------------------------

fig1 = plt.figure()
#plt.plot(z_e, T_e, label='data')
plt.plot(z_e, T_e, label='EDGES Data')
plt.plot(z_e, model_e, label='EDGES model')
plt.plot(z_e, T_guess, label='Original Guess (ARES)')
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
plt.savefig('guess_model')
plt.show()

