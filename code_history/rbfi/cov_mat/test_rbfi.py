import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import normal
from math import ceil
from scipy.interpolate import CubicSpline
from scipy.interpolate import RBFInterpolator
from ares_params import redshifts
import pickle

#producing the edges model on desired redshif range (from the function mentioned in the paper)--------------------------------------------
def func_edges (x):
    #x is frequency
    #y is the array of paramaters
    #y[0]= v_0 is the center of frequency
    #y[1]= w
    #y[2] = tau
    #y[3]=A
    
    y = [78.3, 20.7, 6.5, 0.52] #the fitted parameters to the EDGES Data
    
    B = (4*(x-y[0])**2)*(y[1]**(-2))*(np.log((-1/y[2])*(np.log((1+np.exp(-1*y[2]))/2))))
    return -1 * y[3] * ((1-np.exp(-1*y[2]*np.exp(B)))/(1-np.exp(-1*y[2])))

#introducing the redshift range
z_e = np.linspace(5, 30 , 40)

def z_to_freq(z): # I have checked this! It's true!!!!!!
    v_0 = 1420 #MHz, frequency of 21cm line
    return v_0/(z+1)

#producing the data from the EDGES model
model_e = func_edges(z_to_freq(z_e))
model_e = 1000 * model_e #converting the Data temperatures from k to mK
model_e = model_e/2 #temporary mfactor

#----------------------------------------------------------------------------------------------------------------------------------------
file = open("curve_rbfi.pickle","rb")
rbfi = pickle.load(file)

def call_rbfi(rbfi, params):
    #params_denormalized = np.array(params, dtype='float64')
    #params_denormalized[0] = 10** (params[0])
    #params_denormalized[1] = 10** (params[1])
    #params_denormalized[2] = 10** (params[2])
    #return rbfi(np.append(redshifts[np.newaxis].T, np.tile(params_denormalized, (redshifts.shape[0],1)), axis = 1))
    return rbfi(np.append(redshifts[np.newaxis].T, np.tile(params, (redshifts.shape[0],1)), axis = 1))

def chisquare (fobs, fexp): #returns the chi-square of two values
    chi = (1E-2)*np.sum((fobs-fexp)**2)
    return chi
#-----------------------------------------------------------------------------------------------------------------------------------------
start_guess_1 = [1E4, 1E38, 1E4, 1]
start_guess_2 = [1E4, 1E39, 1E5, 1]

T_guess_1 = call_rbfi(rbfi, start_guess_1)
T_guess_2 = call_rbfi(rbfi, start_guess_2)
print(T_guess_1 - T_guess_2)

#print('chi-square_1: '+ repr(chisquare(T_guess_1, model_e)))
#print('chi-square_2: '+ repr(chisquare(T_guess_2, model_e)))

fig1 = plt.figure()
plt.plot(z_e, T_guess_1, label='guess_1')
plt.plot(z_e, T_guess_2, label='guess_2')
#plt.plot(z_e, mcmc_T, label="MCMC Result")
#plt.title('Result of MCMC (%d Steps)'%nstep, fontsize=12)
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
plt.savefig('test.png')
plt.show()