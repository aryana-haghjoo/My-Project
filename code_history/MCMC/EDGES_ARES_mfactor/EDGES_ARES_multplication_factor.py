import ares
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numpy.random import normal
from math import ceil

#loading the EDGES data (the e subscipt is related to EDGES)-----------------------------------------------------------------
data_1 = pd.read_csv('data_1.csv')

freq_e = data_1.iloc[:,0] #frequency, MHz

T_e = data_1.iloc[:,-1] #21cm brighness temperature, k

model_e = data_1.iloc[:, -2] #the model represented in the paper, k

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
start_guess = {'multiplication_factor': 5, 'fstar': 0.02, 'fX': 0.9, 'fesc':0.1, 'clumping_factor':0.1, 'cX' : 48.9E38} #Do not change the key to multplication factor or change the run_ares function respectively

param_length = len(start_guess)
                 
#converting start guess to two lists
value_guess, key_guess = dict_to_list(start_guess)
                 
nstep = 10 #number of steps to be taken in the MCMC

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

#Defining the MCMC chain--------------------------------------------------------------------------------------------------------
def mcmc (max_steps, start_guess):
    #just in order to check
    random_array = np.zeros(max_steps)
    prob_array = np.zeros(max_steps)
    #converting start guess to two lists
    value_guess, key_guess = dict_to_list(start_guess)
    
    #definig the chain
    chain = np.empty((max_steps, len(start_guess)))
    chain[0, :] = value_guess
    
    #defining the chi-square
    chisq = np.empty(max_steps)
    chisq[0] = chisquare(T_guess, model_e)
    
    #the chain 
    for i in range(1, max_steps):
        print('iteration number', i, 'of', max_steps)
        
        #these two bounds should be changed with regard to the reasonable bound for the fitting parameter
        low = np.zeros((param_length))
        up = np.ones((param_length))
        
        #changing the bounds to include the mfactor which is the 0th parameter
        low[0]= 2
        up[0]= 10
        
        #changing the bounds to include the c_x which is the 5th parameter
        low[5]= 5E38
        up[5]=400E38
        
        new_param = pert_array(low, chain[i-1,:], up)
        #print('new_param', new_param)
        new_param_dict = list_to_dict(new_param, key_guess)
        
        T = run_ares(new_param_dict)
        
        new_chisq = chisquare(T, model_e)
        
        #If chi-square gets big, we should do another step
        if new_chisq >= chisq[i-1]:
            prob = np.exp(-0.5*(new_chisq-chisq[i-1]))
            print(new_chisq-chisq[i-1])
            x=np.random.rand()
            if x >= prob:
                random_array[i] = x
                prob_array[i] = prob
                print('higher chi-square is accepted')
                chisq[i] = new_chisq
                chain[i, :] = new_param
                
            else:
                print('higher chi-square is not accepted')
                chisq[i] = chisq[i-1]
                chain[i, :] = chain[i-1, :]
                
        #if chi-square got small, we accept it        
        else:
            print('lower chi-square')
            chisq[i] = new_chisq
            chain[i, :] = new_param
    print('prob array', prob_array)  
    print('random array', random_array)
    return chain, chisq

#Running the MCMC-------------------------------------------------------------------------------------------------------
params, cs = mcmc(nstep, start_guess)
                 
#MCMC output------------------------------------------------------------------------------------------------------------
mcmc_param= np.empty(param_length)

for i in range(param_length):
    mcmc_param[i] = np.mean(params[:,i])
    
mcmc_param_dict = list_to_dict(mcmc_param, key_guess)

mcmc_T = run_ares(mcmc_param_dict)
                 
#Printing the mcmc outputs----------------------------------------------------------------------------------------------
print(mcmc_param_dict)

print("Chi squared of mcmc:", chisquare(mcmc_T, model_e))
print("Chi squared of original guess:", chisquare(T_guess, model_e))

#Plotting the mcmc outputs-----------------------------------------------------------------------------------------------
fig2 = plt.figure()
plt.plot(z_e, T_e, label='EDGES Data')
plt.plot(z_e, model_e, label='EDGES Model')
plt.plot(z_e, T_guess, label='Original Guess')
plt.plot(z_e, mcmc_T, label="MCMC Result")
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
plt.savefig('mcmc_result')
plt.show()


#Plotting the chi-square trend-------------------------------------------------------------------------------------------
fig3 = plt.figure()
plt.plot(np.log(cs))
plt.xlabel('number of steps', fontsize=12)
plt.title ('Chi-Square Trend', fontsize=12)
plt.ylabel('Log of Chi-Square', fontsize=12)
plt.savefig('chi-square')
plt.show()

#Plotting the parameters trend--------------------------------------------------------------------------------------------
fig4, ax_list = plt.subplots(ceil(param_length/2), 2, figsize=(13,10))
fig4.suptitle('Displaying The Trend of Parameters', fontsize=16)
if((param_length % 2) == 0):
    for i in range(ceil(param_length/2)):
        for j in range(2):
            ax_list[i, j].plot(params[:, i*2+ j])
            ax_list[i, j].set_ylabel(repr(key_guess[i*2+ j]), fontsize=16)
            ax_list[i, j].set_xlabel('number of steps', fontsize=12)

else:
    for i in range(ceil(param_length/2)):
        for j in range(2):
            if(j == 1 and i == (ceil(param_length/2)-1)):
                break
            ax_list[i, j].plot(params[:, i*2+ j])
            ax_list[i, j].set_ylabel(repr(key_guess[i*2+ j]), fontsize=16)
            ax_list[i, j].set_xlabel('number of steps', fontsize=12)
            

plt.tight_layout()
plt.savefig('parameters')
plt.show()    
                 
#Calculating the residuals--------------------------------------------------------------------------------------------------------------
r_paper_model = mcmc_T - model_e
r_data = mcmc_T - T_e                
                 
#Plotting the residuals-----------------------------------------------------------------------------------------------------------------
fig5 = plt.figure()
plt.plot(z_e, r_paper_model, label= 'EDGES Model')
plt.plot(z_e, r_data, label= 'EDGES Data')
plt.legend()
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.title('Residuals of MCMC Model With Respect to Data and Paper Model')
#plt.ylim(top=-0.4)
#plt.xlim(65,90)
plt.savefig('residuals')
plt.show()