#copy the one from the other file, this is not okay
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
start_guess = {'fstar': 0.02, 'fX': 0.9, 'fesc':0.1, 'clumping_factor':0.1, 'cX' : 48.9E38}
param_length = len(start_guess)

#scaling factor
A = 200
                 
#converting start guess to two lists
value_guess, key_guess = dict_to_list(start_guess)
                 
nstep = 30 #number of steps to be taken in the MCMC

#Input guess from ARES---------------------------------------------------------------------------------------------------------------
sim_guess = ares.simulations.Global21cm(**start_guess)
sim_guess.run()

z_guess_raw= sim_guess.history['z'] #the raw output of ARES
T_guess_raw= sim_guess.history['dTb']

#interpolation
def interpolate(z_raw, T_raw):
    f = interp1d(z_raw, T_raw)
    T = f(z_e) #with regard to EDGES redshift range
    return T

T_guess = interpolate(z_guess_raw, T_guess_raw) #interpolating the raw output of ARES to match the redshift range of EDGES

#Plotting data, paper model and original guess---------------------------------------------------------------------------------
T_e = A* T_e #I do not know where this multiplication factor comes from, but the base of depth for ARES, seems to be 240 times lower than that of EDGES data
model_e = A* model_e

fig1 = plt.figure()
plt.plot(z_e, T_e, label='EDGES data')
plt.plot(z_e, model_e, label='EDGES Model')
plt.plot(z_e, T_guess, label='Original Guess')
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(K)', fontsize=12)
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
    chisq[0] = chisquare(T_guess, T_e)
    
    #the chain 
    for i in range(1, max_steps):
        print('iteration number', i, 'of', max_steps)
        
        #these two bounds should be changed with regard to the reasonable bound for the fitting parameter
        low = np.zeros((param_length))
        up = np.ones((param_length))
        
        #changing the bounds to include the c_x which is the 5th parameter
        low[4]= 5E38
        up[4]=400E38
                     
        new_param = pert_array(low, chain[i-1,:], up)
        print('new_param', new_param)
        new_param_dict = list_to_dict(new_param, key_guess)
        
        sim= ares.simulations.Global21cm(**new_param_dict)
        sim.run()

        z_raw= sim.history['z'] #the raw output of ARES
        T_raw= sim.history['dTb']
        T = interpolate(z_raw, T_raw) #interpolating the raw output of ARES to match the redshift range of EDGES
        
        new_chisq = chisquare(T, T_e)
        
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
        
mcmc_sim= ares.simulations.Global21cm(**mcmc_param_dict)
mcmc_sim.run()

mcmc_z_raw= mcmc_sim.history['z'] #the raw output of ARES
mcmc_T_raw= mcmc_sim.history['dTb']
mcmc_T = interpolate(mcmc_z_raw, mcmc_T_raw) #interpolating the raw output of ARES to match the redshift range of EDGES
                 
#Printing the mcmc outputs----------------------------------------------------------------------------------------------
print(mcmc_param_dict)

print("Chi squared of mcmc:", chisquare(mcmc_T, T_e))
print("Chi squared of original guess:", chisquare(T_guess, T_e))

#Plotting the mcmc outputs-----------------------------------------------------------------------------------------------
fig2 = plt.figure()
plt.plot(z_e, T_e, label='EDGES Data')
plt.plot(z_e, model_e, label='EDGES Model')
plt.plot(z_e, T_guess, label='Original Guess')
plt.plot(z_e, mcmc_T, label="MCMC Result")
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(K)', fontsize=12)
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
plt.ylabel('T(K)', fontsize=12)
plt.title('Residuals of MCMC Model With Respect to Data and Paper Model')
#plt.ylim(top=-0.4)
#plt.xlim(65,90)
plt.savefig('residuals')
plt.show()