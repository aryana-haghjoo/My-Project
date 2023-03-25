import ares
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numpy.random import normal
from math import ceil
from scipy.interpolate import CubicSpline

z_e = np.linspace(5,35, 50) 
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


#----------------------------------------------------------------------------------------------------------------------------------


#Defining fuctions-------------------------------------------------------------------------------------------------------------
def chisquare (fobs, fexp): #returns the chi-square of two values
    return (1E-6)*np.sum((fobs-fexp)**2)

def dict_to_list(d): # converts dictionary to two lists (key and value)
    #d must be a dictionary containing the value of parameters and their names
    key = list(d.keys())
    value = list(d.values())
    return value, key

def list_to_dict(value, key): #converts two lists (key and value) to a dictionary
    #value is a list of parameters' values
    #key is a list parameters' names
    return dict(zip(key, value))

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
    #m_factor = start_guess['multiplication_factor'] #mulplication factor is >1 and will be multplied to ARES results to match the EDGES data
    #new_dict = remove_element(param_dict_m_factor, 'multiplication_factor')
    
    new_dict = param_dict_m_factor
    #temp = new_dict['cX']
    #new_dict['cX'] = (1E39) * temp # in order to get rid of the great factor of C_X while runing the chain
    
    sim = ares.simulations.Global21cm(**new_dict)
    sim.run()
    
    z_raw = sim.history['z'] #the raw output of ARES
    T_raw = sim.history['dTb']
    T = interpolate(z_raw, T_raw) #interpolating the raw output of ARES to match the redshift range of EDGES
    #T = T * m_factor #multiply the model with m_factor
   
    return T

def check_limits(new_param, low, up): #to check if the new params exceed the reasonable limits or not (e.g f_Star>1)
    l = len(new_param)
        
    check_array=np.zeros(l, dtype=bool)
    result = True
        
    for i in range (0,l):
        
        if new_param[i]<= up[i]:
            
            if new_param[i] > low[i]:
                check_array[i] = True
            else: 
                check_array[i] =False
        else: 
            check_array[i] =False
        
        result = result & check_array[i] 
        
        
    return result

#MCMC inputs (the guess subscipt is related to the original guess)------------------------------------------------------------------
start_guess = {'fstar': 0.12, 'fX': 0.12, 'fesc':0.12, 'clumping_factor':0.12} #Do not change the key to multplication factor or change the run_ares function respectively

#resonable range for each parameter so ARES won't crash!
up_bound = [10, 1, 1, 1, 1, 40]
low_bound = [2, 0, 0 , 0 , 0, 0.5]

#deviation of change for each param
dev = [0.0004, 0.002, 0.002, 0.002]

param_length = len(start_guess)
                 
#converting start guess to two lists
value_guess, key_guess = dict_to_list(start_guess)
                 
nstep = 50 #number of steps to be taken in the MCMC

#Input guess from ARES---------------------------------------------------------------------------------------------------------------
T_guess = call_ares(start_guess, z_e)

#Plotting data, paper model and original guess---------------------------------------------------------------------------------------
"""
fig1 = plt.figure()
plt.plot(z_e, T_e, label='EDGES data')
plt.plot(z_e, model_e, label='EDGES Model')
plt.plot(z_e, T_guess, label='Original Guess')
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
plt.savefig('/scratch/o/oscarh/aryanah/MCMC_output/guess_model.png')
plt.show()
"""

#Defining the MCMC chain--------------------------------------------------------------------------------------------------------
def mcmc (max_steps, start_guess):
    
    #converting start guess to two lists
    value_guess, key_guess = dict_to_list(start_guess)
    
    #definig the chain
    chain = np.empty((max_steps, len(start_guess)))
    chain[0, :] = value_guess
     
    #defining the chi-square
    chisq = np.empty(max_steps)
    chisq[0] = chisquare(T_guess, model_e)
    
    #writing the details of chain in a txt file
    #txt_1 = open('mcmc_detail.txt','w')
    
    #the chain 
    for i in range(1, max_steps):
        print('iteration number', i, 'of', max_steps)
        #txt_1.write('iteration number  '+ repr(i) + '  of  ' + repr(max_steps) + '\n')
        
        '''
        while (True): # to chack if  all the parameters are in the reasonable limit
            new_param = normal(chain[i-1, :], dev) #new random point in the parameter space
            result = check_limits(new_param, low_bound, up_bound)
            if result: # if all the parameters are in the reasonable limit, the new_param is accepted
                break
        '''
        new_param = normal(chain[i-1, :], dev) #new random point in the parameter space
        new_param_dict = list_to_dict(new_param, key_guess)
        
        T = call_ares(new_param_dict, z_e)
        
        new_chisq = chisquare(T, model_e)
        
        #If chi-square gets big, we should do another step
        if new_chisq >= chisq[i-1]:
            prob = np.exp(-0.5*(new_chisq-chisq[i-1]))
            #txt_1.write('chi-square difference is  '+ repr(new_chisq-chisq[i-1]) + '\n')
            x=np.random.rand()
            
            #print('probability is', prob)
            #print('random number is', x)
            
            #txt_1.write('probability is  '+ repr(prob) + '\n')
            #txt_1.write('random number is  '+ repr(x) + '\n')
            
            if x >= prob: #if the random number is bigger than our probability
                #print('higher chi-square is accepted')
                #txt_1.write('higher chi-square is accepted'+ '\n')
                chisq[i] = new_chisq
                chain[i, :] = new_param
                
            else:
                #print('higher chi-square is not accepted')
                #txt_1.write('higher chi-square is not accepted'+ '\n')
                chisq[i] = chisq[i-1]
                chain[i, :] = chain[i-1, :]
                
        #if chi-square got small, we accept it        
        else:
            #print('lower chi-square')
            #txt_1.write('lower chi-square'+ '\n')
            chisq[i] = new_chisq
            chain[i, :] = new_param
                    
    #txt_1.close()        
    return chain, chisq

#Running the MCMC-------------------------------------------------------------------------------------------------------
params, cs = mcmc(nstep, start_guess)

#printting the outputs to a file
#txt_3 = open('mcmc_params.txt','w')
#txt_3.write(repr(params))
#txt_3.close()
np.savetxt('mcmc_params.gz' , params)
#np.savetxt('/scratch/o/oscarh/aryanah/MCMC_output/mcmc_params.gz' , params)
                 
#MCMC output------------------------------------------------------------------------------------------------------------
mcmc_param= np.empty(param_length)


for i in range(param_length):
    mcmc_param[i] = np.mean(params[:,i]) #array of best parameters
    
mcmc_param_dict = list_to_dict(mcmc_param, key_guess)

mcmc_T = call_ares(mcmc_param_dict, z_e)
                 
#Printing the mcmc outputs----------------------------------------------------------------------------------------------
#printing the results on the terminal
#print(mcmc_param_dict)

#print("Chi-squared of mcmc:", chisquare(mcmc_T, model_e))
#print("Chi-squared of original guess:", chisquare(T_guess, model_e))

#printing the results to the txt file
txt_2=open('mcmc_result.txt','w')
#txt_2=open('/scratch/o/oscarh/aryanah/MCMC_output/mcmc_result.txt','w')
txt_2.write(repr(mcmc_param_dict) + '\n')
txt_2.write("Chi-squared of mcmc:"+ repr(chisquare(mcmc_T, model_e))+ '\n')
txt_2.write("Chi-squared of original guess:"+ repr(chisquare(T_guess, model_e))+ '\n')
txt_2.close()

#Plotting the mcmc outputs-----------------------------------------------------------------------------------------------
fig2 = plt.figure()
plt.plot(z_e, T_e, label='EDGES Data')
plt.plot(z_e, model_e, label='EDGES Model')
plt.plot(z_e, T_guess, label='Original Guess')
plt.plot(z_e, mcmc_T, label="MCMC Result")
plt.title('Result of MCMC after %d Steps'%nstep, fontsize=12)
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
plt.savefig('mcmc_result.png')
#plt.savefig('/scratch/o/oscarh/aryanah/MCMC_output/mcmc_result.png')
plt.show()


#Plotting the chi-square trend-------------------------------------------------------------------------------------------
fig3 = plt.figure()
plt.plot(np.log(cs))
plt.xlabel('number of steps', fontsize=12)
plt.title ('Chi-Square Trend after %d Steps'%nstep, fontsize=12)
plt.ylabel('Log of Chi-Square', fontsize=12)
plt.savefig('chi-square.png')
#plt.savefig('/scratch/o/oscarh/aryanah/MCMC_output/chi-square.png')
plt.show()

#Plotting the parameters trend--------------------------------------------------------------------------------------------
fig4, ax_list = plt.subplots(ceil(param_length/2), 2, figsize=(13,10))
fig4.suptitle('Parameters Trend after %d Steps'%nstep, fontsize=16)
#fig4.suptitle('Displaying The Trend of Parameters', fontsize=16)
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
plt.savefig('parameters.png')
#plt.savefig('/scratch/o/oscarh/aryanah/MCMC_output/parameters.png')
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
plt.title('Residuals of MCMC Model after %d Steps'%nstep, fontsize=16)
plt.savefig('residuals.png')
#plt.savefig('/scratch/o/oscarh/aryanah/MCMC_output/residuals.png')
plt.show()

#error-----------------------------------------------------------------------------------------------------------------------------------
error = r_paper_model/mcmc_T

#Plotting the error-----------------------------------------------------------------------------------------------------------------
fig6 = plt.figure()
plt.plot(z_e, error, label= 'error')
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('error', fontsize=12)
plt.title('Relative Error of EDGES Model and MCMC result', fontsize=12)
plt.savefig('error.png')
#plt.savefig('/scratch/o/oscarh/aryanah/MCMC_output/error.png')
plt.show()