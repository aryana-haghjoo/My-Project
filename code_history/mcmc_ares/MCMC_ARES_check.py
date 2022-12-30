import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil
from scipy.interpolate import CubicSpline
import ares

#loading the EDGES data (the e subscipt is related to EDGES)-----------------------------------------------------------------
data_1 = pd.read_csv('data_1.csv')
freq_e = data_1.iloc[:,0] #frequency, MHz
#model_e = data_1.iloc[:, -2] #the model represented in the paper, k
#model_e = 1000 * model_e #converting the Data temperatures from k to mK 
#model_e = model_e/2 #temporary mfactor

#Changing the axis from frequency to redshift---------------------------------------------------------------------------------
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

#Defining fuctions------------------------------------------------------------------------------------------------------------------------
def dict_to_list(d): # converts dictionary to two lists (key and value)
    #d must be a dictionary containing the value of parameters and their names
    key = list(d.keys())
    value = list(d.values())
    return value, key

def list_to_dict(value, key): #converts two lists (key and value) to a dictionary
    #value is a list of parameters' values
    #key is a list parameters' names
    return dict(zip(key, value))

def check_limits(m):
#'pop_rad_yield_1': upper limit: 96
#'pop_rad_yield_2': upper limit: 31
#'clumping factor': upper limit: 28

    m_new = np.array(m, copy=True, dtype = 'float64')
    if m[1] > 96:
        print('out of bound param 1')
        m_new[1] = 96
        
    if m[2] > 31:
        print('out of bound param 2')
        m[2] = 31
        
    if m[3] > 28:
        print('out of bound param 3')
        m[3] = 28
        
    return m_new

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
    spline = CubicSpline(z, dTb)
    return spline(redshifts) 

def chisquare (pars, data, err): #returns the chi-square of two values-err can be a number/array
    pred = call_ares(list_to_dict(pars, key_guess), z_e)
    chisq = np.sum((pred-data)**2/err**2)
    return chisq

#Defining the MCMC chain--------------------------------------------------------------------------------------------------------
def mcmc(fun, start_guess, data, err, dev, nstep):    
    #definig the chain
    chain = np.empty((nstep, len(start_guess)))
    chain[0, :] = start_guess
     
    #defining the chi-square
    chisq = np.zeros(nstep)
    chisq[0] = fun(start_guess, data, err)
    acceptance_ratio = 0
        
    #the chain 
    for i in range(1, nstep):
        #print('iteration number', i, 'of', nstep)   
        new_param = np.random.normal(chain[i-1, :], dev)
        new_chisq = fun(new_param, data, err)
        
        if new_chisq <= chisq[i-1]:
            acceptance_ratio = acceptance_ratio + 1
            chisq[i] = new_chisq
            chain[i, :] = new_param 
   
        else :
            if np.random.rand(1)<np.exp(-0.5*(new_chisq-chisq[i-1])):
                acceptance_ratio = acceptance_ratio + 1
                chisq[i] = new_chisq
                chain[i, :] = new_param
            else:
                chisq[i] = chisq[i-1]
                chain[i, :] = chain[i-1, :]          
    return chain, chisq, acceptance_ratio/nstep

#MCMC inputs ----------------------------------------------------------------------------------------------------------
start_guess = {'pop_rad_yield_0_': 3.5, 'pop_rad_yield_1_': 21.39, 'pop_rad_yield_2_': 4.71, 'clumping_factor': 0.91} 
value_guess, key_guess = dict_to_list(start_guess) #true params
T_guess = call_ares(start_guess, z_e)
deviation = 0.01 * np.asarray(value_guess)
param_length = len(start_guess)
nstep = 10
err = 1

y_true = call_ares(start_guess, z_e)
model_e = y_true + np.random.randn(len(z_e))*0.1
m0 = value_guess + np.random.randn(param_length)*0.1

#Running the MCMC-------------------------------------------------------------------------------------------------------
params, cs, acceptance_ratio = mcmc(chisquare, m0, model_e, err, deviation, nstep)
#np.savetxt('params.gz' , params)
np.savetxt('/scratch/o/oscarh/aryanah/output_1/params.gz' , params)

#Fourier Transform------------------------------------------------------------------------------------------------------
ps = np.zeros((nstep, param_length))
for i in range(param_length):
    ps[:, i] = np.abs(np.fft.fft(params[:, i]))**2
    
freqs = np.fft.rfftfreq(nstep)
idx = np.argsort(freqs)
#np.savetxt('ps.gz' , ps)
np.savetxt('/scratch/o/oscarh/aryanah/output_1/ps.gz' , ps)

#MCMC output------------------------------------------------------------------------------------------------------------
mcmc_param= np.empty(param_length)
for i in range(param_length):
    mcmc_param[i] = np.mean(params[:,i]) #array of best parameters  
mcmc_param_dict = list_to_dict(mcmc_param, key_guess)
mcmc_T = call_ares(mcmc_param_dict, z_e)

#Printing the mcmc outputs---------------------------------------------------------------------------------------------
#txt_2=open('mcmc_result.txt','w')
txt_2=open('/scratch/o/oscarh/aryanah/output_1/mcmc_result.txt','w')
txt_2.write(repr(mcmc_param_dict) + '\n')
txt_2.write("Chi-squared of mcmc:"+ repr(chisquare(mcmc_param, model_e, err))+ '\n')
txt_2.write("Chi-squared of original guess:"+ repr(chisquare(m0, model_e, err))+ '\n')
txt_2.write("acceptance_ratio for %d Steps: " %nstep + repr(acceptance_ratio*100) +"%"+ '\n')
txt_2.write("The final Curve:" + '\n')
txt_2.write(repr(mcmc_T))
txt_2.close()

#Plotting the mcmc outputs-----------------------------------------------------------------------------------------------
fig2 = plt.figure()
plt.plot(z_e, model_e, label='Start Guess')
plt.plot(z_e, T_guess, label='True Curve')
plt.plot(z_e, mcmc_T, label="MCMC Result")
plt.title('Result of MCMC (%d Steps)'%nstep, fontsize=12)
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
#plt.savefig('mcmc_result.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_1/mcmc_result.png')
plt.show()

#Plotting the chi-square trend-------------------------------------------------------------------------------------------
fig3 = plt.figure()
#plt.plot(np.log(cs))
plt.semilogy(cs)
plt.xlabel('number of steps', fontsize=12)
plt.title ('Chi-Square Trend (%d Steps)'%nstep, fontsize=12)
plt.ylabel('Chi-Square', fontsize=12)
#plt.savefig('chi-square.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_1/chi-square.png')
plt.show()

#Plotting the parameters trend--------------------------------------------------------------------------------------------
fig4, ax_list = plt.subplots(ceil(param_length/2), 2, figsize=(13,10))
fig4.suptitle('Chain (%d Steps)'%nstep, fontsize=16)
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
#plt.savefig('parameters.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_1/parameters.png')
plt.show()    

#Plotting the Fourier Transform-------------------------------------------------------------------------------------------
fig4, ax_list = plt.subplots(ceil(param_length/2), 2, figsize=(13,10))
fig4.suptitle('Power Spectrum of the Chain (%d Steps)'%nstep, fontsize=16)
if((param_length % 2) == 0):
    for i in range(ceil(param_length/2)):
        for j in range(2):
            ax_list[i, j].loglog(freqs[idx], ps[idx, i*2+ j])
            #plt.yscale("log")
            ax_list[i, j].set_ylabel(repr(key_guess[i*2+ j]), fontsize=16)
            ax_list[i, j].set_xlabel('frequency', fontsize=12)
            #ax_list[i, j].set_ylim(0, 1E2)
            #ax_list[i, j].set_xlim(0.2, 0.5)

else:
    for i in range(ceil(param_length/2)):
        for j in range(2):
            if(j == 1 and i == (ceil(param_length/2)-1)):
                break
            ax_list[i, j].loglog(freqs[idx], ps[idx, i*2+ j])
            #plt.yscale("log")
            ax_list[i, j].set_ylabel(repr(key_guess[i*2+ j]), fontsize=16)
            ax_list[i, j].set_xlabel('frequency', fontsize=12)
            #ax_list[i, j].set_ylim(0, 1E2)
            #ax_list[i, j].set_xlim(0.2, 0.5)
            
            

plt.tight_layout()
#plt.savefig('fourier.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_1/fourier.png')
plt.show()    

#Corner Plots--------------------------------------------------------------------------------------------------------------
fig, ax_list = plt.subplots(3, 2, figsize=(12,10))
fig.suptitle('Corner plots of the chain', fontsize=16)

ax_list[0, 0].plot(params[:, 0], params[:, 1], linestyle = "", marker=".")
ax_list[0, 0].set_ylabel(repr(key_guess[1]), fontsize=12)
ax_list[0, 0].set_xlabel(repr(key_guess[0]), fontsize=12)

ax_list[0, 1].plot(params[:, 0], params[:, 2], linestyle = "", marker=".")
ax_list[0, 1].set_ylabel(repr(key_guess[2]), fontsize=12)
ax_list[0, 1].set_xlabel(repr(key_guess[0]), fontsize=12)
    
ax_list[1, 0].plot(params[:, 0], params[:, 3], linestyle = "", marker=".")
ax_list[1, 0].set_ylabel(repr(key_guess[3]), fontsize=12)
ax_list[1, 0].set_xlabel(repr(key_guess[0]), fontsize=12)

ax_list[1, 1].plot(params[:, 1], params[:, 2], linestyle = "", marker=".")
ax_list[1, 1].set_ylabel(repr(key_guess[2]), fontsize=12)
ax_list[1, 1].set_xlabel(repr(key_guess[1]), fontsize=12)

ax_list[2, 0].plot(params[:, 1], params[:, 3], linestyle = "", marker=".")
ax_list[2, 0].set_ylabel(repr(key_guess[3]), fontsize=12)
ax_list[2, 0].set_xlabel(repr(key_guess[1]), fontsize=12)

ax_list[2, 1].plot(params[:, 2], params[:, 3], linestyle = "", marker=".")
ax_list[2, 1].set_ylabel(repr(key_guess[3]), fontsize=12)
ax_list[2, 1].set_xlabel(repr(key_guess[2]), fontsize=12)

plt.tight_layout()
#plt.savefig('corner_plots.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_1/corner_plots.png')
plt.show()