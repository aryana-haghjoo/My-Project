import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import normal
from math import ceil
from scipy.interpolate import CubicSpline
from scipy.interpolate import RBFInterpolator
from ares_params import redshifts
import pickle

#producing the edges model on desired redshif range (from the function mentioned in the paper)---------------------------------------------
def func (x):
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

def z_to_freq(z):
    v_0 = 1420 #MHz, frequency of 21cm line
    return v_0/(z+1)

#producing the model from the EDGES data
model_e = func(z_to_freq(z_e))

#converting the Data temperatures from k to mK ----------------------------------------------------------------------------------------
k = 1000 
model_e = k * model_e

#temporary mfactor
model_e = model_e/2

#Defining fuctions------------------------------------------------------------------------------------------------------------------------
def chisquare (fobs, fexp): #returns the chi-square of two values
    #The !E_6 factor is because of mk
    return (1E-6)*np.sum((fobs-fexp)**2)
    #return np.sum(((fobs-fexp)**2) / fexp)

def dict_to_list(d): # converts dictionary to two lists (key and value)
    #d must be a dictionary containing the value of parameters and their names
    key = list(d.keys())
    value = list(d.values())
    return value, key

def list_to_dict(value, key): #converts two lists (key and value) to a dictionary
    #value is a list of parameters' values
    #key is a list parameters' names
    return dict(zip(key, value))

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

def normalize(x): #this is based on the choise of params, some of them need to be normalized with logarithmic scales
    y = np.copy(x)
    y[0] = np.log10(x[0])
    y[1] = np.log10(x[1])
    y[2] = np.log10(x[2])
    return y

def denormalize(x): #this is based on the choise of params, some of them need to be normalized with logarithmic scales
    y = np.copy(x)
    y[0] = 10** (x[0])
    y[1] = 10** (x[1])
    y[2] = 10** (x[2])
    return y

def denormalize_many(x):
    x[:, 0] = 10** (x[:, 0])
    x[:, 1] = 10** (x[:, 1])
    x[:, 2] = 10** (x[:, 2])
    return x    

#RBFI--------------------------------------------------------------------------------------------------------------------------------
file = open("curve_rbfi.pickle","rb")
rbfi = pickle.load(file)

def call_rbfi(rbfi, params):
	return rbfi(np.append(redshifts[np.newaxis].T, np.tile(params,(redshifts.shape[0],1)), axis = 1))


#MCMC inputs (the guess subscipt is related to the original guess)------------------------------------------------------------------
start_guess = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_1_': 1E38, 'pop_rad_yield_2_': 1E4, 'clumping_factor': 1}
#["pop_rad_yield_0_", 1E3, 1E6, "log"], ["pop_rad_yield_1_", 1E37, 5E40, "log"], ["pop_rad_yield_2_", 1E3, 5E6, "log"], ["clumping_factor", 0.05, 2, "lin"]

#converting start guess to two lists
value_guess, key_guess = dict_to_list(start_guess)

#normalizing the guess
value_guess_normalized = normalize(value_guess)

#resonable range for each parameter so ARES won't crash! (ranges are normalized, the first three params are in logarithmic scales)
low_bound = np.array([3, 37, 3, 0.05])
up_bound = np.array([6, 40, 6, 2])

#deviation of change for each param
dev = 0.2 * (up_bound - low_bound)

param_length = len(start_guess)

nstep = 1000000 #number of steps to be taken in the MCMC

#Input guess from ARES---------------------------------------------------------------------------------------------------------------
#T_guess = call_rbfi(rbfi, value_guess_normalized)
T_guess = call_rbfi(rbfi, value_guess)

#Plotting data, paper model and original guess---------------------------------------------------------------------------------------
"""
fig1 = plt.figure()
plt.plot(z_e, model_e, label='EDGES Model')
plt.plot(z_e, T_guess, label='Original Guess')
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
#plt.savefig('/scratch/o/oscarh/aryanah/rbfi_output/guess_model.png')
#plt.savefig('guess_model.png')
plt.show()
"""

#Defining the MCMC chain--------------------------------------------------------------------------------------------------------
def mcmc (max_steps, start_guess):
    
    #samples from covariance matrix
    #samples = np.loadtxt('samples.txt')
    
    #definig the chain
    chain = np.empty((max_steps, len(start_guess)))
    chain[0, :] = start_guess
     
    #defining the chi-square
    chisq = np.empty(max_steps)
    chisq[0] = chisquare(T_guess, model_e)
    
    acceptance_ratio = 0
    
    #the chain 
    for i in range(1, max_steps):
        print('iteration number', i, 'of', max_steps)       
        while (True): # to chack if  all the parameters are in the reasonable limit
            new_param = normal(chain[i-1, :], dev) #new random point in the parameter space
            #new_param = samples[i, :]
            result = check_limits(new_param, low_bound, up_bound)
            if result: # if all the parameters are in the reasonable limit, the new_param is accepted
                break
        
        
        #new_param_dict = list_to_dict(new_param, key_guess)
        #new_param = samples[i, :]
        
        #new_param_denormalized = denormalize(new_param)
        
        T = call_rbfi(rbfi, denormalize(new_param))
        new_chisq = chisquare(T, model_e)
        #If chi-square gets big, we should do another step
        if new_chisq >= chisq[i-1]:
            prob = np.exp(-0.5*(new_chisq-chisq[i-1]))
            #print('chi-square difference is  '+ repr(new_chisq-chisq[i-1]) + '\n')
            x=np.random.rand()
            
            if x <= prob: #if the random number is smaller than our probability
                #print('higher chi-square is accepted')
                acceptance_ratio = acceptance_ratio + 1
                chisq[i] = new_chisq
                chain[i, :] = new_param
                
            else:
                #print('higher chi-square is not accepted')
                chisq[i] = chisq[i-1]
                chain[i, :] = chain[i-1, :]
                
        #if chi-square got small, we accept it        
        else:
            #print('lower chi-square')
            acceptance_ratio = acceptance_ratio + 1
            chisq[i] = new_chisq
            chain[i, :] = new_param
                          
    return chain, chisq, acceptance_ratio/max_steps

#Running the MCMC-------------------------------------------------------------------------------------------------------
params, cs, acceptance_ratio = mcmc(nstep, value_guess_normalized)
params = denormalize_many(params)
#printting the outputs to a file
#np.savetxt('params.gz' , params)
np.savetxt('/scratch/o/oscarh/aryanah/rbfi_output/params.gz' , params)

#Fourier Transform------------------------------------------------------------------------------------------------------
ps = np.zeros((nstep, param_length))
for i in range(param_length):
    ps[:, i] = np.abs(np.fft.fft(params[:, i]))**2
    
freqs = np.fft.rfftfreq(nstep)
idx = np.argsort(freqs)
#np.savetxt('ps.gz' , ps)
np.savetxt('/scratch/o/oscarh/aryanah/rbfi_output/ps.gz' , ps)

#MCMC output------------------------------------------------------------------------------------------------------------
mcmc_param= np.empty(param_length)


for i in range(param_length):
    mcmc_param[i] = np.mean(params[:,i]) #array of best parameters

#mcmc_param_denormalized = denormalize(np.asarray(mcmc_param)[np.newaxis])    

mcmc_param_dict = list_to_dict(mcmc_param, key_guess)    

mcmc_T = call_rbfi(rbfi, mcmc_param)
                 
#Printing the mcmc outputs---------------------------------------------------------------------------------------------
#printing the results to the txt file
#txt_2=open('mcmc_result.txt','w')
txt_2=open('/scratch/o/oscarh/aryanah/rbfi_output/mcmc_result.txt','w')
txt_2.write(repr(mcmc_param_dict) + '\n')
txt_2.write("Chi-squared of mcmc:"+ repr(chisquare(mcmc_T, model_e))+ '\n')
txt_2.write("Chi-squared of original guess:"+ repr(chisquare(T_guess, model_e))+ '\n')
txt_2.write("acceptance_ratio for %d Steps: " %nstep + repr(acceptance_ratio*100) +"%"+ '\n')
txt_2.write("The final Curve:" + '\n')
txt_2.write(repr(mcmc_T))
txt_2.close()

#Plotting the mcmc outputs-----------------------------------------------------------------------------------------------
fig2 = plt.figure()
plt.plot(z_e, model_e, label='EDGES Model')
plt.plot(z_e, T_guess, label='Original Guess')
plt.plot(z_e, mcmc_T, label="MCMC Result")
plt.title('Result of MCMC (%d Steps)'%nstep, fontsize=12)
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
#plt.savefig('mcmc_result.png')
plt.savefig('/scratch/o/oscarh/aryanah/rbfi_output/mcmc_result.png')
plt.show()

#Plotting the chi-square trend-------------------------------------------------------------------------------------------
fig3 = plt.figure()
#plt.plot(np.log(cs))
plt.semilogy(cs)
plt.xlabel('number of steps', fontsize=12)
plt.title ('Chi-Square Trend (%d Steps)'%nstep, fontsize=12)
plt.ylabel('Log of Chi-Square', fontsize=12)
#plt.savefig('chi-square.png')
plt.savefig('/scratch/o/oscarh/aryanah/rbfi_output/chi-square.png')
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
plt.savefig('/scratch/o/oscarh/aryanah/rbfi_output/parameters.png')
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
plt.savefig('/scratch/o/oscarh/aryanah/rbfi_output/fourier.png')
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
plt.savefig('/scratch/o/oscarh/aryanah/rbfi_output/corner_plots.png')
plt.show()
"""                 
#Calculating the residuals--------------------------------------------------------------------------------------------------------------
r_paper_model = mcmc_T - model_e
                 
#Plotting the residuals-----------------------------------------------------------------------------------------------------------------
fig5 = plt.figure()
plt.plot(z_e, r_paper_model, label= 'EDGES Model')
#plt.plot(z_e, r_data, label= 'EDGES Data')
plt.legend()
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.title('Residuals of MCMC Model after %d Steps'%nstep, fontsize=16)
#plt.savefig('residuals.png')
#plt.savefig('/scratch/o/oscarh/aryanah/rbfi_output/residuals.png')
plt.show()

#error-----------------------------------------------------------------------------------------------------------------------------------
error = r_paper_model/mcmc_T

#Plotting the error-----------------------------------------------------------------------------------------------------------------
fig6 = plt.figure()
plt.plot(z_e, error, label= 'error')
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('error', fontsize=12)
plt.title('Relative Error of EDGES Model and MCMC result', fontsize=12)
#plt.savefig('error.png')
#plt.savefig('/scratch/o/oscarh/aryanah/rbfi_output/error.png')
plt.show()
"""