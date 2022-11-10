import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numpy.random import normal
from math import ceil
from scipy.interpolate import CubicSpline
from scipy.interpolate import RBFInterpolator
from ares_params import ares_params, redshifts
import pickle

#producing the edges model on desired redshif range (from the function mentioned in the paper)
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
model_e = func(z_to_freq(z_e))

#converting the Data temperatures from k to mK 
k = 1000 
model_e = k * model_e

#temporary mfactor
model_e = model_e/2

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

def normalize(arr): # to normalize the parameter range
    #input: an array
    #output: normalized array
    
	op = np.empty(arr.shape)
	for i in range(arr.shape[1]):
		if ares_params[i][3] == "lin":
			op[:,i] = (arr[:,i] - ares_params[i][1])/(ares_params[i][2] - ares_params[i][1])
		elif ares_params[i][3] == "log":
			op[:,i] = (np.log10(arr[:,i]/ares_params[i][1]))/(np.log10(ares_params[i][2]/ares_params[i][1])) 
		else:
			raise ValueError("Invalid normalization type in ares_params")
	return op

def denormalize (arr): # to denormalize the parameter range
    #input: an array
    #output: denormalized array
    
	op = np.empty(arr.shape)
	for i in range(arr.shape[1]):
		if ares_params[i][3] == "lin":
			op[:,i] = arr[:,i] * (ares_params[i][2] - ares_params[i][1]) + ares_params[i][1]
		elif ares_params[i][3] == "log":
			op[:,i] = 10.0**(arr[:,i] * (np.log10(ares_params[i][2]) - np.log10(ares_params[i][1])) + np.log10(ares_params[i][1]))
		else:
			raise ValueError("Invalid normalization type in ares_params")
	return op

#RBFI--------------------------------------------------------------------------------------------------------------------------------
file = open("curve_rbfi.pickle","rb")
rbfi = pickle.load(file)

def call_rbfi(rbfi, params):
	return rbfi(np.append(redshifts[np.newaxis].T, np.tile(params,(redshifts.shape[0],1)), axis = 1))

"""
def call_rbfi_many(rbfi, params):
	# the same function but expecting a list of params
	# returns an array of curve values
	nrs = redshifts.shape[0] #stands for num redshifts
	nc = params.shape[0] #stands for num curves
	
	rs_stacked = np.tile(redshifts[np.newaxis].T, (nc,1))
	params_repeated = np.repeat(params, nrs, axis = 0)

	curves_approxed = rbfi(np.hstack((rs_stacked, params_repeated))) #this has dimensions [nrs*nc,1]
	return curves_approxed.reshape((nc,nrs))
"""


#MCMC inputs (the guess subscipt is related to the original guess)------------------------------------------------------------------
#start_guess = {'fstar': 0.12, 'fX': 0.12, 'fesc':0.12, 'clumping_factor':0.12}
start_guess = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_1_': 1E38, 'pop_rad_yield_2_': 1E4, 'clumping_factor': 1}
               #["pop_rad_yield_0_", 1E3, 1E6, "log"], ["pop_rad_yield_1_", 1E37, 5E40, "log"], ["pop_rad_yield_2_", 1E3, 5E6, "log"], ["clumping_factor", 0.05, 2, "lin"]



#resonable range for each parameter so ARES won't crash!
#ranges are normalized
low_bound = [0,0,0,0]
up_bound = [1,1,1,1]

#deviation of change for each param
dev = [0.005, 0.005, 0.005, 0.005]

param_length = len(start_guess)
                 
#converting start guess to two lists
value_guess, key_guess = dict_to_list(start_guess)
   
#normalizing the guess
value_guess_normalized = normalize(np.asarray(value_guess)[np.newaxis]) #normalizing the guess values -  converting to 2D array

nstep = 10 #number of steps to be taken in the MCMC

#Input guess from ARES---------------------------------------------------------------------------------------------------------------
T_guess = call_rbfi(rbfi, value_guess_normalized)

#Plotting data, paper model and original guess---------------------------------------------------------------------------------------

fig1 = plt.figure()
plt.plot(z_e, model_e, label='EDGES Model')
plt.plot(z_e, T_guess, label='Original Guess')
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
#plt.savefig('/scratch/o/oscarh/aryanah/MCMC_output/guess_model.png')
plt.savefig('guess_model.png')
plt.show()


#Defining the MCMC chain--------------------------------------------------------------------------------------------------------
def mcmc (max_steps, start_guess):
    
    #definig the chain
    chain = np.empty((max_steps, len(start_guess)))
    chain[0, :] = value_guess_normalized
     
    #defining the chi-square
    chisq = np.empty(max_steps)
    chisq[0] = chisquare(T_guess, model_e)
    
    #writing the details of chain in a txt file
    #txt_1 = open('mcmc_detail.txt','w')
    
    #the chain 
    for i in range(1, max_steps):
        #print('iteration number', i, 'of', max_steps)
        #txt_1.write('iteration number  '+ repr(i) + '  of  ' + repr(max_steps) + '\n')
        
        
        while (True): # to chack if  all the parameters are in the reasonable limit
            new_param = normal(chain[i-1, :], dev) #new random point in the parameter space
            result = check_limits(new_param, low_bound, up_bound)
            if result: # if all the parameters are in the reasonable limit, the new_param is accepted
                break
        
        #new_param = normal(chain[i-1, :], dev) #new random point in the parameter space
        #new_param_dict = list_to_dict(new_param, key_guess)
        
        T = call_rbfi(rbfi, new_param)
        
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

mcmc_param_denormalized = denormalize(np.asarray(mcmc_param)[np.newaxis])    

mcmc_param_dict = list_to_dict(mcmc_param_denormalized[0], key_guess)    
#mcmc_param_dict = list_to_dict(mcmc_param, key_guess)

mcmc_T = call_rbfi(rbfi, mcmc_param)
                 
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
txt_2.write("The final Curve:" + '\n')
txt_2.write(repr(mcmc_T))
txt_2.close()

#Plotting the mcmc outputs-----------------------------------------------------------------------------------------------
fig2 = plt.figure()
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
#r_data = mcmc_T - T_e 
                 
#Plotting the residuals-----------------------------------------------------------------------------------------------------------------
fig5 = plt.figure()
plt.plot(z_e, r_paper_model, label= 'EDGES Model')
#plt.plot(z_e, r_data, label= 'EDGES Data')
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