#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import ares
import signal
from math import ceil

#%%
#loading the EDGES data (the e subscipt is related to EDGES)
data_1 = pd.read_csv('/home/o/oscarh/aryanah/My-Project/data/data_1.csv')
#data_1 = pd.read_csv('/home/aryana/GitHub/My-Project/data/data_1.csv')

freq_e = data_1.iloc[:,0] #frequency, MHz
model_e = data_1.iloc[:, 5] #model, mK

#converting the data from mK to K
model_e = model_e*1000
model_e = model_e/2

#Changing the axis from frequency to redshift
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

#%%
def dict_to_list(d): #converts dictionary to two lists (key and value)
    #d must be a dictionary containing the value of parameters and their names
    key = list(d.keys())
    value = list(d.values())
    return value, key

def list_to_dict(value, key): #converts two lists (key and value) to a dictionary
    #value is a list of parameters' values
    #key is a list of parameters' names
    return dict(zip(key, value))

def call_ares (params, redshifts): 
    #params should be a dictionary
    sim = ares.simulations.Global21cm(**params, verbose=False, progress_bar=False)
    sim.run()
    z = sim.history['z'][::-1]
    dTb = sim.history['dTb'][::-1]
    z = z[z<50]
    dTb = dTb[:len(z)]
    spline = CubicSpline(z, dTb)
    
    return spline(redshifts) 

def draw_samples(covariance_matrix, nset):
    #normalizing the covariance matrix
    D = np.diag(np.diag(covariance_matrix)) #diagonal matrix of covariance matrix
    D_sqrt = np.sqrt(D)
    D_inv_sqrt = np.linalg.pinv(D_sqrt)
    covariance_matrix_normalized = D_inv_sqrt @ covariance_matrix @ D_inv_sqrt #normalized covariance matrix

    e,v = np.linalg.eigh(covariance_matrix_normalized)
    e[e<0]=0 #make sure we don't have any negative eigenvalues due to roundoff
    n = len(e)

    #make gaussian random variables
    g=np.random.randn(n, nset)

    #now scale them by the square root of the eigenvalues
    rte=np.sqrt(e)
    for i in range(nset):
        g[:,i]=g[:,i]*rte

    #and rotate back into the original space
    samples = (v@g).T
    samples_denormalized = samples @ D_sqrt
    return samples_denormalized


def chisquare(pars, data, Ninv):
    def timeout_handler(signum, frame):
        raise TimeoutError("chisquare function timed out")

    try:
        # set a timer for 60 seconds
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)

        pred = call_ares(list_to_dict(pars, key), z_e)
        r = data - pred
        chisq = r.T @ Ninv @ r

        # reset the timer
        signal.alarm(0)

    except TimeoutError:
        # handle the timeout exception here
        chisq = 1E10

    return chisq

#Defining the MCMC chain
#def mcmc(fun_chisq, start_guess, covariance_matrix, data, Ninv, nstep):
def mcmc(fun_chisq, start_guess, samples, data, Ninv, nstep):
    #samples = draw_samples(covariance_matrix, nstep)

    #definig the chain
    chain = np.empty((nstep, len(start_guess)))
    chain[0, :] = samples[0, :] + start_guess
     
    #defining the chi-square array
    chisq = np.zeros(nstep)
    chisq[0] = fun_chisq(chain[0, :], data, Ninv)

    #defining the acceptance ratio
    acceptance_ratio = 0
            
    #the chain 
    for i in range(1, nstep):
        #print('iteration number', i, 'of', nstep) 
        new_param = samples[i, :] + chain[i-1, :]
        new_chisq =  fun_chisq(new_param, data, Ninv)
        if new_chisq <= chisq[i-1]:
            acceptance_ratio = acceptance_ratio + 1
            chisq[i] = new_chisq
            chain[i, :] = new_param 
        else :
            betta = 1
            if np.random.rand(1)<betta*(np.exp(-0.5*(new_chisq-chisq[i-1]))):
                acceptance_ratio = acceptance_ratio + 1
                chisq[i] = new_chisq
                chain[i, :] = new_param
            else:
                chisq[i] = chisq[i-1]
                chain[i, :] = chain[i-1, :]          
    return chain, chisq, acceptance_ratio/nstep

#%%
params_dict = {'pop_rad_yield_0_': 4.54933009e+03, 'pop_rad_yield_2_': 2.47592394e+03, 'fesc': 3.70100011e-01, 'fX': 1.36397790e-01}
#params_dict = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_2_': 1E3, 'fesc': 0.1, 'fX': 0.1} 

value, key = dict_to_list(params_dict)
value = np.array(value, copy=True, dtype = 'float64')
param_length = len(value)
err = 1E1 #mk
Ninv = ((err)**(-2))*np.eye(len(z_e))
nstep = 10000
mycovinv = np.array([[ 3.24667471e+04,  5.68112552e+06, -8.51686688e+02,
         5.65856615e-01],
       [ 5.68112552e+06,  2.43358043e+10, -3.63874836e+06,
         5.64163649e+00],
       [-8.51686688e+02, -3.63874836e+06,  5.44075269e+02,
        -9.80887059e-04],
       [ 5.65856615e-01,  5.64163649e+00, -9.80887059e-04,
         3.92215290e-05]])

samples = draw_samples(mycovinv/(1000000), nstep)

#%%
#Running the MCMC
params, cs, acceptance_ratio = mcmc(chisquare, value, samples, model_e, Ninv, nstep)
#np.savetxt('params.gz' , params)
np.savetxt('/scratch/o/oscarh/aryanah/mcmc/params.gz' , params)

#np.savetxt('csq.gz' , cs)
np.savetxt('/scratch/o/oscarh/aryanah/mcmc/csq.gz' , cs)

#MCMC output
mcmc_param= np.empty(param_length)
for i in range(param_length):
    mcmc_param[i] = np.mean(params[:,i]) #array of best parameters  
#%%
#txt = open('results.txt', 'w')
txt = open('/scratch/o/oscarh/aryanah/mcmc/results.txt','w')
txt.write('Starting Parameters: ' + repr(value) + '\n')
txt.write('MCMC Fitted Parameters: ' + repr(mcmc_param) + '\n')
txt.write("Chi-squared of original guess:"+ repr(chisquare(value, model_e, Ninv))+ '\n')
txt.write("Chi-squared of MCMC result:"+ repr(chisquare(mcmc_param, model_e, Ninv))+ '\n')
txt.write("acceptance_ratio for %d Steps: " %nstep + repr(acceptance_ratio*100) +"%"+ '\n')
txt.close()

mcmc_T = call_ares(list_to_dict(mcmc_param, key), z_e) #best fit curve

# %%--------------------------------------------------------------------------------------------------------------------------
fig1 = plt.figure()
plt.plot(z_e, model_e, label = 'EDGES')
plt.plot(z_e, mcmc_T, label = 'MCMC')
plt.plot(z_e, call_ares(params_dict, z_e), label = 'Initial Guess')
plt.legend()
plt.title('Result of MCMC (%d Steps)'%nstep, fontsize=12)
#plt.savefig('mcmc_result.png')
plt.savefig('/scratch/o/oscarh/aryanah/mcmc/mcmc_result.png')

# %%------------------------------------------------------------------------------------------------------------------------------
#Plotting the chi-square trend
fig2 = plt.figure()
plt.semilogy(cs)
plt.xlabel('number of steps', fontsize=12)
plt.title ('Chi-Square Trend (%d Steps)'%nstep, fontsize=12)
plt.ylabel('Chi-Square', fontsize=12)
#plt.savefig('chi-square.png')
plt.savefig('/scratch/o/oscarh/aryanah/mcmc/chi_square.png')

# %%---------------------------------------------------------------------------------------------------------------
fig3, ax_list = plt.subplots(ceil(param_length/2), 2, figsize=(13,10))
fig3.suptitle('Chain (%d Steps)'%nstep, fontsize=16)
#fig4.suptitle('Displaying The Trend of Parameters', fontsize=16)
if((param_length % 2) == 0):
    for i in range(ceil(param_length/2)):
        for j in range(2):
            ax_list[i, j].plot(params[:, i*2+ j])
            #ax_list[i, j].set_ylabel(repr(key[i*2+ j]), fontsize=16)
            ax_list[i, j].set_ylabel('param number %d'%(i*2+ j), fontsize=16)
            ax_list[i, j].set_xlabel('number of steps', fontsize=12)

else:
    for i in range(ceil(param_length/2)):
        for j in range(2):
            if(j == 1 and i == (ceil(param_length/2)-1)):
                break
            ax_list[i, j].plot(params[:, i*2+ j])
            #ax_list[i, j].set_ylabel(repr(key[i*2+ j]), fontsize=16)
            ax_list[i, j].set_ylabel('param number %d'%(i*2+ j), fontsize=16)
            ax_list[i, j].set_xlabel('number of steps', fontsize=12)
            
#%%
plt.tight_layout()
#plt.savefig('parameters.png')
plt.savefig('/scratch/o/oscarh/aryanah/mcmc/parameters.png') 

#Fourier Transform
ps = np.zeros((nstep, param_length))
for i in range(param_length):
    ps[:, i] = np.abs(np.fft.fft(params[:, i]))**2
    
freqs = np.fft.rfftfreq(nstep)
idx = np.argsort(freqs)

# %%
fig4, ax_list = plt.subplots(ceil(param_length/2), 2, figsize=(13,10))
fig4.suptitle('Power Spectrum of the Chain (%d Steps)'%nstep, fontsize=16)
if((param_length % 2) == 0):
    for i in range(ceil(param_length/2)):
        for j in range(2):
            ax_list[i, j].loglog(freqs[idx], ps[idx, i*2+ j])
            #plt.yscale("log")
            #ax_list[i, j].set_ylabel(repr(key[i*2+ j]), fontsize=16)
            ax_list[i, j].set_ylabel('param number %d'%(i*2+ j), fontsize=16)
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
            ax_list[i, j].set_ylabel('param number %d'%(i*2+ j), fontsize=16)
            ax_list[i, j].set_xlabel('frequency', fontsize=12)
            #ax_list[i, j].set_ylim(0, 1E2)
            #ax_list[i, j].set_xlim(0.2, 0.5)
            
            

plt.tight_layout()
#plt.savefig('fourier.png')
plt.savefig('/scratch/o/oscarh/aryanah/mcmc/fourier.png')

# %%----------------------------------------------------------------------------------------------------------------------
params_cut = params[1000:, :]
#params_cut = np.copy(params)
fig5, ax_list = plt.subplots(3, 2, figsize=(10,10))
fig5.suptitle('Corner plots of the chain', fontsize=16)

ax_list[0, 0].plot(params_cut[:, 0], params_cut[:, 1], linestyle = "", marker=".")
ax_list[0, 0].set_ylabel('param 1', fontsize=12)
ax_list[0, 0].set_xlabel('param 0', fontsize=12)

ax_list[0, 1].plot(params_cut[:, 0], params_cut[:, 2], linestyle = "", marker=".")
ax_list[0, 1].set_ylabel('param 2', fontsize=12)
ax_list[0, 1].set_xlabel('param 0', fontsize=12)
    
ax_list[1, 0].plot(params_cut[:, 0], params_cut[:, 3], linestyle = "", marker=".")
ax_list[1, 0].set_ylabel('param 3', fontsize=12)
ax_list[1, 0].set_xlabel('param 0', fontsize=12)

ax_list[1, 1].plot(params_cut[:, 1], params_cut[:, 0], linestyle = "", marker=".")
ax_list[1, 1].set_ylabel('param 2', fontsize=12)
ax_list[1, 1].set_xlabel('param 1', fontsize=12)

ax_list[2, 0].plot(params_cut[:, 1], params_cut[:, 3], linestyle = "", marker=".")
ax_list[2, 0].set_ylabel('param 3', fontsize=12)
ax_list[2, 0].set_xlabel('param 1', fontsize=12)

ax_list[2, 1].plot(params_cut[:, 2], params_cut[:, 3], linestyle = "", marker=".")
ax_list[2, 1].set_ylabel('param 3', fontsize=12)
ax_list[2, 1].set_xlabel('param 2', fontsize=12)

plt.tight_layout()
#plt.savefig('corner_plots.png')
plt.savefig('/scratch/o/oscarh/aryanah/mcmc/corner_plots.png')