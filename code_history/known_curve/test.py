# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import ares
from math import ceil

# %%
#loading the EDGES data (the e subscipt is related to EDGES)
data_1 = pd.read_csv('/home/o/oscarh/aryanah/My-Project/data/data_1.csv')
freq_e = data_1.iloc[:,0] #frequency, MHz

#Changing the axis from frequency to redshift
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

# %%
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
    z = z[z<50]
    dTb = dTb[:len(z)]
    spline = CubicSpline(z, dTb)
    
    return spline(redshifts) 

def func_ares (m, z, d = 1E6): 
    #I can further change this function to include the best dx - 4*int(1E5) is the best number I found so far
    #m is the list of params 
    #z is the redshift range
    #y is the brightness temp
    m = np.array(m)
    T = call_ares(list_to_dict(m, key), z)
    derivs = np.zeros([len(z), len(m)])
    dpars = np.zeros(len(m))
    dpars = m/d 
    for i in range(len(m)):        
        pars_plus = np.array(m, copy=True, dtype = 'float64')
        pars_plus[i] = pars_plus[i] + dpars[i]
        
        pars_minus = np.array(m, copy=True, dtype = 'float64')
        pars_minus[i] = pars_minus[i] - dpars[i]
        
        A_plus = call_ares (list_to_dict(pars_plus, key), z)
        A_minus = call_ares (list_to_dict(pars_minus, key), z)
        A_m = (A_plus - A_minus)/(2*dpars[i])
        derivs[:, i] = A_m    
    return T, derivs

def cov_mat_calc(mat):
    dim = mat.shape[0]
    cov=(mat.T@mat)/dim
    return cov

def chisquare (pars, data, err): #returns the chi-square of two 21cm curves - err can be a number/array
    pred = call_ares(list_to_dict(pars, key), z_e)
    chisq = np.sum((pred-data)**2/err**2)
    return chisq

def draw_samples(covariance_matrix, nset):
    #normalizing the covariance matrix
    D = np.diag(np.diag(covariance_matrix)) #diagonal matrix of covariance matrix
    D_sqrt = np.sqrt(D)
    D_inv_sqrt = np.linalg.inv(D_sqrt)
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

#Defining the MCMC chain
def mcmc(fun_chisq, start_guess, covariance_matrix, data, err, nstep):
    samples = draw_samples(covariance_matrix, nstep)

    #definig the chain
    chain = np.empty((nstep, len(start_guess)))
    chain[0, :] = start_guess
     
    #defining the chi-square array
    chisq = np.zeros(nstep)
    chisq[0] = fun_chisq(start_guess, data, err)

    #defining the acceptance ratio
    acceptance_ratio = 0
            
    #the chain 
    for i in range(1, nstep):
        #print('iteration number', i, 'of', nstep) 
        new_param = samples[i, :] + chain[i-1, :]
        
        try:
            new_chisq =  fun_chisq(new_param, data, err)
        except:
            new_chisq = 1E7
      
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

# %%
#inputs 
dict_true = {'pop_rad_yield_0_': 4.0, 'pop_rad_yield_1_': 29.0, 'pop_rad_yield_2_': 5.0, 'clumping_factor': 1.7} 
m_true, key = dict_to_list(dict_true)
y_true = call_ares(dict_true, z_e)
y = y_true
model_e = y
m0 = [4.1, 29.5, 4.9, 1.8]

param_length = len(dict_true)
nstep = 500
err = 1

y_fit, derivs = func_ares(m_true, z_e)
dim = derivs.shape[0]
mycov = cov_mat_calc(derivs) 

# %%
#Running the MCMC
params, cs, acceptance_ratio = mcmc(chisquare, m0, mycov/2, model_e, err, nstep)
#np.savetxt('params.gz' , params)
np.savetxt('/scratch/o/oscarh/aryanah/output_1/params.gz' , params)

#MCMC output
mcmc_param= np.empty(param_length)
for i in range(param_length):
    mcmc_param[i] = np.mean(params[:,i]) #array of best parameters  
mcmc_param_dict = list_to_dict(mcmc_param, key)
mcmc_T = call_ares(mcmc_param_dict, z_e)

#txt = open('results.txt', 'w')
txt = open('/scratch/o/oscarh/aryanah/output_1/results.txt','w')
txt.write('True Parameters: ' + repr(m_true) + '\n')
txt.write('Starting Parameters: ' + repr(m0) + '\n')
#txt.write('LM Fitted Parameters: ' + repr(m_fit) + '\n')
txt.write('MCMC Fitted Parameters: ' + repr(mcmc_param) + '\n')

txt.write('\n' + "Chi-squared of original guess:"+ repr(chisquare(m0, model_e, err))+ '\n')
#txt.write("Chi-squared of lm result:"+ repr(chisquare(m_fit, model_e, err))+ '\n')
txt.write("Chi-squared of mcmc result:"+ repr(chisquare(mcmc_param, model_e, err))+ '\n')

txt.write("acceptance_ratio for %d Steps: " %nstep + repr(acceptance_ratio*100) +"%"+ '\n')
txt.write('\n' + 'Covariance Matrix: ' + repr(mycov) + '\n')
txt.close()

#Fourier Transform------------------------------------------------------------------------------------------------------
ps = np.zeros((nstep, param_length))
for i in range(param_length):
    ps[:, i] = np.abs(np.fft.fft(params[:, i]))**2
    
freqs = np.fft.rfftfreq(nstep)
idx = np.argsort(freqs)
#np.savetxt('ps.gz' , ps)
np.savetxt('/scratch/o/oscarh/aryanah/output_1/ps.gz' , ps)

#Plotting the mcmc outputs-----------------------------------------------------------------------------------------------
fig2 = plt.figure()
plt.plot(z_e, call_ares(list_to_dict(m0, key), z_e), label='Start Guess')
plt.plot(z_e, y_true, label='True Curve')
plt.plot(z_e, mcmc_T, label="MCMC Result")
plt.title('Result of MCMC (%d Steps)'%nstep, fontsize=12)
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
#plt.savefig('mcmc_result.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_1/mcmc_result.png')

#Plotting the chi-square trend-------------------------------------------------------------------------------------------
fig3 = plt.figure()
#plt.plot(np.log(cs))
plt.semilogy(cs)
plt.xlabel('number of steps', fontsize=12)
plt.title ('Chi-Square Trend (%d Steps)'%nstep, fontsize=12)
plt.ylabel('Chi-Square', fontsize=12)
#plt.savefig('chi-square.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_1/chi-square.png')

#Plotting the parameters trend--------------------------------------------------------------------------------------------
fig4, ax_list = plt.subplots(ceil(param_length/2), 2, figsize=(13,10))
fig4.suptitle('Chain (%d Steps)'%nstep, fontsize=16)
if((param_length % 2) == 0):
    for i in range(ceil(param_length/2)):
        for j in range(2):
            ax_list[i, j].plot(params[:, i*2+ j])
            ax_list[i, j].set_ylabel(repr(key[i*2+ j]), fontsize=16)
            ax_list[i, j].set_xlabel('number of steps', fontsize=12)

else:
    for i in range(ceil(param_length/2)):
        for j in range(2):
            if(j == 1 and i == (ceil(param_length/2)-1)):
                break
            ax_list[i, j].plot(params[:, i*2+ j])
            ax_list[i, j].set_ylabel(repr(key[i*2+ j]), fontsize=16)
            ax_list[i, j].set_xlabel('number of steps', fontsize=12)
            

plt.tight_layout()
#plt.savefig('parameters.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_1/parameters.png') 

#Plotting the Fourier Transform-------------------------------------------------------------------------------------------
fig4, ax_list = plt.subplots(ceil(param_length/2), 2, figsize=(13,10))
fig4.suptitle('Power Spectrum of the Chain (%d Steps)'%nstep, fontsize=16)
if((param_length % 2) == 0):
    for i in range(ceil(param_length/2)):
        for j in range(2):
            ax_list[i, j].loglog(freqs[idx], ps[idx, i*2+ j])
            #plt.yscale("log")
            ax_list[i, j].set_ylabel(repr(key[i*2+ j]), fontsize=16)
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
            ax_list[i, j].set_ylabel(repr(key[i*2+ j]), fontsize=16)
            ax_list[i, j].set_xlabel('frequency', fontsize=12)
            #ax_list[i, j].set_ylim(0, 1E2)
            #ax_list[i, j].set_xlim(0.2, 0.5)
            
            

plt.tight_layout()
#plt.savefig('fourier.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_1/fourier.png')

#Corner Plots--------------------------------------------------------------------------------------------------------------
fig, ax_list = plt.subplots(3, 2, figsize=(12,10))
fig.suptitle('Corner plots of the chain', fontsize=16)

ax_list[0, 0].plot(params[:, 0], params[:, 1], linestyle = "", marker=".")
ax_list[0, 0].set_ylabel(repr(key[1]), fontsize=12)
ax_list[0, 0].set_xlabel(repr(key[0]), fontsize=12)

ax_list[0, 1].plot(params[:, 0], params[:, 2], linestyle = "", marker=".")
ax_list[0, 1].set_ylabel(repr(key[2]), fontsize=12)
ax_list[0, 1].set_xlabel(repr(key[0]), fontsize=12)
    
ax_list[1, 0].plot(params[:, 0], params[:, 3], linestyle = "", marker=".")
ax_list[1, 0].set_ylabel(repr(key[3]), fontsize=12)
ax_list[1, 0].set_xlabel(repr(key[0]), fontsize=12)

ax_list[1, 1].plot(params[:, 1], params[:, 2], linestyle = "", marker=".")
ax_list[1, 1].set_ylabel(repr(key[2]), fontsize=12)
ax_list[1, 1].set_xlabel(repr(key[1]), fontsize=12)

ax_list[2, 0].plot(params[:, 1], params[:, 3], linestyle = "", marker=".")
ax_list[2, 0].set_ylabel(repr(key[3]), fontsize=12)
ax_list[2, 0].set_xlabel(repr(key[1]), fontsize=12)

ax_list[2, 1].plot(params[:, 2], params[:, 3], linestyle = "", marker=".")
ax_list[2, 1].set_ylabel(repr(key[3]), fontsize=12)
ax_list[2, 1].set_xlabel(repr(key[2]), fontsize=12)

plt.tight_layout()
#plt.savefig('corner_plots.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_1/corner_plots.png')