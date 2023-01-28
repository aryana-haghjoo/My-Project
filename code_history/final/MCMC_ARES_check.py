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

def call_ares (params, redshifts): 
    #params should be a dictionary
    value, key = dict_to_list(params)
    
    value_denormalized = np.array(value, dtype='float64')
    value_denormalized[0] = 10** (value[0])
    value_denormalized[1] = 10** (value[1])
    value_denormalized[2] = 10** (value[2])
    params_denormalized = list_to_dict(value_denormalized, key)
    
    #print(params_denormalized)
    #running ares
    sim = ares.simulations.Global21cm(**params_denormalized, verbose=False, progress_bar=False)
    sim.run()
    z = sim.history['z'][::-1]
    dTb = sim.history['dTb'][::-1]
    z = z[z<50]
    dTb = dTb[:len(z)]
    spline = CubicSpline(z, dTb)
    
    return spline(redshifts) 

def check_limits_fix(m):
    m_new = np.array(m, copy=True, dtype = 'float64')
    
    #pop_rad_yield_0_: 1E2 - 1E10 
    if m[0]< 1E2:
        m_new[0] = 2
        
    if m[0]>1E10:
        m_new[0] = 10
    
    #pop_rade_yield_1_: 0 - 1E41
    if m[1]<0:
        m_new[1] = 0

    if m[1]> 41:
        m_new[1] = 41
        
    #pop_rade_yield_2_: 0 - 1E6     
    if m[2]< 0:
        m_new[2] = 0
        
    if m[2]> 6:
        m_new[2] = 6
    
    #clumping_factor: 0-15
    if m[3] < 0:
        m_new[0] = 0
        
    if m[3] > 15:
        m[3] = 15
        
    return m_new

def func_ares (m, z, d = 4*int(1E5)): 
    #I can further change this function to include the best dx
    #m is the list of params 
    #z is the redshift range
    #y is the brightness temp
    m = np.array(m)
    m_prime = check_limits_fix(m)
    T = call_ares (list_to_dict(m_prime, key), z)
    derivs = np.zeros([len(z), len(m)])
    dpars = np.zeros(len(m))
    dpars = m/d 
    for i in range(len(m)):        
        pars_plus = np.array(m, copy=True, dtype = 'float64')
        pars_plus[i] = pars_plus[i] + dpars[i]
        
        pars_minus = np.array(m, copy=True, dtype = 'float64')
        pars_minus[i] = pars_minus[i] - dpars[i]
        
        pars_plus_prime = check_limits_fix(pars_plus)
        A_plus = call_ares (list_to_dict(pars_plus_prime, key), z)
        
        pars_minus_prime = check_limits_fix(pars_minus)
        A_minus = call_ares (list_to_dict(pars_minus_prime, key), z)
        A_m = (A_plus - A_minus)/(2*dpars[i])
        derivs[:, i] = A_m    
    return T, derivs

"""
def check_limits_mcmc(m):
#pop_rad_yield_0_: 1E2 - 1E10 
#pop_rade_yield_1_: 0 - 1E41
#pop_rade_yield_2_: 0 - 1E6 
#clumping_factor: 0-15

    m_new = np.array(m, copy=True, dtype = 'float64')
    if m[1] > 96:
        value_1 = False
    else:
        value_1 = True
        
    if m[2] > 31:
        value_2 = False
    else:
        value_2 = True
        
    if m[3] > 28:
        value_3 = False
    else:
        value_3 = True
        
    return value_1&value_2&value_3
"""

def update_lamda(lamda, success):
    if success:
        lamda = lamda/1.5
        if lamda<0.5:
            lamda=0
    else:
        if lamda==0:
            lamda=1
        else:
            #lamda = lamda*1.5**2
            lamda = lamda*2
    return lamda

def get_matrices(m, fun, x, y, Ninv=None):
    model, derivs = fun(m, x)
    r = y-model
    if Ninv is None:
        lhs = derivs.T@derivs
        rhs = derivs.T@r
        chisq = np.sum(r**2)
    else: 
        lhs = derivs.T@Ninv@derivs
        rhs = derivs.T@(Ninv@r)
        chisq = r.T@Ninv@r
    return chisq, lhs, rhs

def linv(mat, lamda):
    mat = mat + lamda*np.diag(np.diag(mat))
    return np.linalg.pinv(mat)

def chisquare (pars, data, err): #returns the chi-square of two values - err can be a number/array
    pred = call_ares(list_to_dict(pars, key), z_e)
    chisq = np.sum((pred-data)**2/err**2)
    return chisq

def draw_samples(cov, n):
    m=cov.shape[0]
    mat=np.random.randn(m,n)
    L=np.linalg.cholesky(cov)
    return (L@mat).T

##levenberg-Marquardt Fitter -----------------------------------------------------------------------------------------------------------
def LM(m, fun, x, y, Ninv=None, niter=10, chitol= 1): 
    lamda=0
    #m = check_limits_lm(m)
    chisq, lhs, rhs = get_matrices(m, fun, x, y, Ninv)
    
    for i in range(niter):
        lhs_inv = linv(lhs, lamda)
        dm = lhs_inv@rhs
        m_new = m+dm
        #print(dm)
        #m_new = check_limits_lm(m+dm)
        chisq_new, lhs_new, rhs_new = get_matrices(m_new, fun, x, y, Ninv)

        #try:
         #   chisq_new, lhs_new, rhs_new = get_matrices(m+dm, fun, x, y, Ninv)
        #except:
            
        if chisq_new<chisq:  
            if lamda==0:
                if (np.abs(chisq-chisq_new)<chitol):
                    print('Converged after ', i, ' iterations of LM')
                    return m_new
            chisq = chisq_new
            lhs = lhs_new
            rhs = rhs_new
            m = m_new
            lamda = update_lamda(lamda,True)
            
        else:
            lamda = update_lamda(lamda, False)
        #print('on iteration ', i, ' chisq is ', chisq, ' with step ', dm, ' and lamda ', lamda)
        print('\n', 'on iteration ', i, ' chisq is ', chisq, ' and lamda is ', lamda)
        print('step ', dm)
        print('new params ', m_new)
        
    return m

#Defining the MCMC chain--------------------------------------------------------------------------------------------------------
def mcmc(fun, start_guess, data, err, samples, nstep):    
    #definig the chain
    chain = np.empty((nstep, len(start_guess)))
    chain[0, :] = start_guess
     
    #defining the chi-square
    chisq = np.zeros(nstep)
    chisq[0] = fun(start_guess, data, err)
    acceptance_ratio = 0
            
    #the chain 
    for i in range(1, nstep):
        print('iteration number', i, 'of', nstep) 
        #while (True):
         #   new_param = np.random.normal(chain[i-1, :], dev)
         #   result = check_limits_mcmc(new_param)
         #   if result:
         #       break
        #call the draw samples function here
        
        new_param = samples[i, :] + start_guess
        try:
            new_chisq =  fun(new_param, data, err)
        except:
            new_chisq = 1E7
        """
        if not (check_limits_mcmc(new_param)):
            print('out of bound')
            new_chisq = 1E7
        else:
            new_chisq = fun(new_param, data, err)
        """    
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
                print('new param not accepted')
                chisq[i] = chisq[i-1]
                chain[i, :] = chain[i-1, :]          
    return chain, chisq, acceptance_ratio/nstep

#LM inputs ---------------------------------------------------------------------------------------------------------------------------
dict_true = {'pop_rad_yield_0_': 4.03, 'pop_rad_yield_1_': 20.0, 'pop_rad_yield_2_': 3.0, 'clumping_factor': 0.7} 
m_true, key = dict_to_list(dict_true)
y_true = call_ares(dict_true, z_e)
y = y_true*(1+ np.random.randn(len(y_true))*0.01)
#m0 = m_true + np.random.randn(len(m_true))*0.1
m0 = [4.05, 20.5, 2.9, 0.72]
model_e = y

#MCMC inputs --------------------------------------------------------------------------------------------------------------------------
param_length = len(dict_true)
nstep = 10
err = 1

#Running the LM------------------------------------------------------------------------------------------------------------------------
m_fit = LM (m0, func_ares, z_e, model_e, niter=20)
y_fit, cov_mat = func_ares(m_fit, z_e)
print(cov_mat[0:5, 0:5])
dim = cov_mat.shape[0]
np.savetxt('cov_mat.gz', cov_mat)
mycov=(cov_mat.T@cov_mat)/dim
print(mycov)
mycov = mycov+ 1E-3

#Running the MCMC----------------------------------------------------------------------------------------------------------------------
samples = draw_samples(mycov, nstep)
np.savetxt('samples.gz', samples)
#np.savetxt('/scratch/o/oscarh/aryanah/output_1/samples.gz' , samples)
params, cs, acceptance_ratio = mcmc(chisquare, m_fit, model_e, err, samples, nstep)
np.savetxt('params.gz' , params)
#np.savetxt('chi-surf.gz', chi_surf)
#np.savetxt('/scratch/o/oscarh/aryanah/output_1/params.gz' , params)

"""
#chi-square surface-----------------------------------------------------------------------------------------------------------
chi_square = np.zeros(samples.shape[0])
for i in range(samples.shape[0]):
    chi_square[i] = chisquare(samples[i, :], y, 1)
    
np.savetxt('chisquare.txt' , chi_square)
plt.plot(chi_square)
plt.savefig('chi.png')
"""
#MCMC output------------------------------------------------------------------------------------------------------------
mcmc_param= np.empty(param_length)
for i in range(param_length):
    mcmc_param[i] = np.mean(params[:,i]) #array of best parameters  
mcmc_param_dict = list_to_dict(mcmc_param, key)
mcmc_T = call_ares(mcmc_param_dict, z_e)

txt = open('results.txt', 'w')
#txt = open('/scratch/o/oscarh/aryanah/output_1/results.txt','w')
txt.write('True Parameters: ' + repr(m_true) + '\n')
txt.write('Starting Parameters: ' + repr(m0) + '\n')
txt.write('LM Fitted Parameters: ' + repr(m_fit) + '\n')
txt.write('MCMC Fitted Parameters: ' + repr(mcmc_param) + '\n')

txt.write('\n' + "Chi-squared of original guess:"+ repr(chisquare(m0, model_e, err))+ '\n')
txt.write("Chi-squared of lm result:"+ repr(chisquare(m_fit, model_e, err))+ '\n')
txt.write("Chi-squared of mcmc result:"+ repr(chisquare(mcmc_param, model_e, err))+ '\n')

txt.write('\n' + 'Covariance Matrix: ' + repr(mycov) + '\n')
txt.write("acceptance_ratio for %d Steps: " %nstep + repr(acceptance_ratio*100) +"%"+ '\n')
txt.write('y: '+ repr(y)+ '\n')
#txt.write('chi-surf: ' + repr(chi_surf))
txt.close()
"""
#Fourier Transform------------------------------------------------------------------------------------------------------
ps = np.zeros((nstep, param_length))
for i in range(param_length):
    ps[:, i] = np.abs(np.fft.fft(params[:, i]))**2
    
freqs = np.fft.rfftfreq(nstep)
idx = np.argsort(freqs)
np.savetxt('ps.gz' , ps)
#np.savetxt('/scratch/o/oscarh/aryanah/output_1/ps.gz' , ps)

#Plotting the mcmc outputs-----------------------------------------------------------------------------------------------
fig2 = plt.figure()
#plt.plot(z_e, model_e, label='Start Guess')
plt.plot(z_e, y_true, label='True Curve')
plt.plot(z_e, mcmc_T, label="MCMC Result")
plt.title('Result of MCMC (%d Steps)'%nstep, fontsize=12)
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
plt.savefig('mcmc_result.png')
#plt.savefig('/scratch/o/oscarh/aryanah/output_1/mcmc_result.png')
plt.show()

#Plotting the chi-square trend-------------------------------------------------------------------------------------------
fig3 = plt.figure()
#plt.plot(np.log(cs))
plt.semilogy(cs)
plt.xlabel('number of steps', fontsize=12)
plt.title ('Chi-Square Trend (%d Steps)'%nstep, fontsize=12)
plt.ylabel('Chi-Square', fontsize=12)
plt.savefig('chi-square.png')
#plt.savefig('/scratch/o/oscarh/aryanah/output_1/chi-square.png')
plt.show()

#Plotting the parameters trend--------------------------------------------------------------------------------------------
fig4, ax_list = plt.subplots(ceil(param_length/2), 2, figsize=(13,10))
fig4.suptitle('Chain (%d Steps)'%nstep, fontsize=16)
#fig4.suptitle('Displaying The Trend of Parameters', fontsize=16)
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
plt.savefig('parameters.png')
#plt.savefig('/scratch/o/oscarh/aryanah/output_1/parameters.png')
plt.show()    

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
plt.savefig('fourier.png')
#plt.savefig('/scratch/o/oscarh/aryanah/output_1/fourier.png')
plt.show()    

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
plt.savefig('corner_plots.png')
#plt.savefig('/scratch/o/oscarh/aryanah/output_1/corner_plots.png')
plt.show()
"""