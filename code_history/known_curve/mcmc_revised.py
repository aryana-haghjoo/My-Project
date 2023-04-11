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
#data_1 = pd.read_csv('/home/aryana/GitHub/My-Project/data/data_1.csv')

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
    
    sim = ares.simulations.Global21cm(**params, verbose=False, progress_bar=False)
    sim.run()
    z = sim.history['z'][::-1]
    dTb = sim.history['dTb'][::-1]
    z = z[z<50]
    dTb = dTb[:len(z)]
    spline = CubicSpline(z, dTb)
    
    return spline(redshifts) 

"""
def ares_deriv (m, z, d = 1E-2): 
    #I can further change this function to include the best dx - 4*int(1E5) is the best number I found so far
    #m is the list of params 
    #z is the redshift range
    #y is the brightness temp
    m = np.array(m)
    T = call_ares(list_to_dict(m, key), z)
    derivs = np.zeros([len(z), len(m)])
    dpars = d * m
    dpars = np.array(dpars, copy=True, dtype = 'float64')
    
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

def update_lamda(lamda, success):
    if success:
        lamda = lamda/1.5
        if lamda<0.5:
            lamda=0
    else:
        if lamda==0:
            lamda=1
        else:
            lamda = lamda*1.5**2
            #lamda = lamda*2
    return lamda

def get_matrices(m, fun, x, y, Ninv):
    model, derivs = fun(m, x)
    r = y-model
    lhs = derivs.T@Ninv@derivs
    rhs = derivs.T@(Ninv@r)
    chisq = r.T@Ninv@r
    return chisq, lhs, rhs

def linv(mat, lamda):
    mat = mat + lamda*np.diag(np.diag(mat))
    return np.linalg.inv(mat)

def LM(m, fun, x, y, Ninv, niter=10, chitol= 1): 
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
        #print('step ', dm)
        #print('new params ', m_new)
                
    return m
"""

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

def chisquare (pars, data, Ninv): #returns the chi-square of two 21cm curves - err can be a number/array   
    try:
        pred = call_ares(list_to_dict(pars, key), z_e)
        r = data-pred
        chisq = r.T@Ninv@r
    except:
        chisq = 1E10
    return chisq

#Defining the MCMC chain
def mcmc(fun_chisq, start_guess, covariance_matrix, data, Ninv, nstep):
    samples = draw_samples(covariance_matrix, nstep)

    #definig the chain
    chain = np.empty((nstep, len(start_guess)))
    chain[0, :] = start_guess
     
    #defining the chi-square array
    chisq = np.zeros(nstep)
    chisq[0] = fun_chisq(start_guess, data, Ninv)

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

# %%
dict_true = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_2_': 1E5, 'clumping_factor': 1.7, 'fX': 0.1} 

m_true, key = dict_to_list(dict_true)
m_true = np.array(m_true, copy=True, dtype = 'float64')
y_true = call_ares(dict_true, z_e)
m_0 = [10**(3.9), 10**(5.2), 1.5, 0.3]
err = 1E-3
Ninv = ((err)**(-2))*np.eye(len(z_e))

# %%
#m_fit = LM (m_0, ares_deriv, z_e, y_true, niter=20, Ninv = Ninv)
#chisq_f, lhs_f, rhs_f = get_matrices(m_fit, ares_deriv, z_e, y_true, Ninv)
#mycov = lhs_f
#y_fit = call_ares(list_to_dict(m_fit, key), z_e)

# %%
#error bars
#mycovinv= np.linalg.inv(mycov)
#np.sqrt(np.diag(mycovinv))
mycovinv = np.array([[ 6.42193835e-03, -3.35083949e-01, -1.68150631e-05,
        -5.32518724e-07],
       [-3.35083949e-01,  3.02166856e+01,  1.28123050e-03,
         3.09760274e-05],
       [-1.68150631e-05,  1.28123050e-03,  6.01477911e-08,
         1.74641455e-09],
       [-5.32518724e-07,  3.09760274e-05,  1.74641455e-09,
         6.43899943e-11]])

# %%
#MCMC inputs 
param_length = len(m_true)
nstep = 10000

# %%
#Running the MCMC
params, cs, acceptance_ratio = mcmc(chisquare, m_0, mycovinv, y_true, Ninv, nstep)
#np.savetxt('params.gz' , params)
np.savetxt('/scratch/o/oscarh/aryanah/output_2/params.gz' , params)

#np.savetxt('csq.gz' , cs)
np.savetxt('/scratch/o/oscarh/aryanah/output_2/csq.gz' , cs)

#%%
#MCMC output
mcmc_param= np.empty(param_length)
for i in range(param_length):
    mcmc_param[i] = np.mean(params[:,i]) #array of best parameters  

#txt = open('results.txt', 'w')
txt = open('/scratch/o/oscarh/aryanah/output_2/results.txt','w')
txt.write('True Parameters: ' + repr(m_true) + '\n')
txt.write('Starting Parameters: ' + repr(m_0) + '\n')
txt.write('MCMC Fitted Parameters: ' + repr(mcmc_param) + '\n')
txt.write('\n' + "Chi-squared of original guess:"+ repr(chisquare(m_0, y_true, Ninv))+ '\n')
txt.write("Chi-squared of MCMC result:"+ repr(chisquare(mcmc_param, y_true, Ninv))+ '\n')
txt.write("acceptance_ratio for %d Steps: " %nstep + repr(acceptance_ratio*100) +"%"+ '\n')
txt.close()

mcmc_T = call_ares(list_to_dict(mcmc_param, key), z_e) #best fit curve

# %%--------------------------------------------------------------------------------------------------------------------------
fig1 = plt.figure()
plt.plot(z_e, y_true, label = 'True')
plt.plot(z_e, mcmc_T, label = 'MCMC')
plt.plot(z_e, call_ares(list_to_dict(m_0, key), z_e), label = 'Initial Guess')
plt.legend()
plt.title('Result of MCMC (%d Steps)'%nstep, fontsize=12)
#plt.savefig('mcmc_result.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_2/mcmc_result.png')

# %%------------------------------------------------------------------------------------------------------------------------------
#Plotting the chi-square trend
fig2 = plt.figure()
plt.semilogy(cs)
plt.xlabel('number of steps', fontsize=12)
plt.title ('Chi-Square Trend (%d Steps)'%nstep, fontsize=12)
plt.ylabel('Chi-Square', fontsize=12)
#plt.savefig('chi-square.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_2/chi-square.png')

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
            

plt.tight_layout()
#plt.savefig('parameters.png')
plt.savefig('/scratch/o/oscarh/aryanah/output_2/parameters.png') 

# %%
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
plt.savefig('/scratch/o/oscarh/aryanah/output_2/fourier.png')

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
plt.savefig('/scratch/o/oscarh/aryanah/output_2/corner_plots.png')