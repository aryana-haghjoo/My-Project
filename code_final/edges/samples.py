import matplotlib.pyplot as plt
import numpy as np
import ares
from scipy.interpolate import CubicSpline
import pandas as pd

#loading the EDGES data (the e subscipt is related to EDGES)
data_1 = pd.read_csv('/home/o/oscarh/aryanah/My-Project/data/data_1.csv')
freq_e = data_1.iloc[:, 0] #frequency, MHz

#Changing the axis from frequency to redshift
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

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

def chisquare (pars, data, Ninv): #returns the chi-square of two 21cm curves - err can be a number/array   
    pred = call_ares(list_to_dict(pars, key), z_e)
    r = data-pred
    chisq = r.T@Ninv@r
    return chisq

#dict_true = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_2_': 1E5, 'clumping_factor': 2.5, 'fX': 0.1}
dict_true = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_2_': 1E3, 'fesc': 0.1, 'fX': 0.1}
m_true, key = dict_to_list(dict_true)
y_true = call_ares(list_to_dict(m_true, key), z_e)
m_true = np.array(m_true, copy=True, dtype = 'float64')
err = 1E-3
Ninv = ((err)**(-2))*np.eye(len(z_e))
chisq_f = chisquare(m_true, y_true, Ninv)
n_samples = 10000
mycovinv = np.array([[ 1.64033521e-03,  1.96805295e-02, -4.39594925e-07,
        -5.37628828e-08],
       [ 1.96805295e-02,  1.60512857e+01, -1.44165451e-03,
        -5.94152648e-06],
       [-4.39594925e-07, -1.44165451e-03,  1.38232677e-07,
         2.15944098e-10],
       [-5.37628828e-08, -5.94152648e-06,  2.15944098e-10,
         1.38004743e-11]])

samples = draw_samples(mycovinv, n_samples)
csq= np.empty(n_samples)
samples_with_mean= np.empty((n_samples , 4))

for i in range(n_samples):
    samples_with_mean[i, :] = samples[i, :] + m_true
    csq[i] = chisquare(samples_with_mean[i, :], y_true, Ninv)

csq_diff = csq - chisq_f

#np.savetxt('samples.gz' , samples_with_mean)
np.savetxt('/scratch/o/oscarh/aryanah/samples_known_curve/samples.gz' , samples_with_mean)

#np.savetxt('csq.gz' , csq)
np.savetxt('/scratch/o/oscarh/aryanah/samples_known_curve/csq.gz' , csq)

#txt = open('results.txt','w')
txt = open('/scratch/o/oscarh/aryanah/samples_known_curve/results.txt','w')
txt.write('Mean of samples: ' + repr(np.mean(samples_with_mean, axis=0)) + '\n')
txt.write('RMS error of samples: ' + repr(np.std((samples.T@samples)/n_samples - mycovinv)) + '\n')
txt.write('Chi-Sqaure at the point of best fit: ' +repr(chisq_f) + '\n')
txt.write('Mean of difference between the chi-squares: '+ repr(np.mean(csq_diff)))
txt.close()

params_cut = np.copy(samples_with_mean)
fig1, ax_list = plt.subplots(3, 2, figsize=(10,10))
fig1.suptitle('Corner plots of the samples', fontsize=16)

ax_list[0, 0].plot(params_cut[:, 0], params_cut[:, 1], linestyle = "", marker=".")
ax_list[0, 0].set_ylabel('param 1', fontsize=12)
ax_list[0, 0].set_xlabel('param 0', fontsize=12)

ax_list[0, 1].plot(params_cut[:, 0], params_cut[:, 2], linestyle = "", marker=".")
ax_list[0, 1].set_ylabel('param 2', fontsize=12)
ax_list[0, 1].set_xlabel('param 0', fontsize=12)
    
ax_list[1, 0].plot(params_cut[:, 0], params_cut[:, 3], linestyle = "", marker=".")
ax_list[1, 0].set_ylabel('param 3', fontsize=12)
ax_list[1, 0].set_xlabel('param 0', fontsize=12)

ax_list[1, 1].plot(params_cut[:, 1], params_cut[:, 2], linestyle = "", marker=".")
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
plt.savefig('/scratch/o/oscarh/aryanah/samples_known_curve/corner_plots.png')

fig2, ax_list = plt.subplots(2, 2, figsize=(10,10))
ax_list[0, 0].hist(samples_with_mean[:, 0], bins = 50)
ax_list[0, 0].set_ylabel('param 0', fontsize=12)

ax_list[0, 1].hist(samples_with_mean[:, 1], bins = 50)
ax_list[0, 1].set_ylabel('param 1', fontsize=12)

ax_list[1, 0].hist(samples_with_mean[:, 2], bins = 50)
ax_list[1, 0].set_ylabel('param 2', fontsize=12)

ax_list[1, 1].hist(samples_with_mean[:, 3], bins = 50)
ax_list[1, 1].set_ylabel('param 3', fontsize=12)
plt.tight_layout()

#plt.savefig('histogram.png')
plt.savefig('/scratch/o/oscarh/aryanah/samples_known_curve/histogram.png')

fig3 = plt.figure()
plt.hist(csq_diff, bins=100)
#plt.savefig('csq_hist.png')
plt.savefig('/scratch/o/oscarh/aryanah/samples_known_curve/csq_hist.png')

fig4, ax_list = plt.subplots(2, 2, figsize=(12,10))
fig4.suptitle('Chi-Square vs paramters', fontsize=16)

ax_list[0, 0].scatter(samples_with_mean[:, 0], csq)
ax_list[0, 0].set_ylabel('Chi-Square', fontsize=12)
ax_list[0, 0].set_xlabel('param 0', fontsize=12)

ax_list[0, 1].scatter(samples_with_mean[:, 1], csq)
ax_list[0, 1].set_ylabel('Chi-Square', fontsize=12)
ax_list[0, 1].set_xlabel('param 1', fontsize=12)

ax_list[1, 0].scatter(samples_with_mean[:, 2], csq)
ax_list[1, 0].set_ylabel('Chi-Square', fontsize=12)
ax_list[1, 0].set_xlabel('param 2', fontsize=12)

ax_list[1, 1].scatter(samples_with_mean[:, 3], csq)
ax_list[1, 1].set_ylabel('Chi-Square', fontsize=12)
ax_list[1, 1].set_xlabel('param 3', fontsize=12)

#plt.savefig('csq_params.png')
plt.savefig('/scratch/o/oscarh/aryanah/samples_known_curve/csq_params.png')