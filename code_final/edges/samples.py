import matplotlib.pyplot as plt
import numpy as np
import ares
from scipy.interpolate import CubicSpline
import pandas as pd
import signal

#loading the EDGES data (the e subscipt is related to EDGES)
data_1 = pd.read_csv('/home/o/oscarh/aryanah/My-Project/data/data_1.csv')
freq_e = data_1.iloc[:, 0] #frequency, MHz
model_e = data_1.iloc[:, 5] #model, mK

#converting the data from mK to K
model_e = model_e*1000
model_e = model_e/2

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

#dict_true = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_2_': 1E5, 'clumping_factor': 2.5, 'fX': 0.1}
#dict_true = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_2_': 1E3, 'fesc': 0.1, 'fX': 0.1}
params_dict = {'pop_rad_yield_0_': 4.54933009e+03, 'pop_rad_yield_2_': 2.47592394e+03, 'fesc': 3.70100011e-01, 'fX': 1.36397790e-01}
params, key = dict_to_list(params_dict)
#y = call_ares(list_to_dict(params, key), z_e)
params = np.array(params, copy=True, dtype = 'float64')
err = 1E1 #mk
Ninv = ((err)**(-2))*np.eye(len(z_e))
chisq_f = chisquare(params, model_e, Ninv)
n_samples = 10000
mycovinv = np.array([[ 3.24667471e+04,  5.68112552e+06, -8.51686688e+02,
         5.65856615e-01],
       [ 5.68112552e+06,  2.43358043e+10, -3.63874836e+06,
         5.64163649e+00],
       [-8.51686688e+02, -3.63874836e+06,  5.44075269e+02,
        -9.80887059e-04],
       [ 5.65856615e-01,  5.64163649e+00, -9.80887059e-04,
         3.92215290e-05]])

samples = draw_samples(mycovinv/(1000000), n_samples)
csq= np.empty(n_samples)
samples_with_mean= np.empty((n_samples , 4))

for i in range(n_samples):
    samples_with_mean[i, :] = samples[i, :] + params
    csq[i] = chisquare(samples_with_mean[i, :], model_e, Ninv)

csq_diff = csq - chisq_f

#np.savetxt('samples.gz' , samples_with_mean)
np.savetxt('/scratch/o/oscarh/aryanah/samples_edges/samples.gz' , samples_with_mean)

#np.savetxt('csq.gz' , csq)
np.savetxt('/scratch/o/oscarh/aryanah/samples_edges/csq.gz' , csq)

#txt = open('results.txt','w')
txt = open('/scratch/o/oscarh/aryanah/samples_edges/results.txt','w')
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
plt.savefig('/scratch/o/oscarh/aryanah/samples_edges/corner_plots.png')

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
plt.savefig('/scratch/o/oscarh/aryanah/samples_edges/histogram.png')

fig3 = plt.figure()
plt.hist(csq_diff, bins=100)
#plt.savefig('csq_hist.png')
plt.savefig('/scratch/o/oscarh/aryanah/samples_edges/csq_hist.png')

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
plt.savefig('/scratch/o/oscarh/aryanah/samples_edges/csq_params.png')