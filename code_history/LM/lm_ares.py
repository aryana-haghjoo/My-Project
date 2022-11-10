import ares
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from numpy.random import normal
from math import ceil
from scipy.interpolate import CubicSpline
#import cProfile
#import re
#cProfile.run('re.compile("foo|bar")')


#loading the EDGES data (the e subscipt is related to EDGES)-----------------------------------------------------------------
data_1 = pd.read_csv('data_1.csv')

freq_e = data_1.iloc[:,0] #frequency, MHz

#T_e = data_1.iloc[:,-1] #21cm brighness temperature, k

model_e = data_1.iloc[:, -2] #the model represented in the paper, k

#temporary mfactor
model_e = model_e/2

#Changing the data from frequency to redshift---------------------------------------------------------------------------------
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift


#converting the Data temperatures from k to mK 
k = 1000
model_e = k * model_e


#defining functions----------------------------------------------------------------------------------------------

def dict_to_list(d): # converts dictionary to two lists (key and value)
    #d must be a dictionary containing the value of parameters and their names
    key = list(d.keys())
    value = list(d.values())
    return value, key

def list_to_dict(value, key): #converts two lists (key and value) to a dictionary
    #value is a list of parameters' values
    #key is a list parameters' names
    return dict(zip(key, value))


def call_ares (params, redshifts): #4params
    '''
    Returns the temperature fluctuation vs redshift curve.
    INPUTS:
        ares_params: Specify which the values of the desired paramamters.
        redshift: Specify the redshift over which to graph the curve.
    OUTPUTS:
        A cubic spline of the temperature fluctuation vs redshift curve produced by ares.
    '''    
    
    #to denormalize the values
    value, key = dict_to_list(params)
    value_denormalized = np.array(value, dtype='float64')
    value_denormalized[0] = 10** (value[0])
    value_denormalized[1] = 10** (value[1])
    value_denormalized[2] = 10** (value[2])
    params_denormalized = list_to_dict(value_denormalized, key_guess)
    
    #running ares
    sim = ares.simulations.Global21cm(**params_denormalized, verbose=False, progress_bar=False)
    sim.run()
    z = sim.history['z'][::-1]
    dTb = sim.history['dTb'][::-1]
    sorted_idx = np.argsort(z,kind="stable")
    z = z[sorted_idx]
    dTb = dTb[sorted_idx]
    spline = CubicSpline(z, dTb)	
    return spline(redshifts) 

"""
def call_ares(input_dict, redshifts): #5params

    tmp = ares.util.ParameterBundle('mirocha2017:base')
    tmp.update(input_dict)
    tmp.update({"pop_binaries_0_":True})
    params = tmp
    sim = ares.simulations.Global21cm(**params, verbose = False, progress_bar = False)
    sim.run()
    z = sim.history['z'][::-1]
    sorted_idx = np.argsort(z, kind="stable")
    dTb = sim.history['dTb'][::-1]
    z = z[sorted_idx]
    dTb = dTb[sorted_idx]
    spline_dTb = CubicSpline(z, dTb)
    return spline_dTb(redshifts)
"""

def func_ares (m , z, d = 1000): 
    #m is the list of params that shall be denormalized
    #z is the redshift range
    #y is the brightness temp
    T = call_ares (list_to_dict(m, key_guess), z)
    derivs=np.zeros([len(z), len(m)])
    dpars = np.zeros(len(m))
    #calculating the local derivative with respect to each parameter
    #dpars = 0.01* m
    for i in range(len(m)):
        
        #you may want to change this in future
        dpars [i] = m [i] /d
        #pars_plus : m + dm
        pars_plus = m.copy()
        pars_plus[i] = pars_plus[i] + dpars[i]
        
        #pars_minus : m - dm
        pars_minus = m.copy()        
        pars_minus[i] = pars_minus[i] - dpars[i]
        
        A_plus = call_ares (list_to_dict(pars_plus, key_guess), z)
        A_minus = call_ares (list_to_dict(pars_minus, key_guess), z)
        A_m = (A_plus - A_minus)/(2*dpars[i])
        derivs[:, i] = A_m
        
    return T, derivs


def check_limits(new_param, low, up): #to check if the new params exceed the reasonable limits or not (e.g f_Star>1)
    l = len(new_param)
        
    check_array=np.zeros(l, dtype=bool)
    result = True
        
    for i in range (0,l):
        
        if new_param[i]<= up[i]:
            
            if new_param[i] >= low[i]:
                check_array[i] = True
            else: 
                check_array[i] =False
        else: 
            check_array[i] =False
        
        result = result & check_array[i] 
        
        
    return result

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
        lhs=derivs.T@Ninv@derivs
        rhs=derivs.T@(Ninv@r)
        chisq=r.T@Ninv@r
    return chisq, lhs, rhs , derivs

def linv(mat, lamda):
    mat=mat+lamda*np.diag(np.diag(mat))
    return np.linalg.inv(mat)

#LM inputs (the guess subscipt is related to the original guess)------------------------------------------------------------------
#start_guess = {'pop_rad_yield_0_': 1E4, 'pop_rad_yield_1_': 1E38, 'pop_rad_yield_2_': 1E5, 'clumping_factor': 1} #4param version, not normalized

start_guess = {'pop_rad_yield_0_': 4.03, 'pop_rad_yield_1_': 35, 'pop_rad_yield_2_': 4.7, 'clumping_factor': 0.71} #4param version, normalized
#["pop_rad_yield_0_", 1E3, 1E6, "log"], ["pop_rad_yield_1_", 1E37, 5E40, "log"], ["pop_rad_yield_2_", 1E3, 5E6, "log"], ["clumping_factor", 0.05, 2, "lin"]
    
#start_guess ={"pop_Tmin_0_": 1E4, "pop_fesc_0_": 0.2, "pop_rad_yield_1_": 1E39, "pq_func_par2[0]_0_": 0.5, "pq_func_par0[0]_0_": 0.06} #5param version, not normalized, nearly mid-value
#start_guess ={"pop_Tmin_0_": 1E4, "pop_fesc_0_": 0.2, "pop_rad_yield_1_": 5E40, "pq_func_par2[0]_0_": 0.378, "pq_func_par0[0]_0_": 0.0892} #5param version, not normalized, closest values to data
#["pop_Tmin_0_", 1E3, 1.25E5, "log"], ["pop_fesc_0_", 0.108, 0.358, "lin"], ["pop_rad_yield_1_", 5E38, 5E40, "log"], ["pq_func_par2[0]_0_", 0.378, 0.6605, "lin"], ["pq_func_par0[0]_0_", 0.0271, 0.0892, "log"]

#resonable range for each parameter so ARES won't crash!
#low_bound = [1E3, 0.108, 5E38, 0.378, 0.0271] #5param version, not normalized
#up_bound = [1.25E5, 0.358, 5E40, 0.6605, 0.0892] #5param version, not normalized

#low_bound = np.array([2, 30, 2, 1E-5]) #4param version, normalized
#up_bound = np.array([7, 45, 7, 4]) #4param version, normalized

low_bound = np.array([3, 30, 3, 0.05]) #4param version, normalized
up_bound = np.array([6, 40, 6, 2]) #4param version, normalized

param_length = len(start_guess)
                 
#converting start guess to two lists
value_guess, key_guess = dict_to_list(start_guess)

niter = 20 #number of steps to be taken

#-------------------------------------------------------------------------------------------------------------------------------------
def LM(m, fun, x, y, Ninv=None, niter=10, chitol=0.01):
#levenberg-Marquardt Fitter 
    lamda=0
    lamda_trend = []
    lamda_trend.append(lamda)
    chisq, lhs, rhs, cov_mat = get_matrices(m, fun, x, y, Ninv)
    for i in range(niter):
        print('step number ', i, ' of ', niter)
        #lhs_inv = linv(lhs, lamda)
        #dm = lhs_inv@rhs
        
        while (True): # to chack if  all the parameters are in the reasonable limits
            lhs_inv = linv(lhs, lamda)
            dm = lhs_inv@rhs
            result = check_limits(m+dm, low_bound, up_bound)           
            if (result == True):
                break
            else:
                lamda = update_lamda(lamda, False)
                #print('I am here 1!')
                #print('out of range params: ', m+dm)
                lamda_trend.append(lamda)
              
        chisq_new, lhs_new, rhs_new, cov_mat = get_matrices(m+dm, fun, x, y, Ninv)
        if chisq_new<chisq:  
            #accept the step
            #check if we think we are converged - for this, check if 
            #lamda is zero (i.e. the steps are sensible), and the change in 
            #chi^2 is very small - that means we must be very close to the
            #current minimum
            if lamda==0:
                if (np.abs(chisq-chisq_new)<chitol):
                    print(np.abs(chisq-chisq_new))
                    print('Converged after ', i, ' iterations of LM')
                    return m+dm
            chisq = chisq_new
            lhs = lhs_new
            rhs = rhs_new
            m = m+dm
            lamda = update_lamda(lamda,True)
            lamda_trend.append(lamda)
            
        else:
            lamda = update_lamda(lamda,False)
            print('I am here 2!')
            lamda_trend.append(lamda)
        print('on iteration ', i, ' chisq is ', chisq, ' with step ', dm, ' and lamda ', lamda)
        #print('params are: ', m)
    return m, lamda_trend, cov_mat
#------------------------------------------------------------------------------------------------------------------------------
m_fit, lamda_trend, cov_mat = LM (m = value_guess , fun = func_ares , x = z_e , y = model_e , niter = niter)
dim = cov_mat.shape[0]
mycov=(cov_mat.T@cov_mat)/dim
#printing the results------------------------------------------------------------------------------------------------------
txt = open('lm_result.txt', 'w')
txt.write('Fitted Parameters: ' + repr(m_fit) + '\n')
txt.write('Covariance Matrix: ' + repr(mycov) + '\n')
txt.close()
#--------------------------------------------------------------------------------------------------
T_start = call_ares(start_guess, z_e)
T_fit = call_ares(list_to_dict (m_fit , key_guess) , z_e)

fig1 = plt.figure()
plt.plot(z_e , model_e, label = 'EDGES Model')
plt.plot(z_e , T_start, label = 'Start Guess')
plt.plot(z_e , T_fit, label = 'LM result')
plt.title('Result of Levenberg–Marquardt after %d Steps'%niter, fontsize=12)
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
plt.savefig('LM_result.png')
#plt.savefig('/scratch/o/oscarh/aryanah/MCMC_output/mcmc_result.png')

fig2 = plt.figure()
plt.plot(lamda_trend)
plt.title('Trend of Lambda', fontsize=12)
plt.savefig('lambda.png')
#plt.savefig('/scratch/o/oscarh/aryanah/MCMC_output/lambda.png')