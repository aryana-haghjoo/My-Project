import ares
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
#import cProfile
#import re
#cProfile.run('re.compile("foo|bar")')

#loading the EDGES data (the e subscipt is related to EDGES)-----------------------------------------------------------------
data_1 = pd.read_csv('data_1.csv')
freq_e = data_1.iloc[:,0] #frequency, MHz
model_e = data_1.iloc[:, -2] #the model represented in the paper, k
model_e = 1000 * model_e #converting the Data temperatures from k to mK 
model_e = model_e/2 #temporary mfactor

#Changing the axis from frequency to redshift---------------------------------------------------------------------------------
v_0 = 1420 #MHz, frequency of 21cm line
z_e = (v_0/freq_e)-1 #conversion of frequency to redshift

#defining functions-----------------------------------------------------------------------------------------------------------
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
    params_denormalized = list_to_dict(value_denormalized, key_guess)
    
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

def func_ares (m, z, d = 10000): 
    #m is the list of params 
    #z is the redshift range
    #y is the brightness temp
    m = np.array(m)
    T = call_ares (list_to_dict(m, key_guess), z)
    derivs = np.zeros([len(z), len(m)])
    dpars = np.zeros(len(m))
    dpars = m/d 
    for i in range(len(m)):        
        pars_plus = np.array(m, copy=True, dtype = 'float64')
        pars_plus[i] = pars_plus[i] + dpars[i]
        
        pars_minus = np.array(m, copy=True, dtype = 'float64')
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
    return chisq, lhs, rhs

def linv(mat, lamda):
    mat=mat+lamda*np.diag(np.diag(mat))
    return np.linalg.pinv(mat)

#LM inputs ---------------------------------------------------------------------------------------------------------------------------
#start_guess = {'pop_rad_yield_0_': 4.03, 'pop_rad_yield_1_': 36, 'pop_rad_yield_2_': 5, 'clumping_factor': 0.71} 
start_guess = {'pop_rad_yield_0_': 4.04, 'pop_rad_yield_1_': 36, 'pop_rad_yield_2_': 5, 'clumping_factor': 0.71} 
low_bound = np.array([1E3, 1E30, 1E3, 0.05]) 
up_bound = np.array([1E6, 1E40, 1E6, 2])
Ninv = (1E-2)*np.eye(model_e.shape[0], model_e.shape[0]) #inverse of noise matrix
param_length = len(start_guess)                
value_guess, key_guess = dict_to_list(start_guess) #converting start guess to two lists
niter = 100 #number of steps to be taken

#-------------------------------------------------------------------------------------------------------------------------------------
def LM(m, fun, x, y, Ninv=None, niter=10, chitol= 1):
#levenberg-Marquardt Fitter 
    lamda=0
    lamda_trend = []
    lamda_trend.append(lamda)
    chisq, lhs, rhs = get_matrices(m, fun, x, y, Ninv)
    for i in range(niter):
        lhs_inv = linv(lhs, lamda)
        dm = lhs_inv@rhs
        #result = check_limits(m+dm, low_bound, up_bound)
        #print('m: ', m)
        #print('dm: ', dm)
        #if (result==False):
         #   print('I am here 1!')
        """
        while (True): # to chack if  all the parameters are in the reasonable limits
            lhs_inv = linv(lhs, lamda)
            dm = lhs_inv@rhs
            result = check_limits(m+dm, low_bound, up_bound)           
            if (result == True):
                break
            else:
                lamda = update_lamda(lamda, False)
                print('I am here 1!')
                #print('out of range params: ', m+dm)
                lamda_trend.append(lamda)
        """      
        chisq_new, lhs_new, rhs_new = get_matrices(m+dm, fun, x, y, Ninv)
        if chisq_new<chisq:  
            #accept the step
            #check if we think we are converged - for this, check if 
            #lamda is zero (i.e. the steps are sensible), and the change in 
            #chi^2 is very small - that means we must be very close to the
            #current minimum
            if lamda==0:
                if (np.abs(chisq-chisq_new)<chitol):
                    #print(np.abs(chisq-chisq_new))
                    print('Converged after ', i, ' iterations of LM')
                    return m+dm
            chisq = chisq_new
            lhs = lhs_new
            rhs = rhs_new
            m = m+dm
            lamda = update_lamda(lamda,True)
            lamda_trend.append(lamda)
            
        else:
            lamda = update_lamda(lamda, False)
            #print('I am here 2!')
            lamda_trend.append(lamda)
        print('on iteration ', i, ' chisq is ', chisq, ' with step ', dm, ' and lamda ', lamda)
        #print('params are: ', m)
        
    return m, lamda_trend

#------------------------------------------------------------------------------------------------------------------------------
m_fit, lamda_trend = LM (m = value_guess , fun = func_ares , x = z_e , y = model_e , Ninv = Ninv, niter = niter)
T_start, deriv =  func_ares(value_guess, z_e)
T_fit, cov_mat = func_ares(m_fit, z_e)
dim = cov_mat.shape[0]
mycov=(cov_mat.T@cov_mat)/dim

#printing the results----------------------------------------------------------------------------------------------------------
txt = open('/scratch/o/oscarh/aryanah/output_lm/lm_result.txt', 'w')
txt.write('Fitted Parameters: ' + repr(m_fit) + '\n')
txt.write('Covariance Matrix: ' + repr(mycov) + '\n')
txt.close()

fig1 = plt.figure()
plt.plot(z_e , model_e, label = 'EDGES Model')
plt.plot(z_e , T_start, label = 'Start Guess')
plt.plot(z_e , T_fit, label = 'LM result')
plt.title('Result of Levenberg–Marquardt after %d Steps'%niter, fontsize=12)
plt.xlabel('Redshift', fontsize=12)
plt.ylabel('T(mK)', fontsize=12)
plt.legend()
plt.savefig('/scratch/o/oscarh/aryanah/output_lm/lm_result.png')

fig2 = plt.figure()
plt.plot(lamda_trend)
plt.title('Trend of Lambda', fontsize=12)
plt.savefig('/scratch/o/oscarh/aryanah/output_lm/lambda.png')