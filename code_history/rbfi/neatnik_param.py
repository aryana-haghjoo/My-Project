#in order to check if the output params of rbfi works better as an starting point for MCMC
# It did not help much
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from scipy.interpolate import interp1d
from numpy.random import normal
from math import ceil
from scipy.interpolate import CubicSpline
from scipy.interpolate import RBFInterpolator
from ares_params import ares_params, redshifts
import pickle


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


neatnik_values = np.array([0.22516714, 0.45246182, 0.94406205, 0.12613563])
neatnik_values_denormalized = denormalize(np.asarray(neatnik_values)[np.newaxis])

print(neatnik_values_denormalized )