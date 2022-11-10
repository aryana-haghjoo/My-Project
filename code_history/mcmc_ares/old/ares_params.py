import numpy as np

#the parameter space for ARES to use is given as a set of touples
# the values of the touples are the names of the parameters, the minimum and maximum values the parameter can take, and the type of narmalization (log or linear)

ares_params = [
	["pop_rad_yield_0_", 1E3, 1E6, "log"],
	["pop_rad_yield_1_", 1E37, 5E40, "log"],
	["pop_rad_yield_2_", 1E3, 5E6, "log"],
	["clumping_factor", 0.05, 2, "lin"]
]


#here's where we set the redshift values for the training data and the resulting neural network

redshifts = np.linspace(5,30,40)

# function to set all parameters into the [0,1] range
def normalize (arr):
	op = np.empty(arr.shape)
	for i in range(arr.shape[1]):
		if ares_params[i][3] == "lin":
			op[:,i] = (arr[:,i] - ares_params[i][1])/(ares_params[i][2] - ares_params[i][1])
		elif ares_params[i][3] == "log":
			op[:,i] = (np.log10(arr[:,i]/ares_params[i][1]))/(np.log10(ares_params[i][2]/ares_params[i][1])) 
		else:
			raise ValueError("Invalid normalization type in ares_params")
	return op

# function to set all parameters into the [param_min, param_max] range
def denormalize (arr):
	op = np.empty(arr.shape)
	for i in range(arr.shape[1]):
		if ares_params[i][3] == "lin":
			op[:,i] = arr[:,i] * (ares_params[i][2] - ares_params[i][1]) + ares_params[i][1]
		elif ares_params[i][3] == "log":
			op[:,i] = 10.0**(arr[:,i] * (np.log10(ares_params[i][2]) - np.log10(ares_params[i][1])) + np.log10(ares_params[i][1]))
		else:
			raise ValueError("Invalid normalization type in ares_params")
	return op

def rescaled_redshifts():
	min = redshifts[0]; max = redshifts[-1]
	return (redshifts.copy() - min)/(max-min)*25
