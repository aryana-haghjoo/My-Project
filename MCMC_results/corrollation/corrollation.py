import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('mcmc_params_5000', sep=" ", header=None)
cX= data.iloc[:, 2]
fX= data.iloc[:, 5]

plt.plot (fX, cX)
plt.xlabel('fX')
plt.ylabel('cX')
plt.savefig('corr_5000.png')
