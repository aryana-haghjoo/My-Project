import ares
import numpy as np
import matplotlib.pyplot as plt

params = {'pop_rad_yield_0_': 10**(4.03), 'pop_rad_yield_1_': 10**(36), 'pop_rad_yield_2_': 10**(5), 'clumping_factor': 0.71} 
sim = ares.simulations.Global21cm(**params, progress_bar=False, verbose=False)
sim.run()
z = sim.history['z'][::-1]
dTb = sim.history['dTb'][::-1]


plt.plot(z)
plt.ylim(5, 50)
plt.savefig('test.png')