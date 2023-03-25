import ares
import matplotlib.pyplot as plt
import numpy as np

c_f=np.arange(0.1, 1, 0.1)
fig1 = plt.figure()
for i in range(len(c_f)):
    sim = ares.simulations.Global21cm(clumping_factor = c_f[i])
    sim.run()
    plt.plot(sim.history['z'], sim.history['dTb'], label = "CF= "+str(round(c_f[i],1)))

plt.title(r"clumping_factor")
plt.xlabel(r"$z$")
plt.ylabel(r"$\delta T_b$")
plt.legend(loc='lower right')
plt.xlim(5,35)
plt.savefig("clumping_factor.png")