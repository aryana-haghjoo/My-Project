import ares
import matplotlib.pyplot as plt
import numpy as np

fesc=np.arange(0.1, 1, 0.1)
fig1 = plt.figure()
for i in range(len(fesc)):
    sim = ares.simulations.Global21cm(fesc = fesc[i])
    sim.run()
    plt.plot(sim.history['z'], sim.history['dTb'], label = "fesc= "+str(round(fesc[i],1)))

plt.title(r"fesc")
plt.xlabel(r"$z$")
plt.ylabel(r"$\delta T_b$")
plt.legend(loc='lower right')
plt.xlim(5,35)
plt.savefig("fesc")