import ares
import matplotlib.pyplot as plt
import numpy as np

p_f=np.arange(0.1, 1, 0.1)
fig1 = plt.figure()
for i in range(len(p_f)):
    sim = ares.simulations.Global21cm(fstar = p_f[i])
    sim.run()
    plt.plot(sim.history['z'], sim.history['dTb'], label = "PF= "+str(round(p_f[i],1)))

plt.title(r"fstar")
plt.xlabel(r"$z$")
plt.ylabel(r"$\delta T_b$")
plt.legend(loc='lower right')
plt.xlim(5,35)
plt.savefig("fstar.png")