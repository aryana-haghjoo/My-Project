import ares
import matplotlib.pyplot as plt
import numpy as np

temp=np.linspace(1, 10, 10)
fig1 = plt.figure()
for i in range(len(temp)):
    print("iteration number " + repr(i+1))
    a = temp[i]
    print("Parameter value is " + repr(a))
    sim = ares.simulations.Global21cm(hmf_logMmin = a)
    sim.run()
    plt.plot(sim.history['z'], sim.history['dTb'], label = "$hmf_logMmin$= "+str(round(temp[i],1)))

plt.title(r"$hmf_logMmin$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\delta T_b$")
plt.legend(loc='lower right')
plt.xlim(5,35)
plt.savefig("hmf_logMmin.png")