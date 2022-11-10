import ares
import matplotlib.pyplot as plt
import numpy as np

temp=np.linspace(22.9, 23.1, 10)
fig1 = plt.figure()
for i in range(len(temp)):
    print("iteration number " + repr(i+1))
    a = (1)* temp[i]
    print("Parameter value is " + repr(a))
    sim = ares.simulations.Global21cm(lya_nmax = a)
    sim.run()
    plt.plot(sim.history['z'], sim.history['dTb'], label = "$lya\_nmax$= "+str(round(temp[i],1)))

plt.title(r"$lya\_nmax$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\delta T_b$")
plt.legend(loc='lower right')
plt.xlim(5,35)
plt.ylim(0, 15)
plt.savefig("lya_nmax.png")