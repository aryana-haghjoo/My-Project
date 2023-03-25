import ares
import matplotlib.pyplot as plt
import numpy as np

temp=np.arange(0.1, 1, 0.1)
fig1 = plt.figure()
for i in range(len(temp)):
    print("iteration number " + repr(i+1))
    print("Parameter value is " + repr(temp[i]))
    sim = ares.simulations.Global21cm(fXh = temp[i])
    sim.run()
    plt.plot(sim.history['z'], sim.history['dTb'], label = "$fXh$= "+str(round(temp[i],1)))

plt.title(r"$fXh$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\delta T_b$")
plt.legend(loc='lower right')
plt.xlim(5,35)
plt.savefig("fXh.png")