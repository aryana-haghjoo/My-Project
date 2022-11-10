import ares
import matplotlib.pyplot as plt
import numpy as np

fX=np.arange(0.1, 1, 0.1)
fig1 = plt.figure()
for i in range(len(fX)):
    sim = ares.simulations.Global21cm(fX = fX[i])
    sim.run()
    plt.plot(sim.history['z'], sim.history['dTb'], label = "$f_X$= "+str(round(fX[i],1)))

plt.title(r"$f_X$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\delta T_b$")
plt.legend(loc='lower right')
plt.xlim(5,35)
plt.savefig("f_X.png")