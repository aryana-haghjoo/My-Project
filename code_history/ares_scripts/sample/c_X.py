import ares
import matplotlib.pyplot as plt
import numpy as np

cX=np.linspace(5, 400, 10)
fig1 = plt.figure()
for i in range(len(cX)):
    a = (1E38)* cX[i]
    sim = ares.simulations.Global21cm(cX = a)
    sim.run()
    plt.plot(sim.history['z'], sim.history['dTb'], label = "$c_X$= "+str(round(cX[i],1))+ "E38")

plt.title(r"$c_X$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\delta T_b$")
plt.legend(loc='lower right')
plt.xlim(5,35)
plt.savefig("c_X.png")