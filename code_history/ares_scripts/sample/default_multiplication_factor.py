import ares
import matplotlib.pyplot as plt
import numpy as np

temp=np.arange(0.1, 1, 0.1)
fig1 = plt.figure()
for i in range(len(temp)):
    print("iteration number " + repr(i+1))
    print("Parameter value is " + repr(temp[i]))
    #change this line
    sim = ares.simulations.Global21cm(fX = temp[i])
    sim.run()
    #change this line
    plt.plot(sim.history['z'], sim.history['dTb'], label = "$f_X$= "+str(round(temp[i],1)))
    
#change this line
plt.title(r"$f_X$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\delta T_b$")
plt.legend(loc='lower right')
plt.xlim(5,35)
#change this line
plt.savefig("f_X.png")