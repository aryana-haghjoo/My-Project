import ares
import matplotlib.pyplot as plt
import numpy as np

temp=np.linspace(20, 90, 10)
fig1 = plt.figure()
for i in range(len(temp)):
    print("iteration number " + repr(i+1))
    a = temp[i]
    print("Parameter value is " + repr(a))
    #change this line
    sim = ares.simulations.Global21cm(zform = a)
    sim.run()
    #change this line
    plt.plot(sim.history['z'], sim.history['dTb'], label = "$zform$= "+str(round(temp[i],1)))
    
#change this line
plt.title(r"$zform$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\delta T_b$")
plt.legend(loc='lower right')
plt.xlim(5,35)
#change this line
plt.savefig("zform.png")