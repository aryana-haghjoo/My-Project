import ares
import matplotlib.pyplot as plt
import numpy as np

#change this line
temp=np.linspace(5, 400, 10)
fig1 = plt.figure()
for i in range(len(temp)):
    print("iteration number " + repr(i+1))
    a = temp[i]
    print("Parameter value is " + repr(a))
    #change this line
    sim = ares.simulations.Global21cm(cX = a)
    sim.run()
    #change this line
    plt.plot(sim.history['z'], sim.history['dTb'], label = "$c_X$= "+str(round(temp[i],1)))
    
#change this line
plt.title(r"$c_X$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\delta T_b$")
plt.legend(loc='lower right')
plt.xlim(5,35)
#change this line
plt.savefig("c_X.png")