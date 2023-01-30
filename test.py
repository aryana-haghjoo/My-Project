print('hello world!')
import ares
import matplotlib.pyplot as plt

sim = ares.simulations.Global21cm()
sim.run()
plt.semilogx(sim.history['z'], sim.history['dTb'])
plt.savefig('test.png')