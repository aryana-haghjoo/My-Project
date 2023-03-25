import ares
import numpy as np

sim = ares.simulations.Global21cm(pop_solve_rte = [0,1])
sim.run()
index = np.searchsorted(sim.history["z"][::-1],20)
index = sim.history["z"].shape[0] - index
print(sim.history["z"][index-5:index+5])
print(sim.history["Ja"][index-5:index+5])
