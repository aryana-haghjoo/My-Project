import matplotlib.pyplot as pl
import ares

sim = ares.simulations.Global21cm()      # Initialize a simulation object
sim.run()
m= sim.GlobalSignature(fig=2)
pl.savefig('2.png')
pl.semilogx(sim.history['z'], sim.history['dTb'])
pl.savefig('1.png')
sim.save('test_21cm', clobber=True)

#import pickle
#with open('test_21cm.parameters.pkl', 'rb') as f:
 #   data = pickle.load(f)
#print (data)
anl = ares.analysis.Global21cm('test_21cm')
print(anl)
