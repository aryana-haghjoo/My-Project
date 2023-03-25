#%pylab inline
import ares
import numpy as np
import matplotlib.pyplot as pl

pars_1 = \
{
 'problem_type': 100,              # Blank slate global 21-cm signal


 # Setup star formation
 'pop_Tmin{0}': 1e4,               # atomic cooling halos
 'pop_fstar{0}': 1e-1,             # 10% star formation efficiency
 'aryana' : 0.9,

 # Setup UV emission
 'pop_sed_model{0}': True,
 'pop_sed{0}': 'bb',               # PopII stars -> 10^4 K blackbodies
 'pop_temperature{0}': 1e4,
 'pop_rad_yield{0}': 1e42,
 'pop_fesc{0}': 0.2,
 'pop_Emin{0}': 10.19,
 'pop_Emax{0}': 24.6,
 'pop_EminNorm{0}': 13.6,
 'pop_EmaxNorm{0}': 24.6,
 'pop_lya_src{0}': True,
 'pop_ion_src_cgm{0}': True,
 'pop_heat_src_igm{0}': False,

 # Setup X-ray emission
 'pop_sed{1}': 'pl',
 'pop_alpha{1}': -1.5,
 'pop_rad_yield{1}': 2.6e38,
 'pop_Emin{1}': 2e2,
 'pop_Emax{1}': 3e4,
 'pop_EminNorm{1}': 5e2,
 'pop_EmaxNorm{1}': 8e3,

 'pop_lya_src{1}': False,
 'pop_ion_src_cgm{1}': False,
 'pop_heat_src_igm{1}': True,

 'pop_sfr_model{1}': 'link:sfrd:0',
}

#running sim_1 
sim_1 = ares.simulations.Global21cm(**pars_1)
sim_1.run()

ax_1, zax_1 = sim_1.GlobalSignature(color='k')

#running sim_2
pars_2 = pars_1.copy()
pars_2['pop_fstar{0}'] = 3e-1
sim_2 = ares.simulations.Global21cm(**pars_2)
sim_2.run()

ax_2, zax_2 = sim_2.GlobalSignature(color='b')
pl.savefig('multi_param_1.png')


#running sim_3
pars_3 = pars_1.copy()
pars_3['aryana'] = 0.1
sim_3 = ares.simulations.Global21cm(**pars_3)
sim_3.run()

ax_3, zax_3 = sim_3.GlobalSignature(color='r')
pl.savefig('multi_param_1.png')

#changed: clumping factor
#did not change: fxh, cX, 