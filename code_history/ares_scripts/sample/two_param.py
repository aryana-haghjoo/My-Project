import ares
import matplotlib.pyplot as pl
ax = None
for i, fX in enumerate([0.1, 1.]):
    for j, fstar in enumerate([0.1, 0.5]):
        sim = ares.simulations.Global21cm(fX=fX, fstar=fstar,
            verbose=False, progress_bar=False)
        sim.run()


        # Plot the global signal
        ax, zax = sim.GlobalSignature(ax=ax, fig=3, z_ax=i==j==0,
            label=r'$f_X=%.2g, f_{\ast}=%.2g$' % (fX, fstar))


ax.legend(loc='lower right', fontsize=14)
pl.savefig('two_param.png')
