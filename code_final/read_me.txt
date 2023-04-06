This directory contains the last version of LM+MCMC codes to fit the parameters of the gaussian fuction, a known ares curve and edges data.
For the gaussian, all the code is included in one jupyter note book.
For the known ares curve, there is no need to run the LM, since we known the true paramters. We only calculate the covariance matrix from them and run the mcmc.
Please note that for known curve and edges data, each run has 3 scripts:
1) The first script calculates the covariance matrix.
2) The second script draws the samples from the covariance matrix and calculates the chi-square of each of them.
3) The third and final script runs the actuall mcmc chain which uses the samples and their chi-squares from the second script. 