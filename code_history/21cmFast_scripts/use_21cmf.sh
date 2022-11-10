#!/bin/bash

echo "Entering the 21cmfast virtual environment and setting the source"
module load NiaEnv/2019b python/3.8 gcc fftw gsl
source ~/.virtualenvs/21cmfast/bin/activate
