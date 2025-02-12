#!/bin/bash 
# set the number of nodes
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

# set max wallclock time
#SBATCH --time=15:00:00

# set name of job
#SBATCH --job-name=aryana

#SBATCH --mail-type=FAIL

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=aryana.haghjoo@mail.mcgill.ca

# set the output directory to the project directory
#SBATCH --output=/scratch/o/oscarh/aryanah/jobout.out

#cd $SLURM_SUBMIT_DIR
 
#module load intel/2018.2
#module load openmpi/3.1.0

#cd $SCRATCH
module load python/3.7.9
source ~/.virtualenvs/ares/bin/activate 
export PYTHONPATH="/home/o/oscarh/aryanah/ares"

cd /home/o/oscarh/aryanah/MCMC/EDGES_ARES_with_gaussian_distri/check
python check_newparams.py
