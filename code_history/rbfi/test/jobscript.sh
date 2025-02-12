#!/bin/bash 
# set the number of nodes
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

# set max wallclock time
#SBATCH --time=24:00:00

# set name of job
#SBATCH --job-name=aryana-rbfi

#SBATCH --mail-type=FAIL

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=aryana.haghjoo@mail.mcgill.ca

# set the output directory to the project directory
#SBATCH --output=/scratch/o/oscarh/aryanah/rbfi_out.out

#cd $SLURM_SUBMIT_DIR
 
#module load intel/2018.2
#module load openmpi/3.1.0

#cd $SCRATCH
module load python/3.9

cd /home/o/oscarh/aryanah/rbfi/cov_mat
python MCMC_EDGES_RBFI_check.py
