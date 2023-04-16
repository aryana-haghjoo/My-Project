#!/bin/bash 
# set the number of nodes
#SBATCH --nodes=1

#SBATCH --ntasks-per-node=1

# set max wallclock time
#SBATCH --time=20:00:00

# set name of job
#SBATCH --job-name=samples_known_curve

#SBATCH --mail-type=FAIL

# mail alert at start, end and abortion of execution
#SBATCH --mail-type=ALL

# send mail to this address
#SBATCH --mail-user=aryana.haghjoo@mail.mcgill.ca

# set the output directory to the project directory
#SBATCH --output=/scratch/o/oscarh/aryanah/samples_known_curve/job_out.out

module load python/3.9
source ~/.virtualenvs/ares/bin/activate 
export PYTHONPATH="/home/o/oscarh/aryanah/ares"

cd /home/o/oscarh/aryanah/My-Project/code_final/known_curve
python samples.py