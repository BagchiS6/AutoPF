#!/bin/bash

#SBATCH -A mat269
#SBATCH -p batch
#SBATCH -J matensemble_moose_test
#SBATCH -N 4
#SBATCH -t 1:00:00

unset SLURM_EXPORT_ENV
### IMPORTANT: Have to request MORE THAN 1 NODE 
### as matensemble reserves the root node for orchestration

module load miniforge3 openmpi/5.0.5 gcc/12.4.0 
source activate /gpfs/wolf2/cades/mat269/world-shared/autopf_env

# replace with your workflow script
export YOUR_PYTHON_WORKFLOW_CODE=demo_workflow.py  


# finally launch matenseble
matensemble-launch python $YOUR_PYTHON_WORKFLOW_CODE
