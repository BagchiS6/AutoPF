# AutoPF

Automating High-throughput Phase Field simulations using MOOSE and MatEnsemble

## Getting Started

This is currently only for use in `CADES/Baseline` along with specifc conda environment `/gpfs/wolf2/cades/mat269/world-shared/autopf_env`, where MatEnsemble is already setup with compatible modules to run MOOSE high-throughput.
c.f. `example_allen_cahn` directory for a demo to automate a simple Allen-Cahn model for a set of mobility and interfacial energy parameters. The demo is able to run high-throughput converged Allen-Cahn simulations for a set of 100 paramter combinations for small (4cores, 100 meshpoints) to large scale  (64 cores, $10^4$ meshpoints) processes. 

You need to make sure that your MOOSE-app is compiled against `Openmpi/5.0.5` as it might not run otherwise in `CADES/Baseline` otherwise and also wont be compaitble with MatEnsemble. 

For help regarding Moose compilation, you could look at 
`/gpfs/wolf2/cades/mat269/world-shared/MOOSE/build_12.16.2025/build_moose_baseline.sh`
You will also find couple of executables in the path having PhaseField, SolidMechanics and ElectroMagnetics modules activated. 

**⚠️ IMPORTANT: You must always request MORE THAN 1 NODE (i.e. while running either in batch through `sbatch` or interactive through `salloc`) as MatEnsemble reserves the root node for memory management and orchestration.**

```bash

git clone git@github.com:BagchiS6/AutoPF.git
cd AutoPF
pip install -e . --force-reinstall --no-deps
```

## Usage

```python
from autopf.utils import automoose, read_job_stats

# Define simulation parameters
params = {
    'total_jobs': 100,
    'base_input': 'AC.i',
    'arg_list': [...],
    'num_cores': 4,
    'directory_list': [...]
}

# Run ensemble
automoose('path/to/moose-app', params)

# Read results
job_stats = read_job_stats()
```
## load relevant modules and launch your own workflow code with 
```
module load miniforge3 
source activate /gpfs/wolf2/cades/mat269/world-shared/autopf_env
module load openmpi/5.0.5 gcc/12.4.0 

# replace with your workflow script
export YOUR_PYTHON_WORKFLOW_CODE=demo_workflow.py  


# finally launch matenseble
matensemble-launch python $YOUR_PYTHON_WORKFLOW_CODE
```

**Read job stats via `read_job_stats` function**
This will generate a `JSON` file with information regarding jobs which have 1) `Completed`, 2) still `Running`, 3) yet to be scheduled i.e. `Pending` and 4) jobs which might have `Failed` due to some error with some job `metadata` e.g., `directory, args, matensemble task ids` etc. 

**Note**: You should always check the outputs going othe relevant job directories and cehcking through `stdout` and `stderr` for debugging. Also, it's important to note that a `Failed` job according to MatEnsemble might not have failed from MOOSE perspective, it just means an Error code was picked up (which could be very well due to failure from initial `jit` attempts of the Moose app, while the app could still run fine to produce desired solution output). 