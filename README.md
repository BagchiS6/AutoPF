# AutoPF

Automating High-throughput Phase Field simulations using MOOSE and MatEnsemble

## Getting Started

This is currently only for use in `CADES/Baseline`, where the MatEnsemble is already setup with compatible modules to run MOOSE high-throughput
c.f. `example_allen_cahn` directory for a demo to automate a simple Allen-Cahn model for a set of mobility and interfacial energy parameters. The demo is able to run high-throughput converged Allen-Cahn for a set of 100 paramter combinations for small to large moose jobs. 

You need to make sure that your MOOSE-app is compiled against `Openmpi/5.0.5` as it will not run in `CADES/Baseline` otherwise and also wont be compaitble with MatEnsemble. 

For help regarding Moose compilation, you could look at 
`/gpfs/wolf2/cades/mat269/world-shared/MOOSE/build_12.16.2025/build_moose_baseline.sh`
You will also find couple of executables in the path having PhaseField, SolidMechanics and ElectroMagnetics modules activated. 

****Very imporant: You need always request MORE THAN 1 NODES (while running either in batch through `sbatch`or interactive through `salloc`) as MatEnsemble reserves the root node for memory management and orchestration

```bash

git clone https://github.com/BagchiS6/AutoPF.git
cd AutoPF
pip install -e . --force-reinstall --no-deps
```

## Usage

```python
from autopf.utils import autopf, read_job_stats

# Define simulation parameters
params = {
    'total_jobs': 100,
    'base_input': 'AC.i',
    'arg_list': [...],
    'num_cores': 4,
    'directory_list': [...]
}

# Run ensemble
autopf('path/to/moose-app', params)

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