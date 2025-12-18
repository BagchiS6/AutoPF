# AutoPF

Automating High-throughput Phase Field simulations using MOOSE and MatEnsemble

## Getting Started

This is currently only for use in `CADES/Baseline`, where the MatEnsemble is already setup with compatible modules to run MOOSE high-throughput

```bash

git clone 
pip install -e .
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
## launch your own workflow code with 