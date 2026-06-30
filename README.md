# AutoPF

Automating High-throughput Phase Field simulations using MOOSE and MatEnsemble

![AutoPF scalable orchestration overview](images/SI_fig_autopf_matensemble_scalable_orchestration.png)


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
