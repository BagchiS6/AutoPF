# Restartable AutoPF–AEcroscopyWave loop

This example isolates the stable contract between autonomous microscopy and
theory-side HPC. The same `ClosedLoopRunner` works with:

- `ProxyAcquisitionBackend` for replay data, digital twins, tests, and
  development without a microscope;
- `AEcroscopyWaveAcquisitionBackend` for live jobs sent through `afw-server`;
- `ProxySimulationBackend` for workflow testing;
- `MatEnsembleMOOSEBackend` for batched MOOSE evaluations on HPC.

The campaign state machine is:

```text
plan experiment
  -> submit acquisition -> checkpoint remote job ID
  -> collect observation -> checkpoint Tiled/data references
  -> propose theory batch
  -> submit MOOSE batch -> checkpoint batch ID
  -> collect simulations
  -> update POD–GP/posterior/TS state
  -> repeat
```

Every transition is written atomically. Restarting the same command reuses the
saved AEcroscopyWave and simulation handles instead of repeating completed
actions.

## Run entirely with proxy data

From this directory:

```bash
pip install -e ../..
python run_proxy.py --iterations 4
```

Run it again with a larger iteration count to continue the same checkpoint:

```bash
python run_proxy.py --iterations 8
```

## Connect AEcroscopyWave

AEcroscopyWave 0.1.21 exposes the cloud job API used by the adapter. Configure:

```bash
export AFW_CLOUD_SERVER_URL=http://localhost:8765
export AFW_INSTRUMENT_ID=cypher-1
export AFW_AGENT_SECRET='<shared server token>'
```

The AEcroscopyWave agent publishes PFM arrays to Tiled and returns
`tiled_uri`, `tiled_run_key`, and `tiled_array_keys`; the resolver should load
the selected channel. The runnable example aliases `pfm_amplitude` to the
strategy field. For scientific use, supply the experiment's calibrated
amplitude/phase-to-displacement transform through `TiledArrayResolver` before
calling that field `surface_uz`.

```bash
python run_aecroscopywave.py --iterations 4
```

The bearer token is retained only in process memory and is never written to the
AutoPF checkpoint.

## Production path: live microscope, real MOOSE, online POD–GP

`run_production.py` contains no proxy acquisition or proxy simulation. It
constructs the workflow from AutoPF's production components:

- `AEcroscopyWaveAcquisitionBackend` submits the reviewed microscope workflow
  and resolves its Tiled field;
- `MatEnsembleAsyncTSBackend` runs real MOOSE chores from scratch and replaces
  each completed chore immediately;
- `PODGaussianProcess` learns a low-rank full-field emulator and conditions its
  posterior after every returned MOOSE field;
- `ProductionPODGPStrategy` uses the updated posterior for the next Thompson
  query and checkpoints model hashes and candidate accounting.

Copy `production_config.template.json` and `site_hooks.template.py` into a
deployment directory. `physics_candidates.template.json` and
`acquisition_plan.template.json` document the reviewed candidate and microscope
payload schemas. The two site hooks are intentionally explicit:

1. calibrated PFM channel to physical `u_z`;
2. a reviewed MOOSE task wrapper that prints a result containing the extracted
   surface field path.

The scientific objective is selected and frozen at campaign start:

```bash
python run_production.py --config production.json --preflight-only \
  --offline-instrument-preflight
python run_production.py --config production.json \
  --objective-mode posterior_distance --iterations 1
```

Choose `pixel_nmse` for aligned residual learning or `posterior_distance` for
the calibration-noise-aware Gaussian score (Mahalanobis plus log determinant).
The latter objective artifact is an NPZ containing `basis` with shape
`(features, pixels)` and `covariance` with shape `(features, features)`; both
must be estimated from calibration data without accessing the locked holdout.
An agent may write a reviewed JSON receipt containing `objective_mode`; it can
select only these registered policies and cannot inject code or change a
checkpointed campaign.

For the asynchronous backend, the first space-filling MOOSE wave releases a
GPyTorch sparse variational POD–GP. Thereafter every completion is incorporated
under a file lock before a replacement Thompson query is emitted, while the
remaining MOOSE workers stay active. The POD basis, inducing locations, and GP
hyperparameters are frozen after release; short warm-start updates optimize
the variational distribution. BoTorch's reparameterized Sobol sampler draws
one joint posterior realization over the candidate set for each Thompson
query, preserving candidate-to-candidate GP covariance.

## Connect MOOSE/MatEnsemble

For non-adaptive batch campaigns, replace `ProxySimulationBackend` with:

```python
from autopf.closed_loop.backends import MatEnsembleMOOSEBackend

simulation = MatEnsembleMOOSEBackend(
    moose_app="/path/to/application-opt",
    base_input="/path/to/model.i",
    output_root="/scratch/project/autopf-live",
    num_cores=64,
    argument_builder=build_moose_overrides,
    result_loader=load_surface_fields,
)
```

`argument_builder` maps a `SimulationRequest` to MOOSE command-line overrides.
`result_loader` turns each completed run directory into JSON summaries and
portable field paths. Launch the containing program inside the site's
MatEnsemble allocation.

## NERSC Perlmutter deployment (`m5014_g`)

The NERSC profile uses the same production runner, objective policies, online
POD--GP, and asynchronous MatEnsemble controller as Pathfinder. Only storage,
Slurm resources, the Python environment, and the MOOSE executable are
site-specific. Paths containing `$SCRATCH`, `$NERSC_DEPLOY_ROOT`,
`$AUTOPF_REPO`, and `$MOOSE_APP` are expanded when the JSON is loaded.

On a Perlmutter login node, stage the reviewed deployment bundle onto the
all-flash scratch filesystem:

```bash
cd /pscratch/sd/s/sbagchi/AutoPF_workflow_integration/examples/closed-loop-aecroscopywave
export AUTOPF_REPO=/pscratch/sd/s/sbagchi/AutoPF_workflow_integration
export NERSC_DEPLOY_ROOT=$SCRATCH/autopf-aecroscopywave
./stage_nersc_perlmutter.sh /path/to/reviewed/deployment_bundle
```

The bundle must contain `calibration/`, `config/`, and `inputs/`. The staging
script copies them to scratch with checksums and installs
`production.nersc-perlmutter.json`. No campaign output is written to CFS.

Run an allocation-only preflight before connecting a microscope:

```bash
salloc -A m5014_g -C gpu -q interactive -N 1 --gpus-per-node=4 -t 00:30:00
AUTOPF_PREFLIGHT_ONLY=1 bash submit_nersc_perlmutter_production.slurm
exit
```

For a live four-node workflow:

```bash
export AFW_CLOUD_SERVER_URL=https://YOUR-AFW-SERVER
export AFW_INSTRUMENT_ID=YOUR-INSTRUMENT
export AFW_AGENT_SECRET='YOUR-SECRET'
export AFW_TILED_API_KEY='YOUR-TILED-KEY'
sbatch submit_nersc_perlmutter_production.slurm
squeue -u "$USER"
sacct -X -j JOB_ID --format=JobID,State,Elapsed,Start,End,ExitCode
```

The batch gate refuses to run unless the application checkout is exactly
`c7809ccee6216ece36b7767c30ba0b2f5cb4f4f6` and its MOOSE framework is
`dd3d84a665`. It records the version banner, Git revisions, executable hash,
and file timestamp under `$NERSC_DEPLOY_ROOT/provenance`. The requested GPU
nodes follow NERSC's `m5014_g` charging path; MOOSE chores use 64 CPU cores per
node and set `MPICH_GPU_SUPPORT_ENABLED=0`. The variational POD--GP selects
CUDA automatically when PyTorch can see an A100; MOOSE itself remains CPU-only
in this profile.
