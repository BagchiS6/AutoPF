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

## Connect the production POD–GP and asynchronous TS

`example_strategy.make_strategy()` is a transparent scalar-UQ demonstration,
not the publication surrogate. Replace its three callbacks with the production
functions:

1. `propose_experiment(state, iteration)` chooses the next microscopy
   condition from the experimental acquisition policy;
2. `propose_simulations(state, observation)` draws posterior-aware Thompson
   candidates and returns MOOSE parameter requests;
3. `update(state, observation, simulations)` performs the online low-rank
   POD–GP update and returns JSON-safe posterior state.

The orchestration, recovery, and instrument contracts do not change.

## Connect MOOSE/MatEnsemble

Replace `ProxySimulationBackend` with:

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
