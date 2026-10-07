# AutoPF closed-loop experiment integration

## Responsibility boundary

The integration deliberately keeps three systems independent:

| Layer | Owns | Does not own |
|---|---|---|
| AEcroscopyWave | instrument safety, job queue, PFM acquisition, raw arrays, Tiled publication | theory posterior or HPC scheduling |
| AutoPF | campaign state, observation contract, low-rank/UQ strategy, experiment and simulation proposals | microscope drivers or MOOSE internals |
| MatEnsemble + MOOSE | asynchronous full-physics evaluations and run directories | instrument control or scientific stopping policy |

```text
AutoPF strategy
  -> AEcroscopyWave job queue
  -> Tiled observation reference
  -> AutoPF POD--GP / posterior update
  -> MatEnsemble MOOSE batch
  -> field summaries and uncertainty update
  -> next experiment request
```

## Durable iteration protocol

`ClosedLoopRunner` checkpoints these states atomically:

1. experiment planned;
2. acquisition submitted, including the remote AEcroscopyWave job ID;
3. observation collected, including Tiled references;
4. simulation batch planned;
5. simulation batch submitted;
6. simulations collected;
7. surrogate/posterior updated;
8. iteration complete.

If a process stops while waiting for a scan, the next invocation polls the
saved job ID instead of enqueuing the same microscope action again. The same
pattern applies to theory batches.

The AEcroscopyWave 0.1.21 enqueue endpoint creates the job ID server-side and
does not accept a client idempotency key. There remains a narrow failure window
if the server accepts an enqueue request but the client dies before receiving
and checkpointing the response. A live deployment should add a client request
ID/idempotency key to AEcroscopyWave for strict exactly-once instrument actions.
AutoPF already supplies a stable `request_id` for this extension.

## Observation contract

An AutoPF observation contains small JSON-safe values in `data`, large-array
or artifact locators in `data_references`, and acquisition context in
`metadata`.

AEcroscopyWave publishes scan arrays under the job's Tiled node and returns
`tiled_uri`, `tiled_run_key`, and `tiled_array_keys`. `TiledArrayResolver`
loads a selected array lazily. The experiment-specific PFM calibration remains
an explicit transform; raw amplitude must not silently be described as
physical displacement.

## Instrument workflow granularity

The built-in AEcroscopyWave `pfm_image` job acquires and publishes one frame.
If an AutoPF iteration requires a pulse, tip move, and post-pulse image, expose
that sequence as one reviewed AEcroscopyWave executor or named workflow that
returns and publishes the final arrays. AutoPF should enqueue that single job
type. This preserves the instrument-side safety boundary and gives the theory
loop one durable remote handle for the complete experimental action.

## Theory strategy contract

`CallbackStrategy` connects an existing model through three functions:

- `propose_experiment(state, iteration)`;
- `propose_simulations(state, observation)`;
- `update(state, observation, simulations)`.

The migrated `autopf.online_strategy` and `autopf.posterior` modules retain the
campaign's finite-ensemble posterior, condition selection, trust-region/LHS
candidate generation, and resource-budget helpers. The publication POD--GP and
posterior-aware Thompson implementation can be placed behind the callbacks
without coupling it to AEcroscopyWave or MatEnsemble APIs.

Only JSON-safe strategy state should be checkpointed. Large model tensors and
POD bases should be saved as versioned files, with their paths and hashes stored
in `strategy_state`.

## Production asynchronous Thompson path

`examples/closed-loop-aecroscopywave/run_production.py` is the non-proxy entry
point. `MatEnsembleAsyncTSBackend` launches an initial space-filling MOOSE wave.
For every completed field, a MatEnsemble strategy chore calls AutoPF's
file-backed controller, which:

1. acquires a lock and loads the latest GPyTorch variational POD--GP artifact;
2. warm-starts the variational posterior on the returned full field;
3. uses BoTorch to draw and score a joint reparameterized posterior sample over
   all available candidates with the frozen objective;
4. checkpoints state and emits one replacement MOOSE chore;
5. releases the lock while other solves continue.

This is asynchronous Thompson sampling, not merely asynchronous execution of a
fixed design. The task wrapper must return the simulation ID and extracted
field reference so the controller can make the update idempotent.

Two objective policies are registered. `pixel_nmse` minimizes aligned,
variance-normalized residuals. `posterior_distance` minimizes a Gaussian
negative log predictive density in a calibration-only feature basis, combining
the noise-aware Mahalanobis term and the covariance log determinant. The mode
is stored in campaign metadata and cannot change on restart. An agent hook is a
data-only JSON choice between these reviewed policies.

## Simulation contract

`MatEnsembleMOOSEBackend` maps a batch of `SimulationRequest` objects to a fixed
MOOSE input, per-request command-line overrides, isolated run directories, and
JSON summaries or portable field paths returned by a result loader.

The batch specification is written before execution. A completed receipt makes
collection idempotent, while MatEnsemble restart files remain available for
site-level recovery.

## Security and deployment

- Read `AFW_AGENT_SECRET` and Tiled credentials from the environment or a
  secret manager; never place them in campaign JSON.
- Run the microscope agent on the instrument PC and `afw-server`/Tiled on the
  protected lab or cloud host.
- Put AutoPF on a compute-side host that can reach `afw-server` and Tiled.
- Keep all hardware limits and human approval gates in AEcroscopyWave.
- Keep MOOSE executables, input files, and large run data in HPC scratch.
