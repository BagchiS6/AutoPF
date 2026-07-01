# Low-rank POD/SVD and GP surrogate code

This folder contains the surrogate-learning pieces of the inverse-problem
workflow. These scripts are separated from the MOOSE/AutoPF launch scripts so
the reduced-order modeling step is easy to find.

The surrogate stage primarily learns the field response to `g11`, `g12`, and
`g44` across voltage/pulse conditions. The broader example then uses the
selected `g_ij` baseline to run residual-guided hidden-physics refinement in
MOOSE/Ferret.

## Main scripts

- `train_uz_svd_surrogate.py`
  - Builds a low-rank POD/SVD representation of simulated surface `u_z`
    (`disp_z` in the saved fields).
  - Supports two layouts:
    - `candidate-grid`: `[g11, g12, g44] -> full condition grid`
    - `condition-field`: `[g11, g12, g44, voltage, pulse_width] -> one surface`
  - Fits a NumPy/ridge coefficient model for retained SVD coefficients.

- `train_uz_gp_surrogate.py`
  - Reuses the POD/SVD preprocessing from `train_uz_svd_surrogate.py`.
  - Trains a variational multitask GPyTorch GP in SVD-coefficient space.
  - Provides predictive uncertainty in the low-rank coefficient space.

- `propose_condition_gp_batch.py`
  - Uses the trained condition-aware GP/POD surrogate to propose new
    candidate `g_ij` batches for active learning.

- `run_bo_loop.py`
  - Earlier scalar BO driver using GP/Thompson-sampling style candidate
    selection.

- `make_corrected_ic_dense_inverse_posterior.py`
  - Dense posterior/inversion script using a small POD + RBF-kernel surrogate
    over corrected-IC anchor fields.

- `make_latest_residual_svd_analysis.py`
  - Residual SVD diagnostic used to quantify low-rank residual structure after
    hidden-physics fitting.
  - Helps determine whether remaining PFM mismatch is structured enough to
    justify scalar, spatial, anisotropic, or flexo-proxy hidden-physics
    candidates.

## Representative trained artifacts

The representative trained surrogate state is stored in:

```text
../../data/surrogate_gp_pod/gp_surrogate_combined/
```

Important files:

- `condition_field_gp_state.pt`: trained GPyTorch variational multitask GP.
- `condition_field_preprocess.npz`: POD/SVD preprocessing arrays and scalers.
- `condition_field_summary.json`: training summary, retained component count,
  explained variance, source campaign summaries, and GP training history.

The active-learning proposal JSONs are stored in:

```text
../../data/surrogate_gp_pod/condition_gp_active/
```

## Conceptual mapping

The condition-aware surrogate learns:

```text
[g11, g12, g44, V, tau] -> POD/SVD coefficients -> reconstructed surface u_z
```

The first inverse stage searches candidate `g_ij` values by comparing
reconstructed/simulated `u_z` fields with PFM-derived experimental targets.

## Hidden-physics refinement boundary

Hidden-physics refinement is downstream of the surrogate fit. It uses the best
or posterior-guided `g_ij` point as a baseline and evaluates additional MOOSE
campaigns with controls such as:

```text
screen_lambda
spatial_screen_amp
vegard_strain
spatial_vegard_amp
anis_vegard_*_amp
flexo_proxy_*
```

Those runs are scored against the same PFM field targets and validated on
holdout voltage/pulse conditions. See the top-level example `README.md` and
`DATA_DICTIONARY.md` for the hidden-physics campaign structure and final
anisotropic refinement parameters.
