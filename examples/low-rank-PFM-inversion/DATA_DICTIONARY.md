# Data dictionary

## Experimental PFM grids

Location: `data/experimental_pfm/`

Files are named as:

```text
voltage_<V>V_pulsewidth_<tau>s_polar_z.txt
```

Each file is a two-dimensional PFM-derived surface target grid for one
bias-voltage and pulse-width condition. These grids are used as the experimental
targets for inverse fitting and holdout validation.

## Baseline inverse posterior

Location: `data/baseline_inverse_posterior/`

Important files:

- `candidate_scores.csv`: dense candidate table for scalar gradient-energy
  coefficients. Columns include `g11`, `g12`, `g44`, `mse`, and/or normalized
  loss values depending on the analysis script.
- `posterior_summary.json`: summary of scalar inverse-posterior results.
- `observed_positive_rounds_transpose.json`: observed MOOSE candidate results
  after corrected orientation/transposition handling.
- `loss_profiles.png`, `loss_pair_projections.png`, `posterior_marginals.png`:
  diagnostic plots for the scalar-gradient posterior.

The baseline scalar model solves for the gradient-energy coefficients
`g11`, `g12`, and `g44` while leaving hidden-physics controls off.

## POD and GP surrogate

Location: `data/surrogate_gp_pod/`

The condition-aware surrogate uses low-rank POD compression of simulated
surface displacement fields. The learned map is:

```text
[g11, g12, g44, V, tau] -> retained POD coefficients -> reconstructed surface u_z
```

Important files:

- `gp_surrogate_combined/condition_field_gp_state.pt`: trained variational
  multitask GPyTorch GP state.
- `gp_surrogate_combined/condition_field_preprocess.npz`: POD/SVD basis,
  scalers, normalization information, and coefficient preprocessing arrays.
- `gp_surrogate_combined/condition_field_summary.json`: source campaign
  summaries, retained POD/SVD modes, explained variance, GP training history,
  and model metadata.
- `condition_gp_active/gp_active_bulk_*.json`: active-learning batch proposals
  generated from the GP/POD surrogate.
- `condition_gp_active/gp_active_transpose_*.json`: transpose-corrected active
  candidates/validation proposals.

Representative summary:

```text
field = disp_z
dataset sample count = 4769 surface fields
condition count = 36
retained POD/SVD components = 25
GP library = torch/gpytorch
```

Relevant code is in:

```text
code/surrogate_low_rank/
```

## Hidden-physics analysis

Location: `data/hidden_physics_analysis/`

Important files:

- `anisotropic_hidden_analysis/anisotropic_holdout_full_000/candidate_ranking.json`:
  ranking of hidden-physics candidates on the holdout set.
- `anisotropic_hidden_analysis/anisotropic_holdout_full_000/condition_metrics.csv`:
  per-condition normalized MSE and related diagnostics.
- `anisotropic_hidden_analysis/latest_hidden_physics_residual_svd_summary.json`:
  residual SVD summary used to evaluate low-rank residual structure.
- `anisotropic_hidden_analysis/fig_candidate_basis_function_comparison.*`:
  comparison of hidden-field basis choices.
- `refinement_rounds/`: scalar/spatial hidden-physics refinement rankings from
  the previous and corrected-IC guided campaigns.

Hidden-physics parameters include:

- `screen_lambda`: scalar depolarization/screening proxy.
- `spatial_screen_amp`: amplitude of a localized screening field.
- `vegard_strain`: scalar isotropic eigenstrain/Vegard proxy.
- `spatial_vegard_amp`: localized isotropic Vegard/eigenstrain amplitude.
- `anis_vegard_*_amp`: anisotropic eigenstrain amplitudes. The final selected
  candidate used `anis_vegard_yz_amp = 0.00075`.
- `flexo_proxy_*`: phenomenological flexoelectric-like proxy terms used as
  comparison candidates, not as a rigorous flexoelectric kernel.

## Final anisotropic relaxation trajectory

Location: `data/final_anisotropic_relaxation/`

Important files:

- `out_best_anisotropic_refined_experimental_ic_relaxation.e`: Exodus output
  with full fields through the final refined corrected-IC relaxation.
- `out_best_anisotropic_refined_experimental_ic_relaxation.csv`: scalar
  postprocessors from the same run.
- `best_anisotropic_refined_experimental_ic_relaxation_55121418.log`: full
  MOOSE run log from the interactive allocation.

Representative condition:

```text
tip_voltage = 7.8
pulse_end = 0.81
Nx = 40
Ny = 40
Nz = 24
sim_end = 5.0
```

Final model parameters:

```text
g11 = 0.5
g12 = -0.06
g44 = 0.02
screen_lambda = 0.0
spatial_screen_amp = 0.08
spatial_screen_sigma_xy = 80.0
spatial_screen_sigma_z = 25.0
vegard_strain = 0.0
spatial_vegard_amp = 0.0
anis_vegard_yz_amp = 0.00075
anis_vegard_sigma_xy = 120.0
anis_vegard_sigma_z = 35.0
```

Fields commonly used from the Exodus file:

- `polar_x`, `polar_y`, `polar_z`: polarization components.
- `potential_E_int`: electrostatic potential.
- `u_x`, `u_y`, `u_z`: displacement solution variables.
- `disp_x`, `disp_y`, `disp_z`: global-displacement auxiliary outputs.
- `e00`, `e11`, `e22`: strain tensor diagonal components.
- `s00`, `s11`, `s22`: stress tensor diagonal components.

CSV postprocessors include:

- `Fb`: bulk Landau free energy contribution.
- `Fw`: wall/gradient energy contribution.
- `Fela`: elastic energy contribution.
- `Fele`: electrostatic energy contribution.
- `Fc`: electrostrictive coupling contribution.
- `Ftot`: reported total diagnostic used for relaxation monitoring.
- `perc_change`: relative change diagnostic used by the energy terminator.

The final run reached `t = 5.0` with `perc_change = 3.691513e-04`, below the
requested `energy_tol = 5e-4`.
