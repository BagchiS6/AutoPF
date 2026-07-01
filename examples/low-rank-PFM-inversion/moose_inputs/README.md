# MOOSE input

This Zenodo package keeps a single canonical corrected-IC MOOSE/Ferret input:

```text
BTO_DW_anisotropic_hidden_physics_zdecay_ic.i
```

This is the final input used for the representative learned-physics results in
the package. It is a superset model for both the gradient-coefficient inversion
and the residual-guided hidden-physics refinement:

- Baseline / hidden-off runs are obtained by setting all hidden-physics
  amplitudes to zero. These runs are used to fit or validate `g11`, `g12`, and
  `g44`.
- Spatial screening, isotropic Vegard/eigenstrain, anisotropic eigenstrain, and
  flexo-proxy comparison terms are activated through command-line parameters.
- Hidden-physics refinement runs keep the selected `g_ij` point fixed or
  tightly constrained while sweeping hidden controls such as `screen_lambda`,
  `spatial_screen_amp`, `spatial_vegard_amp`, `anis_vegard_*_amp`, and
  `flexo_proxy_*`.
- The final included representative run used the anisotropic hidden-physics
  setting documented in the top-level `DATA_DICTIONARY.md`.
