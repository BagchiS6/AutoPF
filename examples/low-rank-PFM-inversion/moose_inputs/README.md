# MOOSE input

This Zenodo package keeps a single canonical corrected-IC MOOSE/Ferret input:

```text
BTO_DW_anisotropic_hidden_physics_zdecay_ic.i
```

This is the final input used for the representative learned-physics results in
the package. It is a superset model:

- Baseline / hidden-off runs are obtained by setting all hidden-physics
  amplitudes to zero.
- Spatial screening, isotropic Vegard/eigenstrain, anisotropic eigenstrain, and
  flexo-proxy comparison terms are activated through command-line parameters.
- The final included representative run used the anisotropic hidden-physics
  setting documented in the top-level `DATA_DICTIONARY.md`.

