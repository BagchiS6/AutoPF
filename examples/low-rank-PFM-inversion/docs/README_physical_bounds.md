# Bulk-BTO Campaign

This campaign is cloned from `sim_bo_larger_bounds` and keeps the first-stage
active search inside the trained gradient-coefficient box around bulk BTO:

```text
g11 = 0.5
g12 = -0.02
g44 = 0.02
```

The BO parameter box in `sim_bo_config.json` is:

```text
g11: [0.3, 1.0]
g12: [-0.1, -0.001]
g44: [0.001, 0.1]
```

## Interactive anchor run

The interactive config `sim_bo_config_interactive.json` evaluates six fixed
anchor candidates across six representative voltage/pulse conditions, for
36 MOOSE jobs total. The anchors include the previous inverse-boundary point,
bulk BTO, and one-axis perturbations around bulk BTO. This is intended for a
4-node, 4-hour Perlmutter interactive allocation.

Run from inside an allocation:

```bash
./perlmutter_anchor_campaign.sh
```

The output summary will be:

```text
anchor_results.json
round_bulk_bto_anchors/
```

## 4-node BO run

For a small physical-bounds BO campaign on 4 nodes:

```bash
./perlmutter_physical_bo_4node.sh
```

This uses `sim_bo_config.json` with 12 rounds and batch size 3 over the full
29-condition training set. It is a starter configuration for additional
bulk-BTO-centered exploration. This stage searches `g11`, `g12`, and `g44`;
it does not by itself complete the hidden-physics refinement.

## Condition-aware variational GP active cycle

This is the preferred active-learning path for the inverse problem. It combines
the old `sim_bo_larger_bounds` records with any new records in this campaign,
trains the condition-aware variational multitask GP,

```text
(g11, g12, g44, voltage, pulse_width) -> SVD coefficients -> uz
```

and proposes a new candidate batch using

```text
A(g) = predicted_mismatch(g) - kappa * predicted_uncertainty(g)
```

Lower acquisition values are selected. Use `kappa=0` for exploitation and
larger `kappa` for exploration.

Build a proposal manifest without running it:

```bash
./perlmutter_condition_gp_active_cycle.sh
```

Train the GP, propose candidates, and immediately run the proposed batch:

```bash
RUN_PROPOSED=1 ./perlmutter_condition_gp_active_cycle.sh
```

The main outputs are:

```text
gp_surrogate_combined/
condition_gp_active/gp_active_000_proposal.json
round_gp_active_000/
```

## Residual-guided hidden-physics refinement

After selecting a posterior-guided `g_ij` point, the campaign continues with a
second-stage refinement over hidden-physics controls. This stage uses the PFM
residual fields left by the gradient-only model to test scalar screening,
isotropic Vegard/eigenstrain, localized spatial screening, anisotropic
eigenstrain, and flexo-proxy comparison candidates.

The staged workflow pattern is:

```text
g_ij anchor / dense posterior
  -> residual sensitivity
  -> scalar hidden anchors
  -> spatial hidden-field pilots
  -> local refinement rounds
  -> holdout validation
  -> hidden-physics report
```

Representative outputs are documented in the top-level `DATA_DICTIONARY.md`
under `data/hidden_physics_analysis/` and
`data/final_anisotropic_relaxation/`.
