#!/bin/bash
set -euo pipefail

campaign_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$campaign_dir"
mkdir -p logs

export CAMPAIGN_DIR="$campaign_dir"
export INPUT="$campaign_dir/BTO_DW_spatial_hidden_physics_zdecay_ic.i"
export RUN_DIR="$campaign_dir/presentation_refined_experimental_ic_relaxation_run"
export FILE_BASE="out_refined_experimental_ic_relaxation"
export NTASKS="${NTASKS:-64}"
export SRUN_EXTRA="${SRUN_EXTRA:---exclusive -N 1}"

# Corrected-IC showcase condition from the training grid.
export TIP_VOLTAGE="${TIP_VOLTAGE:-7.8}"
export PULSE_END="${PULSE_END:-0.81}"
export PFM_IMAGE_FILE="${PFM_IMAGE_FILE:-$campaign_dir/experimental/voltage_-7.8V_pulsewidth_0.81s_polar_z.txt}"

# Refine z relative to the production corrected-IC campaign (Nz=10).
export NX="${NX:-40}"
export NY="${NY:-40}"
export NZ="${NZ:-24}"

# Let the pulse complete, then allow the energy-change gate to terminate.
export SIM_END="${SIM_END:-5.0}"
export RELAX_START_TIME="${RELAX_START_TIME:-1.10}"
export ENERGY_TOL="${ENERGY_TOL:-5e-4}"

# Current-best corrected-IC spatial hidden-physics point.
export G11="${G11:-1.0}"
export G12="${G12:--0.02}"
export G44="${G44:-0.02}"
export SCREEN_LAMBDA="${SCREEN_LAMBDA:-0.0}"
export VEGARD_STRAIN="${VEGARD_STRAIN:-0.0}"
export SPATIAL_SCREEN_AMP="${SPATIAL_SCREEN_AMP:-0.0225}"
export SPATIAL_SCREEN_SIGMA_XY="${SPATIAL_SCREEN_SIGMA_XY:-120.0}"
export SPATIAL_SCREEN_SIGMA_Z="${SPATIAL_SCREEN_SIGMA_Z:-35.0}"
export SPATIAL_SCREEN_X0="${SPATIAL_SCREEN_X0:-0.0}"
export SPATIAL_SCREEN_Y0="${SPATIAL_SCREEN_Y0:-0.0}"
export SPATIAL_SCREEN_Z0="${SPATIAL_SCREEN_Z0:-100.0}"
export SPATIAL_VEGARD_AMP="${SPATIAL_VEGARD_AMP:-0.0}"
export SPATIAL_VEGARD_SIGMA_XY="${SPATIAL_VEGARD_SIGMA_XY:-120.0}"
export SPATIAL_VEGARD_SIGMA_Z="${SPATIAL_VEGARD_SIGMA_Z:-52.5}"
export SPATIAL_VEGARD_X0="${SPATIAL_VEGARD_X0:-0.0}"
export SPATIAL_VEGARD_Y0="${SPATIAL_VEGARD_Y0:-7.0}"
export SPATIAL_VEGARD_Z0="${SPATIAL_VEGARD_Z0:-100.0}"

# PFM-weighted ellipse/cap inferred for voltage_-7.8V_pulsewidth_0.81s.
export IC_X0="${IC_X0:-2.5349602436526104}"
export IC_Y0="${IC_Y0:-3.3716127037579304}"
export IC_RX="${IC_RX:-124.17759954854296}"
export IC_RY="${IC_RY:-90.92576621114381}"
export IC_RZ="${IC_RZ:-75.21647832223316}"
export IC_COS_THETA="${IC_COS_THETA:--0.9828244588370582}"
export IC_SIN_THETA="${IC_SIN_THETA:-0.18454290317333802}"
export IC_AMP_DECAY_NM="${IC_AMP_DECAY_NM:-120.0}"
export IC_ELLIPSOID_SMOOTH="${IC_ELLIPSOID_SMOOTH:-0.18}"

export PHASE_METRIC="${PHASE_METRIC:-positive}"
export GIF_STEM="phasefield_refined_experimental_ic_relaxation"
export GIF_TITLE="PFM-initialized corrected-IC relaxation: refined z mesh (-7.8 V, 0.81 s)"
export GIF_MAX_FRAMES="${GIF_MAX_FRAMES:-16}"
export GIF_DURATION="${GIF_DURATION:-0.55}"

log="$campaign_dir/logs/refined_experimental_ic_relaxation.log"
bash "$campaign_dir/perlmutter_presentation_seed_growth.sh" > "$log" 2>&1

"${AUTOPF_ENV:-/global/cfs/cdirs/m5014/PhaseField/autopf_env}/bin/python" \
  "$campaign_dir/paper_figures/make_refined_experimental_ic_relaxation_panel.py" \
  --run-dir "$RUN_DIR" \
  --file-base "$FILE_BASE" \
  --stem "fig_refined_experimental_ic_relaxation_xz_energy" \
  --title "PFM-initialized corrected-IC relaxation: refined z mesh (-7.8 V, 0.81 s)" \
  --target-times 0.0 0.81 1.3 2.5 5.0 \
  --phase-sign positive
