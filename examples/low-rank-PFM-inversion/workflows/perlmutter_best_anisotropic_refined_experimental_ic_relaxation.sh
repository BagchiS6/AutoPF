#!/bin/bash
#SBATCH -A m5064_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -J bto_bestani_si
#SBATCH -o logs/bto_bestani_si_%j.out
#SBATCH -e logs/bto_bestani_si_%j.err

set -euo pipefail

campaign_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$campaign_dir"
mkdir -p logs

export CAMPAIGN_DIR="$campaign_dir"
export INPUT="$campaign_dir/BTO_DW_anisotropic_hidden_physics_zdecay_ic.i"
export RUN_DIR="$campaign_dir/paper_best_anisotropic_refined_experimental_ic_relaxation_run"
export FILE_BASE="out_best_anisotropic_refined_experimental_ic_relaxation"
export NTASKS="${NTASKS:-64}"
export SRUN_EXTRA="${SRUN_EXTRA:---exclusive -N 1}"

# Representative training-grid condition used for the SI seed-relaxation panel.
export TIP_VOLTAGE="${TIP_VOLTAGE:-7.8}"
export PULSE_END="${PULSE_END:-0.81}"
export PFM_IMAGE_FILE="${PFM_IMAGE_FILE:-$campaign_dir/experimental/voltage_-7.8V_pulsewidth_0.81s_polar_z.txt}"

# Refined z mesh for publication visualization.
export NX="${NX:-40}"
export NY="${NY:-40}"
export NZ="${NZ:-24}"

# Let the pulse complete, then allow the free-energy-change terminator to stop.
export SIM_END="${SIM_END:-5.0}"
export RELAX_START_TIME="${RELAX_START_TIME:-1.10}"
export ENERGY_TOL="${ENERGY_TOL:-5e-4}"

# Best anisotropic hidden-physics candidate from anisotropic_holdout_full_000.
export G11="${G11:-0.5}"
export G12="${G12:--0.06}"
export G44="${G44:-0.02}"
export SCREEN_LAMBDA="${SCREEN_LAMBDA:-0.0}"
export VEGARD_STRAIN="${VEGARD_STRAIN:-0.0}"
export SPATIAL_SCREEN_AMP="${SPATIAL_SCREEN_AMP:-0.08}"
export SPATIAL_SCREEN_SIGMA_XY="${SPATIAL_SCREEN_SIGMA_XY:-80.0}"
export SPATIAL_SCREEN_SIGMA_Z="${SPATIAL_SCREEN_SIGMA_Z:-25.0}"
export SPATIAL_SCREEN_X0="${SPATIAL_SCREEN_X0:-0.0}"
export SPATIAL_SCREEN_Y0="${SPATIAL_SCREEN_Y0:-0.0}"
export SPATIAL_SCREEN_Z0="${SPATIAL_SCREEN_Z0:-100.0}"
export SPATIAL_VEGARD_AMP="${SPATIAL_VEGARD_AMP:-0.0}"
export SPATIAL_VEGARD_SIGMA_XY="${SPATIAL_VEGARD_SIGMA_XY:-120.0}"
export SPATIAL_VEGARD_SIGMA_Z="${SPATIAL_VEGARD_SIGMA_Z:-35.0}"
export SPATIAL_VEGARD_X0="${SPATIAL_VEGARD_X0:-0.0}"
export SPATIAL_VEGARD_Y0="${SPATIAL_VEGARD_Y0:-0.0}"
export SPATIAL_VEGARD_Z0="${SPATIAL_VEGARD_Z0:-100.0}"
export ANIS_VEGARD_SIGMA_XY="${ANIS_VEGARD_SIGMA_XY:-120.0}"
export ANIS_VEGARD_SIGMA_Z="${ANIS_VEGARD_SIGMA_Z:-35.0}"
export ANIS_VEGARD_X0="${ANIS_VEGARD_X0:-0.0}"
export ANIS_VEGARD_Y0="${ANIS_VEGARD_Y0:-0.0}"
export ANIS_VEGARD_Z0="${ANIS_VEGARD_Z0:-100.0}"
export ANIS_VEGARD_XX_AMP="${ANIS_VEGARD_XX_AMP:-0.0}"
export ANIS_VEGARD_YY_AMP="${ANIS_VEGARD_YY_AMP:-0.0}"
export ANIS_VEGARD_ZZ_AMP="${ANIS_VEGARD_ZZ_AMP:-0.0}"
export ANIS_VEGARD_XY_AMP="${ANIS_VEGARD_XY_AMP:-0.0}"
export ANIS_VEGARD_XZ_AMP="${ANIS_VEGARD_XZ_AMP:-0.0}"
export ANIS_VEGARD_YZ_AMP="${ANIS_VEGARD_YZ_AMP:-0.00075}"
export FLEXO_PROXY_AMP="${FLEXO_PROXY_AMP:-0.0}"
export FLEXO_PROXY_GRAD_AMP="${FLEXO_PROXY_GRAD_AMP:-0.0}"
export FLEXO_PROXY_SIGMA_XY="${FLEXO_PROXY_SIGMA_XY:-80.0}"
export FLEXO_PROXY_SIGMA_Z="${FLEXO_PROXY_SIGMA_Z:-25.0}"
export FLEXO_PROXY_X0="${FLEXO_PROXY_X0:-0.0}"
export FLEXO_PROXY_Y0="${FLEXO_PROXY_Y0:-0.0}"
export FLEXO_PROXY_Z0="${FLEXO_PROXY_Z0:-100.0}"

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
export GIF_STEM="phasefield_best_anisotropic_refined_experimental_ic_relaxation"
export GIF_TITLE="Best anisotropic hidden-physics model: refined corrected-IC relaxation (-7.8 V, 0.81 s)"
export GIF_MAX_FRAMES="${GIF_MAX_FRAMES:-16}"
export GIF_DURATION="${GIF_DURATION:-0.55}"

job_tag="${SLURM_JOB_ID:-local}"
log="$campaign_dir/logs/best_anisotropic_refined_experimental_ic_relaxation_${job_tag}.log"
bash "$campaign_dir/perlmutter_presentation_seed_growth.sh" > "$log" 2>&1

"${AUTOPF_ENV:-/global/cfs/cdirs/m5014/PhaseField/autopf_env}/bin/python" \
  "$campaign_dir/paper_figures/make_si_refined_experimental_ic_relaxation_contours.py" \
  --run-dir "$RUN_DIR" \
  --file-base "$FILE_BASE" \
  --stem "fig_si_best_anisotropic_refined_experimental_ic_relaxation_contours" \
  --target-times 0.0 0.2 0.6125 1.2875 2.4875 5.0

echo "Best-anisotropic refined SI figure written to:"
echo "  $campaign_dir/paper_figures/generated/fig_si_best_anisotropic_refined_experimental_ic_relaxation_contours.pdf"
