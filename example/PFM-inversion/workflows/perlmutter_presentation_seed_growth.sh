#!/bin/bash
#SBATCH -A m5064_g
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 01:00:00
#SBATCH -N 1
#SBATCH -J bto_seedgif
#SBATCH -o logs/bto_seedgif_%j.out
#SBATCH -e logs/bto_seedgif_%j.err

set -euo pipefail

export MPICH_GPU_SUPPORT_ENABLED=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-simbo}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/tmp}"

AUTOPF_ENV="${AUTOPF_ENV:-/global/cfs/cdirs/m5014/PhaseField/autopf_env}"
PYTHON="${AUTOPF_ENV}/bin/python"
EXECUTABLE="${EXECUTABLE:-/global/cfs/cdirs/m5014/PhaseField/MOOSE/projects/ferret/ferret-opt}"

if [[ -n "${CAMPAIGN_DIR:-}" ]]; then
  campaign_dir="$CAMPAIGN_DIR"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/BTO_DW_presentation_seed_growth.i" ]]; then
  campaign_dir="$SLURM_SUBMIT_DIR"
else
  campaign_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

cd "$campaign_dir"
mkdir -p logs

INPUT="${INPUT:-$campaign_dir/BTO_DW_presentation_seed_growth.i}"
RUN_DIR="${RUN_DIR:-$campaign_dir/presentation_seed_growth_run}"
NTASKS="${NTASKS:-64}"
TIP_VOLTAGE="${TIP_VOLTAGE:-12}"
PULSE_END="${PULSE_END:-1.0}"
SIM_END="${SIM_END:-0.8}"
FILE_BASE="${FILE_BASE:-out_seed_growth}"
GIF_STEM="${GIF_STEM:-phasefield_seed_growth_current_best}"
GIF_TITLE="${GIF_TITLE:-Current-best model: high-bias growth from compact nucleus}"
PHASE_METRIC="${PHASE_METRIC:-negative}"
SRUN_EXTRA="${SRUN_EXTRA:-}"
NX="${NX:-}"
NY="${NY:-}"
NZ="${NZ:-}"
G11="${G11:-}"
G12="${G12:-}"
G44="${G44:-}"
PFM_IMAGE_FILE="${PFM_IMAGE_FILE:-}"
IC_X0="${IC_X0:-}"
IC_Y0="${IC_Y0:-}"
IC_RX="${IC_RX:-}"
IC_RY="${IC_RY:-}"
IC_RZ="${IC_RZ:-}"
IC_COS_THETA="${IC_COS_THETA:-}"
IC_SIN_THETA="${IC_SIN_THETA:-}"
IC_AMP_DECAY_NM="${IC_AMP_DECAY_NM:-}"
IC_ELLIPSOID_SMOOTH="${IC_ELLIPSOID_SMOOTH:-}"
TIP_RADIUS="${TIP_RADIUS:-}"
SEED_RADIUS_XY="${SEED_RADIUS_XY:-}"
SEED_RADIUS_Z="${SEED_RADIUS_Z:-}"
SEED_Z0="${SEED_Z0:-}"
SEED_INTERFACE_WIDTH="${SEED_INTERFACE_WIDTH:-}"
SEED_INSIDE_PZ="${SEED_INSIDE_PZ:-}"
SEED_OUTSIDE_PZ="${SEED_OUTSIDE_PZ:-}"
SCREEN_LAMBDA="${SCREEN_LAMBDA:-}"
SCREEN_PERMITTIVITY="${SCREEN_PERMITTIVITY:-}"
ENERGY_TOL="${ENERGY_TOL:-}"
RELAX_START_TIME="${RELAX_START_TIME:-}"
SPATIAL_SCREEN_AMP="${SPATIAL_SCREEN_AMP:-}"
SPATIAL_SCREEN_SIGMA_XY="${SPATIAL_SCREEN_SIGMA_XY:-}"
SPATIAL_SCREEN_SIGMA_Z="${SPATIAL_SCREEN_SIGMA_Z:-}"
SPATIAL_SCREEN_X0="${SPATIAL_SCREEN_X0:-}"
SPATIAL_SCREEN_Y0="${SPATIAL_SCREEN_Y0:-}"
SPATIAL_SCREEN_Z0="${SPATIAL_SCREEN_Z0:-}"
VEGARD_STRAIN="${VEGARD_STRAIN:-}"
SPATIAL_VEGARD_AMP="${SPATIAL_VEGARD_AMP:-}"
SPATIAL_VEGARD_SIGMA_XY="${SPATIAL_VEGARD_SIGMA_XY:-}"
SPATIAL_VEGARD_SIGMA_Z="${SPATIAL_VEGARD_SIGMA_Z:-}"
SPATIAL_VEGARD_X0="${SPATIAL_VEGARD_X0:-}"
SPATIAL_VEGARD_Y0="${SPATIAL_VEGARD_Y0:-}"
SPATIAL_VEGARD_Z0="${SPATIAL_VEGARD_Z0:-}"
ANIS_VEGARD_SIGMA_XY="${ANIS_VEGARD_SIGMA_XY:-}"
ANIS_VEGARD_SIGMA_Z="${ANIS_VEGARD_SIGMA_Z:-}"
ANIS_VEGARD_X0="${ANIS_VEGARD_X0:-}"
ANIS_VEGARD_Y0="${ANIS_VEGARD_Y0:-}"
ANIS_VEGARD_Z0="${ANIS_VEGARD_Z0:-}"
ANIS_VEGARD_XX_AMP="${ANIS_VEGARD_XX_AMP:-}"
ANIS_VEGARD_YY_AMP="${ANIS_VEGARD_YY_AMP:-}"
ANIS_VEGARD_ZZ_AMP="${ANIS_VEGARD_ZZ_AMP:-}"
ANIS_VEGARD_XY_AMP="${ANIS_VEGARD_XY_AMP:-}"
ANIS_VEGARD_XZ_AMP="${ANIS_VEGARD_XZ_AMP:-}"
ANIS_VEGARD_YZ_AMP="${ANIS_VEGARD_YZ_AMP:-}"
FLEXO_PROXY_AMP="${FLEXO_PROXY_AMP:-}"
FLEXO_PROXY_GRAD_AMP="${FLEXO_PROXY_GRAD_AMP:-}"
FLEXO_PROXY_SIGMA_XY="${FLEXO_PROXY_SIGMA_XY:-}"
FLEXO_PROXY_SIGMA_Z="${FLEXO_PROXY_SIGMA_Z:-}"
FLEXO_PROXY_X0="${FLEXO_PROXY_X0:-}"
FLEXO_PROXY_Y0="${FLEXO_PROXY_Y0:-}"
FLEXO_PROXY_Z0="${FLEXO_PROXY_Z0:-}"
GIF_DURATION="${GIF_DURATION:-0.54}"
GIF_MAX_FRAMES="${GIF_MAX_FRAMES:-12}"

mkdir -p "$RUN_DIR"
cd "$RUN_DIR"

echo "========================================"
echo "BTO presentation seed-growth run"
echo "Job ID:      ${SLURM_JOB_ID:-local}"
echo "Node list:   ${SLURM_NODELIST:-local}"
echo "Campaign:    $campaign_dir"
echo "Input:       $INPUT"
echo "Run dir:     $RUN_DIR"
echo "Executable:  $EXECUTABLE"
echo "Python:      $PYTHON"
echo "NTASKS:      $NTASKS"
echo "SRUN extra:  ${SRUN_EXTRA:-none}"
echo "tip_voltage: $TIP_VOLTAGE"
echo "pulse_end:   $PULSE_END"
echo "sim_end:     $SIM_END"
echo "file_base:   $FILE_BASE"
if [[ -n "$NX" || -n "$NY" || -n "$NZ" ]]; then
  echo "mesh:        ${NX:-input} x ${NY:-input} x ${NZ:-input}"
fi
if [[ -n "$G11" || -n "$G12" || -n "$G44" ]]; then
  echo "g_ij:        ${G11:-input}, ${G12:-input}, ${G44:-input}"
fi
if [[ -n "$PFM_IMAGE_FILE" ]]; then
  echo "pfm image:   $PFM_IMAGE_FILE"
fi
if [[ -n "$IC_RX" || -n "$IC_RY" || -n "$IC_RZ" ]]; then
  echo "ic center:   ${IC_X0:-input}, ${IC_Y0:-input}"
  echo "ic radii:    ${IC_RX:-input}, ${IC_RY:-input}, ${IC_RZ:-input}"
fi
if [[ -n "$TIP_RADIUS" || -n "$SEED_RADIUS_XY" || -n "$SEED_RADIUS_Z" || -n "$SEED_Z0" ]]; then
  echo "tip_radius:  ${TIP_RADIUS:-input default}"
  echo "seed radius: ${SEED_RADIUS_XY:-input default} lateral, ${SEED_RADIUS_Z:-input default} depth"
  echo "seed_z0:     ${SEED_Z0:-input default}"
fi
if [[ -n "$SEED_INSIDE_PZ" || -n "$SEED_OUTSIDE_PZ" ]]; then
  echo "seed_inside: ${SEED_INSIDE_PZ:-input default}"
  echo "seed_outside:${SEED_OUTSIDE_PZ:-input default}"
fi
if [[ -n "$SCREEN_LAMBDA" || -n "$SPATIAL_SCREEN_AMP" ]]; then
  echo "screen_lambda:      ${SCREEN_LAMBDA:-input default}"
  echo "spatial_screen_amp: ${SPATIAL_SCREEN_AMP:-input default}"
fi
if [[ -n "$VEGARD_STRAIN" || -n "$SPATIAL_VEGARD_AMP" ]]; then
  echo "vegard_strain:      ${VEGARD_STRAIN:-input default}"
  echo "spatial_vegard_amp: ${SPATIAL_VEGARD_AMP:-input default}"
fi
if [[ -n "$ANIS_VEGARD_XX_AMP" || -n "$ANIS_VEGARD_YZ_AMP" || -n "$FLEXO_PROXY_AMP" || -n "$FLEXO_PROXY_GRAD_AMP" ]]; then
  echo "anis_vegard amps:   xx=${ANIS_VEGARD_XX_AMP:-input}, yy=${ANIS_VEGARD_YY_AMP:-input}, zz=${ANIS_VEGARD_ZZ_AMP:-input}, xy=${ANIS_VEGARD_XY_AMP:-input}, xz=${ANIS_VEGARD_XZ_AMP:-input}, yz=${ANIS_VEGARD_YZ_AMP:-input}"
  echo "flexo proxy amps:   amp=${FLEXO_PROXY_AMP:-input}, grad=${FLEXO_PROXY_GRAD_AMP:-input}"
fi
if [[ -n "$ENERGY_TOL" ]]; then
  echo "energy_tol:         $ENERGY_TOL"
fi
if [[ -n "$RELAX_START_TIME" ]]; then
  echo "relax_start_time:   $RELAX_START_TIME"
fi
echo "========================================"

extra_args=()
if [[ -n "$NX" ]]; then
  extra_args+=("Nx=$NX")
fi
if [[ -n "$NY" ]]; then
  extra_args+=("Ny=$NY")
fi
if [[ -n "$NZ" ]]; then
  extra_args+=("Nz=$NZ")
fi
if [[ -n "$G11" ]]; then
  extra_args+=("g11=$G11")
fi
if [[ -n "$G12" ]]; then
  extra_args+=("g12=$G12")
fi
if [[ -n "$G44" ]]; then
  extra_args+=("g44=$G44")
fi
if [[ -n "$PFM_IMAGE_FILE" ]]; then
  extra_args+=("pfm_image_file=$PFM_IMAGE_FILE")
fi
if [[ -n "$IC_X0" ]]; then
  extra_args+=("ic_x0=$IC_X0")
fi
if [[ -n "$IC_Y0" ]]; then
  extra_args+=("ic_y0=$IC_Y0")
fi
if [[ -n "$IC_RX" ]]; then
  extra_args+=("ic_rx=$IC_RX")
fi
if [[ -n "$IC_RY" ]]; then
  extra_args+=("ic_ry=$IC_RY")
fi
if [[ -n "$IC_RZ" ]]; then
  extra_args+=("ic_rz=$IC_RZ")
fi
if [[ -n "$IC_COS_THETA" ]]; then
  extra_args+=("ic_cos_theta=$IC_COS_THETA")
fi
if [[ -n "$IC_SIN_THETA" ]]; then
  extra_args+=("ic_sin_theta=$IC_SIN_THETA")
fi
if [[ -n "$IC_AMP_DECAY_NM" ]]; then
  extra_args+=("ic_amp_decay_nm=$IC_AMP_DECAY_NM")
fi
if [[ -n "$IC_ELLIPSOID_SMOOTH" ]]; then
  extra_args+=("ic_ellipsoid_smooth=$IC_ELLIPSOID_SMOOTH")
fi
if [[ -n "$TIP_RADIUS" ]]; then
  extra_args+=("tip_radius=$TIP_RADIUS")
fi
if [[ -n "$SEED_RADIUS_XY" ]]; then
  extra_args+=("seed_radius_xy=$SEED_RADIUS_XY")
fi
if [[ -n "$SEED_RADIUS_Z" ]]; then
  extra_args+=("seed_radius_z=$SEED_RADIUS_Z")
fi
if [[ -n "$SEED_Z0" ]]; then
  extra_args+=("seed_z0=$SEED_Z0")
fi
if [[ -n "$SEED_INTERFACE_WIDTH" ]]; then
  extra_args+=("seed_interface_width=$SEED_INTERFACE_WIDTH")
fi
if [[ -n "$SEED_INSIDE_PZ" ]]; then
  extra_args+=("seed_inside_pz=$SEED_INSIDE_PZ")
fi
if [[ -n "$SEED_OUTSIDE_PZ" ]]; then
  extra_args+=("seed_outside_pz=$SEED_OUTSIDE_PZ")
fi
if [[ -n "$SCREEN_LAMBDA" ]]; then
  extra_args+=("screen_lambda=$SCREEN_LAMBDA")
fi
if [[ -n "$SCREEN_PERMITTIVITY" ]]; then
  extra_args+=("screen_permitivitty=$SCREEN_PERMITTIVITY")
fi
if [[ -n "$ENERGY_TOL" ]]; then
  extra_args+=("energy_tol=$ENERGY_TOL")
fi
if [[ -n "$RELAX_START_TIME" ]]; then
  extra_args+=("relax_start_time=$RELAX_START_TIME")
fi
if [[ -n "$SPATIAL_SCREEN_AMP" ]]; then
  extra_args+=("spatial_screen_amp=$SPATIAL_SCREEN_AMP")
fi
if [[ -n "$SPATIAL_SCREEN_SIGMA_XY" ]]; then
  extra_args+=("spatial_screen_sigma_xy=$SPATIAL_SCREEN_SIGMA_XY")
fi
if [[ -n "$SPATIAL_SCREEN_SIGMA_Z" ]]; then
  extra_args+=("spatial_screen_sigma_z=$SPATIAL_SCREEN_SIGMA_Z")
fi
if [[ -n "$SPATIAL_SCREEN_X0" ]]; then
  extra_args+=("spatial_screen_x0=$SPATIAL_SCREEN_X0")
fi
if [[ -n "$SPATIAL_SCREEN_Y0" ]]; then
  extra_args+=("spatial_screen_y0=$SPATIAL_SCREEN_Y0")
fi
if [[ -n "$SPATIAL_SCREEN_Z0" ]]; then
  extra_args+=("spatial_screen_z0=$SPATIAL_SCREEN_Z0")
fi
if [[ -n "$VEGARD_STRAIN" ]]; then
  extra_args+=("vegard_strain=$VEGARD_STRAIN")
fi
if [[ -n "$SPATIAL_VEGARD_AMP" ]]; then
  extra_args+=("spatial_vegard_amp=$SPATIAL_VEGARD_AMP")
fi
if [[ -n "$SPATIAL_VEGARD_SIGMA_XY" ]]; then
  extra_args+=("spatial_vegard_sigma_xy=$SPATIAL_VEGARD_SIGMA_XY")
fi
if [[ -n "$SPATIAL_VEGARD_SIGMA_Z" ]]; then
  extra_args+=("spatial_vegard_sigma_z=$SPATIAL_VEGARD_SIGMA_Z")
fi
if [[ -n "$SPATIAL_VEGARD_X0" ]]; then
  extra_args+=("spatial_vegard_x0=$SPATIAL_VEGARD_X0")
fi
if [[ -n "$SPATIAL_VEGARD_Y0" ]]; then
  extra_args+=("spatial_vegard_y0=$SPATIAL_VEGARD_Y0")
fi
if [[ -n "$SPATIAL_VEGARD_Z0" ]]; then
  extra_args+=("spatial_vegard_z0=$SPATIAL_VEGARD_Z0")
fi
if [[ -n "$ANIS_VEGARD_SIGMA_XY" ]]; then
  extra_args+=("anis_vegard_sigma_xy=$ANIS_VEGARD_SIGMA_XY")
fi
if [[ -n "$ANIS_VEGARD_SIGMA_Z" ]]; then
  extra_args+=("anis_vegard_sigma_z=$ANIS_VEGARD_SIGMA_Z")
fi
if [[ -n "$ANIS_VEGARD_X0" ]]; then
  extra_args+=("anis_vegard_x0=$ANIS_VEGARD_X0")
fi
if [[ -n "$ANIS_VEGARD_Y0" ]]; then
  extra_args+=("anis_vegard_y0=$ANIS_VEGARD_Y0")
fi
if [[ -n "$ANIS_VEGARD_Z0" ]]; then
  extra_args+=("anis_vegard_z0=$ANIS_VEGARD_Z0")
fi
if [[ -n "$ANIS_VEGARD_XX_AMP" ]]; then
  extra_args+=("anis_vegard_xx_amp=$ANIS_VEGARD_XX_AMP")
fi
if [[ -n "$ANIS_VEGARD_YY_AMP" ]]; then
  extra_args+=("anis_vegard_yy_amp=$ANIS_VEGARD_YY_AMP")
fi
if [[ -n "$ANIS_VEGARD_ZZ_AMP" ]]; then
  extra_args+=("anis_vegard_zz_amp=$ANIS_VEGARD_ZZ_AMP")
fi
if [[ -n "$ANIS_VEGARD_XY_AMP" ]]; then
  extra_args+=("anis_vegard_xy_amp=$ANIS_VEGARD_XY_AMP")
fi
if [[ -n "$ANIS_VEGARD_XZ_AMP" ]]; then
  extra_args+=("anis_vegard_xz_amp=$ANIS_VEGARD_XZ_AMP")
fi
if [[ -n "$ANIS_VEGARD_YZ_AMP" ]]; then
  extra_args+=("anis_vegard_yz_amp=$ANIS_VEGARD_YZ_AMP")
fi
if [[ -n "$FLEXO_PROXY_AMP" ]]; then
  extra_args+=("flexo_proxy_amp=$FLEXO_PROXY_AMP")
fi
if [[ -n "$FLEXO_PROXY_GRAD_AMP" ]]; then
  extra_args+=("flexo_proxy_grad_amp=$FLEXO_PROXY_GRAD_AMP")
fi
if [[ -n "$FLEXO_PROXY_SIGMA_XY" ]]; then
  extra_args+=("flexo_proxy_sigma_xy=$FLEXO_PROXY_SIGMA_XY")
fi
if [[ -n "$FLEXO_PROXY_SIGMA_Z" ]]; then
  extra_args+=("flexo_proxy_sigma_z=$FLEXO_PROXY_SIGMA_Z")
fi
if [[ -n "$FLEXO_PROXY_X0" ]]; then
  extra_args+=("flexo_proxy_x0=$FLEXO_PROXY_X0")
fi
if [[ -n "$FLEXO_PROXY_Y0" ]]; then
  extra_args+=("flexo_proxy_y0=$FLEXO_PROXY_Y0")
fi
if [[ -n "$FLEXO_PROXY_Z0" ]]; then
  extra_args+=("flexo_proxy_z0=$FLEXO_PROXY_Z0")
fi

srun_extra=()
if [[ -n "$SRUN_EXTRA" ]]; then
  # shellcheck disable=SC2206
  srun_extra=($SRUN_EXTRA)
fi

srun "${srun_extra[@]}" -n "$NTASKS" "$EXECUTABLE" \
  -i "$INPUT" \
  tip_voltage="$TIP_VOLTAGE" \
  pulse_end="$PULSE_END" \
  sim_end="$SIM_END" \
  Outputs/file_base="$FILE_BASE" \
  "${extra_args[@]}"

"$PYTHON" "$campaign_dir/paper_figures/make_phasefield_evolution_gif.py" \
  --campaign-dir "$campaign_dir" \
  --exodus "$RUN_DIR/$FILE_BASE.e" \
  --stem "$GIF_STEM" \
  --title "$GIF_TITLE" \
  --max-frames "$GIF_MAX_FRAMES" \
  --duration "$GIF_DURATION" \
  --phase-metric "$PHASE_METRIC"

echo "GIF written to $campaign_dir/paper_figures/generated/${GIF_STEM}.gif"
