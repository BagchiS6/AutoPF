tip_voltage = 10
pulse_end = 0.5
# domain_radius = 200.0
tip_radius = 20.0
Nx = 40
Ny = 40
Nz = 10

pfm_image_file = 'initial_Pz_data.txt'

# Independent Elastic Constants
c11 = 275.0
c12 = 179.0
c44 = 54.3

# Independent Electrostrictive Constants (Strain-based)
Q11 = -0.11
Q12 = 0.045
Q44 = -0.029

# GRADIENT ENERGY COEFFICIENTS
g11 = 0.51
g12 = -0.02
g44 = 0.02

# Hidden-physics controls. Defaults reduce exactly to the baseline model.
#
# vegard_strain is an isotropic chemical/eigenstrain proxy for defect- or
# vacancy-induced Vegard expansion/contraction:
#   eps_ij^Vegard = vegard_strain * delta_ij
#
# screen_lambda adds a FERRET DepolEnergy residual to the polar_z equation:
#   F_screen ~ 0.5 * screen_lambda / eps_b * Pz * <Pz>
vegard_strain = 0.0
screen_lambda = 0.0
screen_permitivitty = 0.08854187

# Spatial hidden-physics controls. The scalar parameters above are constant
# offsets; the amplitudes below add smooth surface-localized fields. These are
# meant for low-dimensional coefficient search, not pixelwise BO.
spatial_screen_amp = 0.0
spatial_screen_sigma_xy = 120.0
spatial_screen_sigma_z = 35.0
spatial_screen_x0 = 0.0
spatial_screen_y0 = 0.0
spatial_screen_z0 = 100.0

spatial_vegard_amp = 0.0
spatial_vegard_sigma_xy = 120.0
spatial_vegard_sigma_z = 35.0
spatial_vegard_x0 = 0.0
spatial_vegard_y0 = 0.0
spatial_vegard_z0 = 100.0

# Optional response-basis hidden fields. Defaults are zero, so the input
# remains identical to the Gaussian hidden-field model unless a manifest
# explicitly passes these coefficients. The z envelope localizes the basis near
# the surface, while low-order polynomial/ring terms let us test defect-like
# spatial Vegard responses without pixelwise BO.
basis_screen_sigma_z = 35.0
basis_vegard_sigma_z = 30.0

leg_screen_c0 = 0.0
leg_screen_cx = 0.0
leg_screen_cy = 0.0
leg_screen_cxx = 0.0
leg_screen_cyy = 0.0
leg_screen_cxy = 0.0

leg_vegard_c0 = 0.0
leg_vegard_cx = 0.0
leg_vegard_cy = 0.0
leg_vegard_cxx = 0.0
leg_vegard_cyy = 0.0
leg_vegard_cxy = 0.0

ring_screen_amp = 0.0
ring_vegard_amp = 0.0
odd_ring_vegard_amp = 0.0
ring_radius = 100.0
ring_sigma_xy = 35.0
ring_x0 = 0.0
ring_y0 = 0.0

# Low-rank anisotropic eigenstrain controls. These are additional spatial
# eigenstrain components beyond the isotropic Vegard proxy above:
#   eps^*_ij(x) = eps^iso(x) delta_ij + phi_a(x) A_ij
# Defaults are zero so this input reduces to the corrected z-decay baseline.
anis_vegard_sigma_xy = 120.0
anis_vegard_sigma_z = 35.0
anis_vegard_x0 = 0.0
anis_vegard_y0 = 0.0
anis_vegard_z0 = 100.0
anis_vegard_xx_amp = 0.0
anis_vegard_yy_amp = 0.0
anis_vegard_zz_amp = 0.0
anis_vegard_xy_amp = 0.0
anis_vegard_xz_amp = 0.0
anis_vegard_yz_amp = 0.0

# Phenomenological flexo proxy. This is not a rigorous flexoelectric tensor
# coupling; it is an effective surface-localized internal-field source in the
# polar_z equation, used to test whether residuals prefer a gradient-like
# surface driving term before committing to new flexoelectric kernels.
flexo_proxy_amp = 0.0
flexo_proxy_grad_amp = 0.0
flexo_proxy_sigma_xy = 80.0
flexo_proxy_sigma_z = 25.0
flexo_proxy_x0 = 0.0
flexo_proxy_y0 = 0.0
flexo_proxy_z0 = 100.0

# Initial-condition smoothing for PFM-derived surface seed.
#
# The experimental PFM map constrains the top surface, not the full switched
# volume. We therefore relax the surface signal back to the down-polarized
# background through thickness and confine the switched seed to a smooth
# condition-specific ellipsoidal cap estimated from the PFM footprint.
#
# This gives a surface-nucleated cap/teardrop-like seed instead of extruding
# the 2D PFM mask as a rectangular column.
polar_down = -0.26
ic_surface_z = 100.0
ic_amp_decay_nm = 120.0
ic_val_threshold_surface = 0.0
ic_lateral_smooth = 0.035
ic_x0 = 0.0
ic_y0 = 0.0
ic_rx = 90.0
ic_ry = 70.0
ic_rz = 55.0
ic_cos_theta = 1.0
ic_sin_theta = 0.0
ic_ellipsoid_smooth = 0.18

# Stopping tolerance (in %) for free energy change (for the Terminator)
energy_tol = 1e-2
relax_start_time = 0.0
sim_end = 10.0


[Mesh]
  [gen]
    ############################################
    ##
    ##  Type and dimension of the mesh
    ##
    ############################################

    type = GeneratedMeshGenerator
    dim = 3

  nx = ${Nx}
  ny = ${Ny}
  nz = ${Nz}

  #############################################
  ##
  ##   Actual spatial coordinates of mesh.
  ##   Jmax - Jmin = nJ/2 for J = x, y, z
  ##   Units are in nanometers
  ##
  #############################################

  xmin = -250.0
  xmax = 250.0
  ymin = -250.0
  ymax = 250.0
  zmin = -100.0
  zmax = 100.0

    #############################################
    ##
    ##  FE type/order (hexahedral, tetrahedral
    ##
    #############################################

    elem_type = HEX8
  []
  [./cnode]
    input = gen

    ############################################
    ##
    ##   additional boundary sideset (one node)
    ##   to zero one of the elastic displacement vectors
    ##   vectors and eliminates rigid body translations
    ##   from the degrees of freedom
    ##
    ##   NOTE: This must conform with the about
    ##         [Mesh] block settings
    ##
    ############################################

    type = ExtraNodesetGenerator
    coord = '0.0 0.0 0.0'
    new_boundary = 100
  [../]
[]

[GlobalParams]
  len_scale = 1.0

  polar_x = polar_x
  polar_y = polar_y
  polar_z = polar_z
  potential_E_int = potential_E_int
  vol = vol

  displacements = 'u_x u_y u_z'


  ##############################################
  ##
  ##  IMPORTANT(!): Units are nm, kg,
  ##                seconds, and attocoulombs
  ##
  ##############################################

  u_x = u_x
  u_y = u_y
  u_z = u_z
[]

[Functions]
  [./stripe1]
    type = ParsedFunction
    value = 0.1*cos(0.10471975512*(x+2))  #2pi/L = 0.10471975512
  [../]
  [./afm_tip_voltage]
    type = ParsedFunction
    value = '${tip_radius}^2/(x^2 + y^2 + ${tip_radius}^2)*(${tip_voltage} / 2) * (tanh( t / 0.001 ) - tanh( (t - ${pulse_end}) / 0.05 ))' # Lorentzian tip, sharp turn-on (tau=0.001), smooth turn-off (tau=0.05) to avoid dt collapse at long pulse_end
  [../]  
  [./pfm_data_func]
    type = PiecewiseMultilinear
    data_file = ${pfm_image_file}
  [../]
  [./pfm_3d_nucleus]
    type = ParsedFunction
    symbol_names = 'val'
    symbol_values = 'pfm_data_func'
    # d = zsurf-z. The smooth PFM gate suppresses background noise at the
    # surface. The ellipsoidal gate gives a finite surface-nucleated cap whose
    # lateral radii and center are set per condition from the PFM footprint.
    value = '${polar_down} + (val - ${polar_down})*0.5*(1 + tanh((val - ${ic_val_threshold_surface})/${ic_lateral_smooth}))*0.5*(1 + tanh((1 - ((((x - ${ic_x0})*${ic_cos_theta} + (y - ${ic_y0})*${ic_sin_theta})^2/${ic_rx}^2) + (((-(x - ${ic_x0})*${ic_sin_theta} + (y - ${ic_y0})*${ic_cos_theta})^2/${ic_ry}^2)) + ((${ic_surface_z} - z)^2/${ic_rz}^2)))/${ic_ellipsoid_smooth}))*exp(-(${ic_surface_z} - z)/${ic_amp_decay_nm})'
  [../]
  [./spatial_screen_lambda]
    type = ParsedFunction
    value = '${screen_lambda} + ${spatial_screen_amp} * exp(-((x - ${spatial_screen_x0})^2 + (y - ${spatial_screen_y0})^2)/(2*${spatial_screen_sigma_xy}^2)) * exp(-((z - ${spatial_screen_z0})^2)/(2*${spatial_screen_sigma_z}^2)) + exp(-((z - 100.0)^2)/(2*${basis_screen_sigma_z}^2)) * (${leg_screen_c0} + ${leg_screen_cx}*(x/250.0) + ${leg_screen_cy}*(y/250.0) + ${leg_screen_cxx}*(0.5*(3*(x/250.0)^2 - 1)) + ${leg_screen_cyy}*(0.5*(3*(y/250.0)^2 - 1)) + ${leg_screen_cxy}*(x/250.0)*(y/250.0) + ${ring_screen_amp} * exp(-((sqrt((x - ${ring_x0})^2 + (y - ${ring_y0})^2) - ${ring_radius})^2)/(2*${ring_sigma_xy}^2)))'
  [../]
  [./spatial_vegard_prefactor]
    type = ParsedFunction
    value = '${vegard_strain} + ${spatial_vegard_amp} * exp(-((x - ${spatial_vegard_x0})^2 + (y - ${spatial_vegard_y0})^2)/(2*${spatial_vegard_sigma_xy}^2)) * exp(-((z - ${spatial_vegard_z0})^2)/(2*${spatial_vegard_sigma_z}^2)) + exp(-((z - 100.0)^2)/(2*${basis_vegard_sigma_z}^2)) * (${leg_vegard_c0} + ${leg_vegard_cx}*(x/250.0) + ${leg_vegard_cy}*(y/250.0) + ${leg_vegard_cxx}*(0.5*(3*(x/250.0)^2 - 1)) + ${leg_vegard_cyy}*(0.5*(3*(y/250.0)^2 - 1)) + ${leg_vegard_cxy}*(x/250.0)*(y/250.0) + ${ring_vegard_amp} * exp(-((sqrt((x - ${ring_x0})^2 + (y - ${ring_y0})^2) - ${ring_radius})^2)/(2*${ring_sigma_xy}^2)) + ${odd_ring_vegard_amp}*(y/250.0)*exp(-((sqrt((x - ${ring_x0})^2 + (y - ${ring_y0})^2) - ${ring_radius})^2)/(2*${ring_sigma_xy}^2)))'
  [../]
  [./anis_vegard_xx_prefactor]
    type = ParsedFunction
    value = '${anis_vegard_xx_amp} * exp(-((x - ${anis_vegard_x0})^2 + (y - ${anis_vegard_y0})^2)/(2*${anis_vegard_sigma_xy}^2)) * exp(-((z - ${anis_vegard_z0})^2)/(2*${anis_vegard_sigma_z}^2))'
  [../]
  [./anis_vegard_yy_prefactor]
    type = ParsedFunction
    value = '${anis_vegard_yy_amp} * exp(-((x - ${anis_vegard_x0})^2 + (y - ${anis_vegard_y0})^2)/(2*${anis_vegard_sigma_xy}^2)) * exp(-((z - ${anis_vegard_z0})^2)/(2*${anis_vegard_sigma_z}^2))'
  [../]
  [./anis_vegard_zz_prefactor]
    type = ParsedFunction
    value = '${anis_vegard_zz_amp} * exp(-((x - ${anis_vegard_x0})^2 + (y - ${anis_vegard_y0})^2)/(2*${anis_vegard_sigma_xy}^2)) * exp(-((z - ${anis_vegard_z0})^2)/(2*${anis_vegard_sigma_z}^2))'
  [../]
  [./anis_vegard_xy_prefactor]
    type = ParsedFunction
    value = '${anis_vegard_xy_amp} * exp(-((x - ${anis_vegard_x0})^2 + (y - ${anis_vegard_y0})^2)/(2*${anis_vegard_sigma_xy}^2)) * exp(-((z - ${anis_vegard_z0})^2)/(2*${anis_vegard_sigma_z}^2))'
  [../]
  [./anis_vegard_xz_prefactor]
    type = ParsedFunction
    value = '${anis_vegard_xz_amp} * exp(-((x - ${anis_vegard_x0})^2 + (y - ${anis_vegard_y0})^2)/(2*${anis_vegard_sigma_xy}^2)) * exp(-((z - ${anis_vegard_z0})^2)/(2*${anis_vegard_sigma_z}^2))'
  [../]
  [./anis_vegard_yz_prefactor]
    type = ParsedFunction
    value = '${anis_vegard_yz_amp} * exp(-((x - ${anis_vegard_x0})^2 + (y - ${anis_vegard_y0})^2)/(2*${anis_vegard_sigma_xy}^2)) * exp(-((z - ${anis_vegard_z0})^2)/(2*${anis_vegard_sigma_z}^2))'
  [../]
  [./flexo_proxy_drive]
    type = ParsedFunction
    value = '(${flexo_proxy_amp} + ${flexo_proxy_grad_amp} * ((z - ${flexo_proxy_z0})/${flexo_proxy_sigma_z})) * exp(-((x - ${flexo_proxy_x0})^2 + (y - ${flexo_proxy_y0})^2)/(2*${flexo_proxy_sigma_xy}^2)) * exp(-((z - ${flexo_proxy_z0})^2)/(2*${flexo_proxy_sigma_z}^2))'
  [../]
[]


[Variables]

  #################################
  ##
  ##  Variable definitions
  ##    P, u, phi, e^global_ij
  ##  and their initial conditions
  ##
  #################################

  [./global_strain]
    order = SIXTH
    family = SCALAR
  [../]
  [./polar_x]
    order = FIRST
    family = LAGRANGE
    [./InitialCondition]
      type = RandomIC
      min = -0.01e-4
      max = 0.01e-4
    [../]
  [../]
  [./polar_y]
    order = FIRST
    family = LAGRANGE
    [./InitialCondition]
      type = RandomIC
      min = -0.01e-4
      max = 0.01e-4
    [../]
  [../]
  # [./polar_z]
  #   order = FIRST
  #   family = LAGRANGE
  #   [./InitialCondition]
  #     type = SmoothCircleIC
  #     x1 = 0                   # Center X
  #     y1 = 0                   # Center Y
  #     radius = ${domain_radius}
  #     invalue = 1.0            # Polarization inside the circle
  #     outvalue = -1.0          # Polarization outside the circle
  #     int_width = 1.0          # Width of the diffuse interface (adjust as needed)
  #   [../]
  # [../]

  [./polar_z]
    order = FIRST
    family = LAGRANGE
    [./InitialCondition]
      type = FunctionIC
      variable = polar_z
      function = pfm_3d_nucleus
    [../]
  [../]

  [./potential_E_int]
    order = FIRST
    family = LAGRANGE
  [../]

  [./u_x]
    order = FIRST
    family = LAGRANGE
  [../]
  [./u_y]
    order = FIRST
    family = LAGRANGE
  [../]
  [./u_z]
    order = FIRST
    family = LAGRANGE
  [../]
[]

[AuxVariables]

  ######################################
  ##
  ##  Auxiarilly variable definitions
  ##   (can be intermediate variables
  ##   or for postprocessed quantities)
  ##
  ######################################


  ######################################
  ##
  ##  Global displacements
  ##
  ######################################

  [./disp_x]
  [../]
  [./disp_y]
  [../]
  [./disp_z]
  [../]

  ######################################
  ##
  ##  Stress/strain tensor components
  ##
  ######################################

  [./e00]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./e01]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./e10]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./e11]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./e12]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./e22]
    order = CONSTANT
    family = MONOMIAL
  [../]

  [./s00]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./s01]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./s10]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./s11]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./s12]
    order = CONSTANT
    family = MONOMIAL
  [../]
  [./s22]
    order = CONSTANT
    family = MONOMIAL
  [../]
[]

[AuxKernels]

  ######################################
  ##
  ##  Auxiarilly Kernel definitions
  ##   (can be intermediate "operations"
  ##   or for postprocessed quantities)
  ##
  ######################################

  [./disp_x]
    type = GlobalDisplacementAux
    variable = disp_x
    scalar_global_strain = global_strain
    global_strain_uo = global_strain_uo
    component = 0
    use_displaced_mesh = false
  [../]
  [./disp_y]
    type = GlobalDisplacementAux
    variable = disp_y
    scalar_global_strain = global_strain
    global_strain_uo = global_strain_uo
    component = 1
    use_displaced_mesh = false
  [../]
  [./disp_z]
    type = GlobalDisplacementAux
    variable = disp_z
    scalar_global_strain = global_strain
    global_strain_uo = global_strain_uo
    component = 2
    use_displaced_mesh = false
  [../]
  [./e00]
    type = RankTwoAux
    variable = e00
    rank_two_tensor = total_strain
    index_i = 0
    index_j = 0
  [../]
  [./e01]
    type = RankTwoAux
    variable = e01
    rank_two_tensor = total_strain
    index_i = 0
    index_j = 1
  [../]
  [./e10]
    type = RankTwoAux
    variable = e10
    rank_two_tensor = total_strain
    index_i = 1
    index_j = 0
  [../]
  [./e12]
    type = RankTwoAux
    variable = e12
    rank_two_tensor = total_strain
    index_i = 1
    index_j = 2
  [../]
  [./e11]
    type = RankTwoAux
    variable = e11
    rank_two_tensor = total_strain
    index_i = 1
    index_j = 1
  [../]
  [./e22]
    type = RankTwoAux
    variable = e22
    rank_two_tensor = total_strain
    index_i = 2
    index_j = 2
  [../]

  [./s00]
    type = RankTwoAux
    variable = s00
    rank_two_tensor = stress
    index_i = 0
    index_j = 0
  [../]
  [./s01]
    type = RankTwoAux
    variable = s01
    rank_two_tensor = stress
    index_i = 0
    index_j = 1
  [../]
  [./s10]
    type = RankTwoAux
    variable = s10
    rank_two_tensor = stress
    index_i = 1
    index_j = 0
  [../]
  [./s12]
    type = RankTwoAux
    variable = s12
    rank_two_tensor = stress
    index_i = 1
    index_j = 2
  [../]
  [./s11]
    type = RankTwoAux
    variable = s11
    rank_two_tensor = stress
    index_i = 1
    index_j = 1
  [../]
  [./s22]
    type = RankTwoAux
    variable = s22
    rank_two_tensor = stress
    index_i = 2
    index_j = 2
  [../]
[]

[ScalarKernels]

  ######################################
  ##
  ##  Necessary for PBC system
  ##
  ######################################

  [./global_strain]
    type = GlobalStrain
    variable = global_strain
    global_strain_uo = global_strain_uo
    use_displaced_mesh = false
  [../]
[]

[Materials]

  #################################################
  ##
  ## Bulk free energy and electrostrictive
  ## coefficients gleaned from
  ## Marton and Hlinka
  ##    Phys. Rev. B. 74, 104014, (2006)
  ##
  ## NOTE: there might be some Legendre transforms
  ##        depending on what approach you use
  ##        -i.e. inhomogeneous strain vs
  ##            homogeneous strain [renormalized]
  ##
  ##################################################

  [./Landau_P]
    type = GenericConstantMaterial
    prop_names = 'alpha1 alpha11 alpha12 alpha111 alpha112 alpha123 alpha1111 alpha1112 alpha1122 alpha1123'
    prop_values = '-0.027721 -0.64755 0.323 8.004 4.47 4.91 0.0 0.0 0.0 0.0'
  [../]

  ############################################
  ##
  ## Gradient coefficients from
  ## Marton and Hlinka
  ##    Phys. Rev. B. 74, 104014, (2006)
  ##
  ############################################

  [./Landau_G]
    type = GenericConstantMaterial
    prop_names = 'G110 G11_G110 G12_G110 G44_G110 G44P_G110'
    prop_values = '0.5 ${g11} ${g12} ${g44} 0.0'
  [../]

  [mat_C]
    type = GenericConstantMaterial
    prop_names = 'C11 C12 C44'
    prop_values = '${c11} ${c12} ${c44}'
  [../]

  [mat_Q]
    type = GenericConstantMaterial
    prop_names = 'Q11 Q12 Q44'
    prop_values = '${Q11} ${Q12} ${Q44}'
  [../]

  # --- 2. Calculated q coefficients (Physical Consistency) ---
  # Note: Including the overall factor of -1 for Ferret sign convention
  [mat_q11]
    type = ParsedMaterial
    property_name = 'q11'
    expression = '1.0 * (${c11} * ${Q11} + 2.0 * ${c12} * ${Q12})'
  [../]
  [mat_q12]
    type = ParsedMaterial
    property_name = 'q12'
    expression = '1.0 * (${c11} * ${Q12} + ${c12} * (${Q11} + ${Q12}))'
  [../]
  [mat_q44]
    type = ParsedMaterial
    property_name = 'q44'
    expression = '1.0 * (${c44} * ${Q44})'
  [../]

  # --- 3. Tensors and Mechanics ---
  [./vegard_prefactor_mat]
    type = GenericFunctionMaterial
    prop_names = 'vegard_prefactor'
    prop_values = 'spatial_vegard_prefactor'
  [../]
  [./anis_vegard_prefactor_mat]
    type = GenericFunctionMaterial
    prop_names = 'anis_vegard_xx_prefactor anis_vegard_yy_prefactor anis_vegard_zz_prefactor anis_vegard_xy_prefactor anis_vegard_xz_prefactor anis_vegard_yz_prefactor'
    prop_values = 'anis_vegard_xx_prefactor anis_vegard_yy_prefactor anis_vegard_zz_prefactor anis_vegard_xy_prefactor anis_vegard_xz_prefactor anis_vegard_yz_prefactor'
  [../]

  [eigen_strain]
    type = ComputeEigenstrain
    eigen_base = '1.0 0 0 0 1.0 0 0 0 1.0'
    eigenstrain_name = eigenstrain
    prefactor = vegard_prefactor
  [../]
  [eigen_strain_anis_xx]
    type = ComputeEigenstrain
    eigen_base = '1.0 0 0 0 0 0 0 0 0'
    eigenstrain_name = eigenstrain_anis_xx
    prefactor = anis_vegard_xx_prefactor
  [../]
  [eigen_strain_anis_yy]
    type = ComputeEigenstrain
    eigen_base = '0 0 0 0 1.0 0 0 0 0'
    eigenstrain_name = eigenstrain_anis_yy
    prefactor = anis_vegard_yy_prefactor
  [../]
  [eigen_strain_anis_zz]
    type = ComputeEigenstrain
    eigen_base = '0 0 0 0 0 0 0 0 1.0'
    eigenstrain_name = eigenstrain_anis_zz
    prefactor = anis_vegard_zz_prefactor
  [../]
  [eigen_strain_anis_xy]
    type = ComputeEigenstrain
    eigen_base = '0 1.0 0 1.0 0 0 0 0 0'
    eigenstrain_name = eigenstrain_anis_xy
    prefactor = anis_vegard_xy_prefactor
  [../]
  [eigen_strain_anis_xz]
    type = ComputeEigenstrain
    eigen_base = '0 0 1.0 0 0 0 1.0 0 0'
    eigenstrain_name = eigenstrain_anis_xz
    prefactor = anis_vegard_xz_prefactor
  [../]
  [eigen_strain_anis_yz]
    type = ComputeEigenstrain
    eigen_base = '0 0 0 0 0 1.0 0 1.0 0'
    eigenstrain_name = eigenstrain_anis_yz
    prefactor = anis_vegard_yz_prefactor
  [../]

  [elasticity_tensor_1]
    type = ComputeElasticityTensor
    fill_method = symmetric9
    # Format: C11 C12 C13 C22 C23 C33 C44 C55 C66
    C_ijkl = '${c11} ${c12} ${c12} ${c11} ${c12} ${c11} ${c44} ${c44} ${c44}'
  [../]

  [strain_1]
    type = ComputeSmallStrain
    global_strain = global_strain
    eigenstrain_names = 'eigenstrain eigenstrain_anis_xx eigenstrain_anis_yy eigenstrain_anis_zz eigenstrain_anis_xy eigenstrain_anis_xz eigenstrain_anis_yz'
  [../]

  [stress_1]
    type = ComputeLinearElasticStress
  [../]

  [global_strain]
    type = ComputeGlobalStrain
    scalar_global_strain = global_strain
    global_strain_uo = global_strain_uo
  [../]

  [slab_ferroelectric]
    type = ComputeElectrostrictiveTensor
    # Format: Q11 Q12 Q13 Q22 Q23 Q33 Q44 Q55 Q66
    Q_mnkl = '${Q11} ${Q12} ${Q12} ${Q11} ${Q12} ${Q11} ${Q44} ${Q44} ${Q44}'
    C_ijkl = '${c11} ${c12} ${c12} ${c11} ${c12} ${c11} ${c44} ${c44} ${c44}'
  [../]

  [./permitivitty_1]

    ###############################################
    ##
    ##  so-called background dielectric constant
    ##  (it encapsulates the motion of core electrons
    ##  at high frequency) = e_b*e_0 (here we use
    ##  e_b = 10), see PRB. 74, 104014, (2006)
    ##
    ###############################################

    type = GenericConstantMaterial
    prop_names = 'permittivity'
    prop_values = '0.08854187'
  [../]
[]


[Kernels]

  ###############################################
  ##
  ## Physical Kernel operators
  ## to enforce TDLGD evolution
  ##
  ###############################################


  #Elastic problem
  [./SolidMechanics]
    use_displaced_mesh = false
  [../]

  [./bed_x]
    type = BulkEnergyDerivativeEighth
    variable = polar_x
    component = 0

  [../]
  [./bed_y]
    type = BulkEnergyDerivativeEighth
    variable = polar_y
    component = 1
  [../]
  [./bed_z]
    type = BulkEnergyDerivativeEighth
    variable = polar_z
    component = 2
  [../]

  [./walled_x]
    type = WallEnergyDerivative
    variable = polar_x
    component = 0
  [../]
  [./walled_y]
    type = WallEnergyDerivative
    variable = polar_y
    component = 1
  [../]
  [./walled_z]
     type = WallEnergyDerivative
     variable = polar_z
     component = 2
  [../]

  [./electrostr_ux]
    type = ElectrostrictiveCouplingDispDerivative
    variable = u_x
    component = 0

  [../]
  [./electrostr_uy]
    type = ElectrostrictiveCouplingDispDerivative
    variable = u_y
    component = 1
  [../]
  [./electrostr_uz]
    type = ElectrostrictiveCouplingDispDerivative
    variable = u_z
    component = 2
  [../]

  [./electrostr_polar_coupled_x]
    type = ElectrostrictiveCouplingPolarDerivative
    variable = polar_x
    component = 0
  [../]
  [./electrostr_polar_coupled_y]
    type = ElectrostrictiveCouplingPolarDerivative
    variable = polar_y
    component = 1
  [../]
  [./electrostr_polar_coupled_z]
    type = ElectrostrictiveCouplingPolarDerivative
    variable = polar_z
    component = 2
  [../]


  [./polar_x_electric_E]
     type = PolarElectricEStrong
     variable = potential_E_int
  [../]
  [./FE_E_int]
     type = Electrostatics
     variable = potential_E_int
  [../]

  [./polar_electric_px]
     type = PolarElectricPStrong
     variable = polar_x
     component = 0
  [../]
  [./polar_electric_py]
     type = PolarElectricPStrong
     variable = polar_y
     component = 1
  [../]
  [./polar_electric_pz]
     type = PolarElectricPStrong
     variable = polar_z
     component = 2
  [../]

  [./screening_depol_z]
     type = BodyForce
     variable = polar_z
     value = ${fparse -0.5 / ${screen_permitivitty}}
     postprocessor = avePz
     function = spatial_screen_lambda
  [../]
  [./flexo_proxy_pz]
     type = BodyForce
     variable = polar_z
     value = 1.0
     function = flexo_proxy_drive
  [../]

  [./polar_x_time]
     type = TimeDerivativeScaled
     variable=polar_x
     time_scale = 1.0
  [../]
  [./polar_y_time]
     type = TimeDerivativeScaled
     variable=polar_y
     time_scale = 1.0
  [../]
  [./polar_z_time]
     type = TimeDerivativeScaled
     variable = polar_z
     time_scale = 1.0
  [../]
[]


[BCs]
  [./Periodic]
    [./xyz]
      auto_direction = 'x y'
      variable = 'u_x u_y u_z polar_x polar_y polar_z potential_E_int'
    [../]
  [../]

  # fix center point location
  #[./centerfix_x]
  # type = DirichletBC
  #  boundary = 100
  #  variable = u_x
  #  value = 0
  #[../]
  #[./centerfix_y]
  #  type = DirichletBC
  #  boundary = 100
  #  variable = u_y
  #  value = 0
  #[../]
  #[./centerfix_z]
  #  type = DirichletBC
  #  boundary = 100
  #  variable = u_z
  #  value = 0
  #[../]

  [./ux_bottom_fix]
    type = DirichletBC
    boundary = 'back'
    variable = u_x
    value = 0.0
  [../]
  [./uy_bottom_fix]
    type = DirichletBC
    boundary = 'back'
    variable = u_y
    value = 0.0
  [../]
  [./uz_bottom_fix]
    type = DirichletBC
    boundary = 'back'
    variable = u_z
    value = 0.0
  [../]
  [./phi_bottom_fix]
    type = DirichletBC
    boundary = 'back'
    variable = potential_E_int
    value = 0.0
  [../]
  [./phi_top]
    type = FunctionDirichletBC
    boundary = 'front'
    variable = potential_E_int
    function = 'afm_tip_voltage'
  [../]
[]

[Postprocessors]

  ###############################################
  ##
  ##  Postprocessors (integrations over the
  ##  computational domain) to calculate the total energy
  ##  decomposed into linear combinations of the
  ##  different physics.
  ##
  ###############################################

  [./Fb]
    type = BulkEnergyEighth
    execute_on = 'initial timestep_end'
  [../]
  [./Fw]
    type = WallEnergy
    execute_on = 'initial timestep_end'
  [../]
  [./Fela]
    type = ElasticEnergy
    execute_on = 'initial timestep_end'
    use_displaced_mesh = false
  [../]
  [./Fc]
    type = ElectrostrictiveCouplingEnergy
    execute_on = 'initial timestep_end'
  [../]
  [./Fele]
    type = ElectrostaticEnergy
    execute_on = 'initial timestep_end'
  [../]
  [./Ftot]
    type = LinearCombinationPostprocessor
    pp_names = 'Fb Fw Fc Fele'
    pp_coefs = ' 1 1 1 1'
    execute_on = 'initial timestep_end'
  [../]
  [./vol]
    type = VolumePostprocessor
    execute_on = 'initial timestep_end'
  [../]
  [./px]
    type = DomainVariantPopulation
    execute_on = 'timestep_end'
    component = 0
  [../]
  [./py]
    type = DomainVariantPopulation
    execute_on = 'timestep_end'
    component = 1
  [../]
  [./pz]
    type = DomainVariantPopulation
    execute_on = 'timestep_end'
    component = 2
  [../]
  [./avePz]
    type = ElementAverageValue
    variable = polar_z
    execute_on = 'initial linear nonlinear timestep_begin timestep_end'
  [../]
  [./perc_change]
    type = PercentChangePostprocessor
    postprocessor = Ftot
    execute_on = 'initial timestep_end'
  [../]
  [./sim_time]
    type = TimePostprocessor
    execute_on = 'initial timestep_end'
  [../]
[]

[UserObjects]

  ###############################################
  ##
  ##  GlobalStrain system to enforce periodicity
  ##  in the anisotropic strain field
  ##
  ###############################################

  [./global_strain_uo]
    type = GlobalATiO3MaterialRVEUserObject
    use_displaced_mesh = false
    execute_on = 'Initial Linear Nonlinear'
  [../]

  ###############################################
  ##
  ##  terminator to end energy evolution when the energy difference
  ##  between subsequent time steps is lower than 5e-6
  ##
  ##  NOTE: can fail if the time step is small
  ##
  ###############################################

  [./kill]
     type = Terminator
     expression = '(sim_time >= ${relax_start_time}) * (perc_change <= ${energy_tol})'
  [../]
[]

[Preconditioning]

  ###############################################
  ##
  ##  Numerical preconditioning/solver options
  ##
  ###############################################

  [./smp]
    type = SMP
    full = true
    petsc_options = '-snes_ksp_ew'
    petsc_options_iname = '-ksp_gmres_restart -snes_atol -snes_rtol -ksp_rtol -pc_type  -build_twosided'
    petsc_options_value = '    160               1e-10      1e-8      1e-6          bjacobi       allreduce'
  [../]
[]

[Executioner]

  ##########################################
  ##
  ##  Time integration/solver options
  ##
  ##########################################

  type = Transient
  solve_type = 'PJFNK'
  scheme = 'implicit-euler'
  abort_on_solve_fail = false
  dtmin = 1e-13
  dtmax = 0.5
  start_time = 0.0
  end_time = ${sim_end}

  ###########################################
  ##
  ##  dtmax is material dependent!
  ##  for PTO is about 0.8 but BTO more like 3-10
  ##
  ###########################################

  # dtmax = 3.0

  l_max_its = 200

  [./TimeStepper]
    type = IterationAdaptiveDT
    optimal_iterations = 8
    cutback_factor = 0.75
    linear_iteration_ratio = 1000
    dt = 0.05
  [../]
  verbose = true
[]



[Outputs]
  # Performance and console settings
  print_linear_residuals = false
  perf_graph = false
  
  # Standard CSV output for Postprocessors (Crucial for Machine Learning)
  csv = true
  
  # Base name for all output files (can be overridden via CLI)
  file_base = out_bto_wall_T298K_2

  [out]
    type = Exodus
    # Optional: interval = 10 (saves space by only saving every 10th frame)
    elemental_as_nodal = true
  []

  [console]
    type = Console
    time_precision = 3
    # Optional: max_rows = 1 (keeps terminal clean during high-throughput runs)
  []
[]
