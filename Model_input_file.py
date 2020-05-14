#! /usr/bin/env python
# -*- coding: utf-8 -*-
#################################################
#   This is the input file for the MTPK codes   #
#################################################


########################################################################
#     Window function deconvolution: HOWTO
#
# We use "training simulations" to correct estimation biases
# as well as the window/selection function deconvolution
#
# We then apply this deconvolution either to a "test simulation" or to data.
#
# For "test simulation" (sims_only=True), the FIRST map of n_maps
# will be considered "data", and the maps 2...n_maps + 1 will be the "training simulations"
#
# For real data (sims_only=False), the data map (in grid format) will be read separately,
# and added as the first map (in that case, the remaining n_maps are the training sims)
import numpy as np

#sims_only = False
sims_only = True

# Use existing window function normalization?
#   -> True -- will obtain window function for all tracers from file
#   -> False -- will normalize data ("dec") to theoretical spectra ("model")
# N.B.: No matter what is your choice, a window function will be created by the Estimate code, 
# which normalizes the mean of the spectra to the fiducial model.
#
#use_window_function = True
use_window_function = False

########################################


########################################################################
#    Tracers: halos v. galaxies
#
# Sometimes we will analyze the power spectra of halos (from, e.g. simulations)
# Sometimes we will analyze the power spectra of galaxies (either from data, or
# generated from halos by some HODs)

#do_galaxies = False
do_galaxies = True

# Make Poisson sampling of the combined maps?
do_poisson = False

########################################


########################################################################
#     HODs
#
# Each halo is divided into bins of equal halo bias.
# Usually these bins are simply an interval in halo mass.
# One can include secondary halo bias parameters but, the halo bias bins must be "flat"
#
# The files below determine:
#   - mass function for the given halo bin i : mean number of halos(i)/ h^-3 Mpc^3 (array, N_halos)
#   - halo bias for the bins i : b_i (array, N_halos)
#   - HOD -- mean number of galaxies type g on each bin i : H_gi (array, N_galx x N_halos)

#mass_fun_file = "TEST_mass_fun_Tinker.dat"
#halo_bias_file = "TEST_halo_bias_Tinker.dat"
#hod_file = "TEST_hod_Tinker.dat"

mass_fun_file = "renan_mass_function.dat"
halo_bias_file = "renan_halo_bias.dat"
hod_file = "renan_hod_comb.dat"

# Number of halo bins
nhalos = 2
halos_ids = ['M1', 'M2']
# Number of galaxy types
ngals = 1
gal_ids = ['G']


########################################################################
#     Number of *lognormal simulations* of the maps for halo type
n_maps = 50
########################################


########################################################################
#     Box properties
#
# These are the physical characteristics of the rectangular box in which
# the data (true or sims) is circumscribed.

# Cell physical size, in units of h^-1 Mpc
cell_size = 10.0

# Number of cells in x,y and z directions.
# Must be integer.
n_x = 155
n_y = 155
n_z = 50

# Displacement of the origin (0,0,0) of the box with respect to Earth
# (in cell units; doesn't need to be integer)
n_x_orig = -n_x/2.
n_y_orig = -n_x/2.
n_z_orig = 96.1
################################


################################################################
#   Selection function parameters.
#
#   If the galaxy selection function is given ("data" selection function):
sel_fun_data = False
#sel_fun_data = False

# If sel_fun_data = True, then specify file with data selection function
# (must be a grid identical to the one defined in the previous section:
# nx X ny X nz , one for each type of galaxy
#sel_fun_file = "V0_completeness_selection_function_z1.hdf5"

## !! NEW !!
# If low-count cells must be masked out, then cells with counts 
# below this threshold will be eliminated from the mocks AND from the data
cell_low_count_thresh = 0.0

# One may add a shift and/or a multiplicative factor, for testing:
#   n_bar --> factor * (n_bar + shift)
# N.B.: shift in units of number/(cell size)^3 , NOT number/(h^-3 Mpc^3)
mult_sel_fun, shift_sel_fun = 1.0 , 0.0

# If sel_fun_data=False, then an analytical fit for the selection function is used.
#   ATTENTION: sel. f. fit parameters are in units of grid cells !!!
ncentral = [10.0]
nsigma   = [20000.0]
################################


################################################################
# Galaxy bias -- in the map-creating tool, halo bias is given in the file;
# and galaxy bias can be computed from the HODs and halo bias.
#
# HOWEVER, if you want to assume another fiducial value for the bias
# in the power spectrum estimation, specify it here, and this will
# SUPERSEDE the bias from the HOD calculation -- ONLY FOR THE ESTIMATION TOOL!
#
# You may also define a DATA bias that is different from the mocks


# Here, define the interval in k that you want to use to estimate the bias
kmin_bias = 0.1
kmax_bias = 0.2

################################################################
# Shot noise and 1-halo term
#
# Shot noise subtraction can be made to be more or less conservative.
# For this, employ the "shot_fudge" parameter below.
# shot_fudge=0.0 is a conservative shot-noise subtraction
# shot_fudge=1.0 is very agressive
shot_fudge=0.1

# 1-halo term amplitude -- again, this can be computed from the HOD.
# But if you want to assume some other value, define it here.
# If nonzero, it will be added to the theoretical expectation of the power spectra

#gal_p1h = [10.0,10.0,10.0,10.0]




################################################################
#   Target value of k where the power estimation will be optimal
# (units of h/Mpc)
kph_central = 0.1
# Binning in k -- this should be ~ 1.4/(smallest side of box). In units of h^-1 Mpc
dkph_bin = 0.005
# Max k -- this should be < 2 \pi * Nyquist frequency = \pi /cell_size. In units of h^-1 Mpc
kmax_phys = 0.3
# Min k -- this should be > 1.0/box_size. In units of h^-1 Mpc
kmin_phys = 0.006

#
# Typically, for boxes of ~1 h^-1 Gpc side, a binning dkph_bin ~0.003 - 0.01 is OK.
# For smaller boxes, that binning has to be increased.
#
# Very small bins make the code run slower, and may be useless unless there are
# "kinky" features in your power spectrum.
#
# ATTENTION!!!
# This binning is used both in the spectrum estimation tool AND in the lognormal sims tool
# (angle averages in Fourier space on a square box depend on the binning!)
################################


################################################################
#   Matter growth rate f(z_central) of redshift-space distortions
#matgrowcentral=0.001
# Leave commented if you want f(z) given by the assumed cosmological model
# If computing the power spectrum of halos from an N-body simulation in real space, set this to zero.
################################


################################################################
# Central (mean, or median) redshift of the catalog or simulated data, and width of that bin
zcentral = 0.35
zbinwidth = 0.1
################################


################################################################
# Other parameters useful for power spectrum estimation
# NEED TO IMPLEMENT TOOLS TO ACTUALLY USE THIS!
#
# Gaussian redshift errors of galaxy maps
# Units of the cell size!
sigz_est = ngals*[0.00001] #Adimensional: (1+\beta*mu**2) --> (1+\beta*mu**2)*exp(-k**2*mu**2*(sigz_est*c/H)/2)
# Alpha-type dipoles
adip = sigz_est
# Velocity dispersions for RSDs: f*mu^2 --> f*mu^2*exp(-k^2*mu^2*(vdisp/H)**2/2)
vdisp = sigz_est #km/s -- This is usually of the order 150km/s
# Speed of light
c = 299792.458 #km/s
################################


################################################################
# Which spectrum to use in the ln sims and estimation.
# (0) linear (1) HaloFit (2) PkEqual
whichspec = 1
################################




################################################################
#   Jing deconvolution of mass assignement method (NGP, CIC, etc.)
#
# (1) Ln simulations only, no averaging -- NO deconv.
# (2) Simulations only, with averaging -- YES deconv.
#                 (2x2x2 -> power_jing=~ 1.65, 4x4x4 -> power_jing = 2.0)
# (3) Real/N-body map (NGP) + LN sims, no averaging -- YES deconv. for real map (p=2.0),
#                                                       NO deconv. for sims
# (4) Real/N-body map (CIC) + LN sims, no averaging -- YES deconv. for real map (p=3.0),
#                                                       NO deconv. for sims
# (5) Real/N-body map (NGP) + LN sims, with 2x2x2 averaging -- YES deconv. for real map (p=3.0),
#                                                              YES deconv. for sims (p=1.65)
# etc.

# Jing deconvolution for sims: False or True
jing_dec_sims = False

# Power used for deconvolution window function of sims:
power_jing_sims = 1.65
#power_jing_sims = 2.0

# Power used for data window function:
power_jing_data = 2.0
################################

###############################################################################
# Whether or not to plot the 2D covariances (FKP v. MT)
plot_all_cov = True
###############################################


###############################################################################
#   Parameters used for map creation
#
# Fudge factor to create correlated multi-tracer lognormal maps
# (guarantees that covariance matrix is positive-definite).
# If when running the map-generating code appears an error message that ends with:
# "LinAlgError: Matrix is not positive definite" ,
# then increase this number. Ideally, though, it should be << 0.01 .
fudge = 0.0000001
###############################################


###############################################################################
#   Cosmological model parameters (used by NumCosmo)
Omegak=0.0
w0= -0.999
w1= 0.0
Omegab=0.048206
Omegac=0.259
H0=67.7
n_SA=0.96
ln10e10ASA=3.085
z_re=9.9999
k_min_camb = 1.e-4
k_max_camb = 1.e+0
gamma = 0.5454

