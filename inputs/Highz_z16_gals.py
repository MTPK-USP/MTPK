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
#     HOD configuration -- used by MTPK_HOD code
#

# Each halo is divided into bins of equal halo bias.
# Usually these bins are simply an interval in halo mass.
# One can include secondary halo bias parameters but, the halo bias bins must be "flat"
#
# The files below determine:
#   - mass function for the given halo bin i : mean number of halos(i)/ h^-3 Mpc^3 (array, N_halos)
#   - halo bias for the bins i : b_i (array, N_halos)
#   - HOD -- mean number of galaxies type g on each bin i : H_gi (array, N_galx x N_halos)

# The halo mass bins for this "quasar + Ly-alpha-emitters" run (Highz) are:
# Log_10 (M) = 11.5-12 , 12-12.5 , 12.5-13, 13-13.5, 13.5-14, 14-14.5

mass_fun_file = "Tinker_HMF_z1.7.dat"
halo_bias_file = "Tinker_Hbias_z1.7.dat"
hod_file = "Highz_hod.dat"

# Number of halo bins
nhalos = 6
halos_ids = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

# Number of galaxy types
ngals = 2
gal_ids = ['G','Q']


# Total number of tracers ("galaxies") in dataset
# Used by code create_grid_randoms_weights.py (which generates the grid for the selection function)
# If you are also generating galaxies with an HOD using the MTPK_HOD code, this number should
# match closely the total number of created objects

# z=1.7
Ntot_Gals = [4652387.71020369,  590595.17353218]

# z=1.7 : 4652387.71020369,  590595.17353218
# z=2.1 : 2161823.16957911,  259828.70707985
# z=2.5 : 1913499.0284404 ,  216246.88933453


#
#
########################################################################



########################################################################
#
# Tracers that will be analyzed in MTPK_estimate code

ntracers = ngals
tracer_ids = gal_ids
bias_file = "Highz_bias_z1.6.dat"

#
########################################################################



########################################################################
#     Number of *lognormal simulations* of the maps for halo type
n_maps = 100

# If you are creating *additional* maps (n_maps), adding them to some 
# pre-existing maps, indicate here where to start counting them.
# The maps will be numbered starting from n_maps_init to n_maps_init + n_maps

#n_maps_init = 75

########################################



########################################################################
#     Box properties
#
#  z  		chi (h^-1 Gpc)  	V (h^-3 Gpc^3)
#  2.0		3.589				193,6
#  2.2		3.779				226,03
#  2.4		3.953				258,68
#  2.6		4.113				291,37
#  2.8		4.260				323,89
#
#  The volumes between those redshifts are almost the same: for 4500 deg^2,
#  V -> 3.5 h^-3 Gpc^3 . The radial distance between slices is:
#
#  2.0-2.2	0.200 h^-1 Gpc
#  2.2-2.4	0.174 h^-1 Gpc
#  2.4-2.6	0.160 h^-1 Gpc
#  2.6-2.8	0.147 h^-1 Gpc
#
#  Which means that the area of the x-y plane for each map is L_xy^2, where L_xy is:
#
#  2.0-2.2	4.18 h^-1 Gpc
#  2.2-2.4	4.48 h^-1 Gpc
#  2.4-2.6	4.68 h^-1 Gpc
#  2.6-2.8	4.88 h^-1 Gpc
#
#  With 20 h^-1 Mpc cells, we arrive at geometries of:
#
#  2.0-2.2	560 x 560 x 28
#  2.2-2.4	666 x 666 x 26
#  2.4-2.6	716 x 716 x 25
#  2.6-2.8	780 x 780 x 24
#
# These are the physical characteristics of the rectangular box in which
# the data (true or sims) is circumscribed.

# Cell physical size, in units of h^-1 Mpc
cell_size = 10.0

# Number of cells in x,y and z directions.
# Must be integer.
n_x = 190
n_y = 172
n_z = 67

# Displacement of the origin (0,0,0) of the box with respect to Earth
# (in cell units; doesn't need to be integer)
n_x_orig= -95.22820039444235
n_y_orig= -94.77136522090437
n_z_orig= 271.49218306692416


# Volume occupied by tracers in this map: 1.635 h^-3 Gpc^3 (74.7% of the full box)
################################


################################################################
#   Selection function parameters.
#
#   If the *GALAXY/TRACER* selection function is given ("data" selection function):
#sel_fun_data = True
sel_fun_data = False

# If sel_fun_data = True, then specify file with data selection function
# (must be a grid identical to the one defined in the previous section:
# nx X ny X nz , one for each type of galaxy
sel_fun_file = "sel_fun_highz_z1.6.hdf5"

# If sel_fun_data=False, then an *analytical fit* for the selection function is used.
#   ATTENTION: selection function fit parameters are ALWAYS in UNITS of GRID CELLS !!!
#nbar = ntracers*[1.0,2.0]
ncentral = ntracers*[ 50.0 ]
nsigma   = ntracers*[1000000.0]


## !! NEW !!
# If low-count cells must be masked out, then cells with counts 
# below this threshold will be eliminated from the mocks AND from the data
cell_low_count_thresh = 0.0

# One may add a shift and/or a multiplicative factor, for testing:
#   n_bar --> factor * (n_bar + shift)
# N.B.: shift in units of number/(cell size)^3 , NOT number/(h^-3 Mpc^3)
mult_sel_fun, shift_sel_fun = 5.0 , 0.0

################################



################################################################
# Galaxy bias -- in the map-creating tool, halo bias is given in the file;
# and galaxy bias can be computed from the HODs and halo bias.
#
# HOWEVER, if you want to assume another fiducial value for the bias
# in the power spectrum estimation, specify it here, and this will
# SUPERSEDE the bias from the HOD calculation -- ONLY FOR THE ESTIMATION TOOL!
#
# bias = [1.0,2.0]
#
# You may also define a DATA bias that is different from the mocks
#
# data_bias = [1.0,2.0]


# Here, define the interval in k that you want to use to estimate the bias
kmin_bias = 0.05
kmax_bias = 0.15

################################################################
# Shot noise and 1-halo term
#
# Shot noise subtraction can be made to be more or less conservative.
# For this, employ the "shot_fudge" parameter below.
# shot_fudge=0.0 is a conservative shot-noise subtraction
# shot_fudge=1.0 is very agressive
shot_fudge=ntracers*[0.0]

# 1-halo term amplitude -- again, this can be computed from the HOD.
# But if you want to assume some other value, define it here.
# If nonzero, it will be added to the theoretical expectation of the power spectra

#gal_p1h = [10.0,10.0,10.0,10.0]




################################################################
#   Target value of k where the power estimation will be optimal
# (units of h/Mpc)
kph_central = 0.1
# Binning in k -- this should be ~ 1.4/(smallest side of box). In units of h^-1 Mpc
dkph_bin = 0.01
# Max k -- this should be < 2 \pi * Nyquist frequency = \pi /cell_size. In units of h^-1 Mpc
kmax_phys = 0.3
# Min k -- this should be > 1.0/box_size. In units of h^-1 Mpc
kmin_phys = 0.02

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
# matgrowcentral=0.001
# Leave commented if you want f(z) given by the assumed cosmological model
# If computing the power spectrum of halos from an N-body simulation in real space, set this to zero.
################################


################################################################
# Central (mean, or median) redshift of the catalog or simulated data, and width of that bin
zcentral = 1.6
zbinwidth = 0.2
################################


################################################################
# Other parameters useful for power spectrum estimation
# NEED TO IMPLEMENT TOOLS TO ACTUALLY USE THIS!
#
# Gaussian redshift errors of GALAXIES/TRACERS created by HOD
# Units of the cell size!
sigz_est = ntracers*[0.0000001]
# Alpha-type dipoles
adip = ntracers*[0.0000000001]
# Velocity dispersions for RSDs: f*mu^2 --> f*mu^2/(1+v^2*k^2*mu**2)
vdisp = adip

#
# Redshift errors and dipoles of HALOS
halos_sigz_est = nhalos*[0.00001]
# Alpha-type dipoles
halos_adip = halos_sigz_est
# Velocity dispersions for RSDs: f*mu^2 --> f*mu^2/(1+v^2*k^2*mu**2)
halos_vdisp = halos_sigz_est

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
#power_jing_sims = 1.65
power_jing_sims = 2.0

# Power used for data window function:
power_jing_data = 2.0
################################


###############################################################################
# Whether or not to plot the 2D covariances (FKP v. MT)
plot_all_cov = False
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
w0= -0.9999
w1= 0.0
Omegab=0.048206
Omegac=0.26377
H0=67.556
n_SA=0.96
ln10e10ASA=3.0978
z_re=9.9999
k_min_camb = 1.e-4
k_max_camb = 1.e+0
gamma = 0.5454

