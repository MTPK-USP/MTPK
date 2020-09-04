#! /usr/bin/env python
# -*- coding: utf-8 -*-
#################################################
#   This is the HOD configu file for the MTPK codes   #
#################################################


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

mass_fun_file = "TEST_Tinker_HMF_z1.6.dat"
halo_bias_file = "TEST_Tinker_Hbias_z1.6.dat"
hod_file = "Highz_hod.dat"

# Number of halo bins
nhalos = 6
halos_ids = ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']

# Number of galaxy/tracer types
ngals = 2
gal_ids = ['G','Q']


# Total number of tracers ("galaxies") in dataset
# Used by code create_grid_randoms_weights.py (which generates the grid for the selection function)
Ntot_Gals = [1847827,1258801]


########################################################################
#     Number of *lognormal simulations* of the maps for halo type
n_maps = 20
########################################

# Cell physical size, in units of h^-1 Mpc
cell_size = 10.0

# Number of cells in x,y and z directions.
# Must be integer.
n_x = 189
n_y = 171
n_z = 66

# Displacement of the origin (0,0,0) of the box with respect to Earth
# (in cell units; doesn't need to be integer)
n_x_orig= -95.22820039444235
n_y_orig= -94.77136522090437
n_z_orig= 271.49218306692416
################################


################################################################
#   Selection function parameters.
#
#   If the *GALAXY* selection function is given ("data" selection function):
sel_fun_data = True
#sel_fun_data = False

# If sel_fun_data = True, then specify file with data selection function
# (must be a grid identical to the one defined in the previous section:
# nx X ny X nz , one for each type of galaxy
sel_fun_file = "sel_fun_grid_counts.hdf5"


# If sel_fun_data=False, then an *analytical fit* for the selection function is used.
#   ATTENTION: selection function fit parameters are ALWAYS in UNITS of GRID CELLS !!!
#ncentral = [10.0,1.0]
#nsigma   = [20000.0,20000.0]


# One may add a multiplicative factor, e.g., for testing:
#   n_bar --> factor * n_bar
# N.B.: shift in units of number/(cell size)^3 , NOT number/(h^-3 Mpc^3)
mult_sel_fun = 1.0 , 0.0
