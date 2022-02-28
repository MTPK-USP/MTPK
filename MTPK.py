#! /usr/bin/env python
# -*- coding: utf-8 -*-
#########################################################
# This is the level-0 input file for the MCMaps code    #
#########################################################

#########################################################
#########################################################
# Handles (names) for this estimation.
#


#########################################################
# The "data" handle specifies the real/N-body maps and data selection function and real maps
# which should be found in /maps/data/
# If running simulations-only, with a real selection function,
# the real maps will be ignored; if running simulations-only with a
# theory selection function, this filename is irrelevant.
#
# The SIMS handle will be used to:
# (1) Find the input file with the main properties of the run (in /inputs)
# (2) Save those parameters, as well as spectra, in /spectra
# (3) The lognormal simulations will be saved in /maps/sims/
# (4) When running MTPK_estimate, results will be saved in /spectra and /figures

# This is the directory where the data files and data selection function are stored
# This is only called if either sims_only=False, or if sel_fun_data=True

#handle_data = "Highz_z16"
#handle_data = "Highz_z16_gals"
#handle_data = "SHUFFLE_Highz_z16_gals"
handle_data = "ExSHalos"
# handle_data = "PM"

#handle_data = "Highz_z20"
#handle_data = "Highz_z20_gals"
#handle_data = "SHUFFLE_Highz_z20_gals"

#handle_data = "Highz_z24"
#handle_data = "Highz_z24_gals"
#handle_data = "SHUFFLE_Highz_z24_gals"

# ATTENTION: data files should be stored in hdf5 (N_tracers,nx,ny,nz) tables,
# with names ending in ..._DATA.hdf5
#########################################################


#########################################################
# The "sims" handle points to the directories (/maps/sims) where the simulated maps
# (and, if you used our lognormal map-creation tool, the spectral corrections)
# N.B.: Sims are stored in (N_maps) files of format (N_tracers,nx,ny,nz), in hdf5 format (XXX.hdf5) 
# The estimation tool will look at this directory for the simulated maps.

handle_sims = handle_data

#handle_sims = "SDSSHalos2_z3_H"
#handle_sims= "MD_Big_box_Vm_age"
#handle_sims= "MD_Big_box_Vm_c200"
#handle_sims= "MD_Big_box_Vm_spin"
#########################################################


#########################################################
# The "estimates" handle determines the input file (XXX.py) with the ASSUMED properties of the maps.
# (It can be, and often is, identical to handle_sims)
# This handle also points to the directories of the same name which will store
# the estimations of the power spectra, the figures, and mains results.

handle_estimates = handle_data
#handle_estimates = "Renan_2tracers_JPAS_z035_v11_comb"


#handle_estimates= "JPAS_z_05_halos"
#handle_estimates= "JPAS_z_05"

#handle_estimates = "MD_Small_box_Vm_age"
#handle_estimates = "MD_Small_box_Vm_c200"
#handle_estimates = "MD_Small_box_Vm_spin"

#########################################################

