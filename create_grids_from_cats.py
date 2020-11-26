#! /usr/bin/env python
# -*- coding: utf-8 -*-
########################################
# This code creates a grid (nx , ny, nz) of cells of size cell_size, 
# containint the selection functions of any number of tracers (Nt).
#
# The output has the format (Nt,nx,ny,nz)
#
# INPUTS:
#
#  - files "catalog_XXX_1.hdf5", "catalog_XXX_2.hdf5", ..., which are 
#    table with any number of "points" with columns given by:
#    (...) (theta) ... (phi) ... (redshift) ... (weight) ...
#
#  - standard input file containing grid properties, cosmological model, etc.
#
########################################
import numpy as np
import h5py
import sys
import os
from cosmo_funcs import *
import glob

if sys.platform == "darwin":
    import pylab as pl
    from matplotlib import cm
else:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pylab, mlab, pyplot
    from matplotlib import cm
    from IPython.display import display
    from IPython.core.pylabtools import figsize, getfigs
    pl=pyplot


####################################################################################################
#
# User definitions
#

####################
# Files containing the catalogs
#filenames_catalogs = ['TESTCAT_001_tracer1.hdf5','TESTCAT_001_tracer2.hdf5','TESTCAT_002_tracer1.hdf5','TESTCAT_002_tracer2.hdf5','TESTCAT_003_tracer1.hdf5','TESTCAT_003_tracer2.hdf5']

# Or, use glob and read all catalogs in some directory 
handle = "Map"

this_dir = os.getcwd()

filenames_catalogs = glob.glob(this_dir + '/catalogs/SHUFFLE_Highz_z16_gals/*' + handle + '*')

if len(filenames_catalogs)==0 :
    print ('Files not found! Aborting now...')
    sys.exit(-1)

# Specify how many tracers, how many catalogs, and which files refer to which catalog/tracer
# Final format should be filenames(cat,tracer)
filenames_catalogs = np.sort(np.array(filenames_catalogs))
Ncats = 100
Ntracers = 2
filenames_catalogs = np.reshape(filenames_catalogs,(Ncats,Ntracers))

# Specify columns for theta, phi, z and weights
col_theta = 0
col_phi = 1
col_z = 2

# Do you have weights? If yes, which column? Also, specify bins for the weights
do_weights = False
col_w = 3
wbins = np.arange(0.5,1.5,0.1)

# Re-impose redshift mask? (I.e., demand that redshifts fall within the bounds?)
re_mask_redshift = False

# Which directory to write grids/weight to:
dir_out = this_dir + "/maps/sims/SHUFFLE_Highz_z16_gals/"
filenames_out = "SHUFFLED"


####################
# Input file with grid properties and cosmological model
input_filename = "SHUFFLE_Highz_z16_gals"



####################################################################################################
# Start main code
#
#

print()
print("Will load maps stored in files:")
print(filenames_catalogs)
yesno = input(" Continue? Y or N :")
if yesno != "Y":
	print('Aborting now.')
	sys.exit(-1)
print()


if not os.path.exists(dir_out):
    os.makedirs(dir_out)

# Weight bin centers
if do_weights == True:
	wcent = 0.5*(wbins[1:] + wbins[:-1])

# Read input file
this_dir = os.getcwd()
input_dir = this_dir + '/inputs'
sys.path.append(input_dir)

try:
	exec ("from " + input_filename + " import *")
except:
	print("Failed importing inputs. Check file:", input_filename)


# Determine boundaries of box in (x,y,z)
# By definition, the box STARTS in the Cartesian z direction,
# at the lowest value of z(cart) 
# By definition, the map is already centralized around the z axis, and is 


zred_min = zcentral - zbinwidth
zred_max = zcentral + zbinwidth

print()
print("Using the redshift range z_min, z_max:", zred_min, zred_max)

Omegam = Omegab + Omegac

chi_min = chi_h(Omegam, 1-Omegam-Omegak, w0, w1, zred_min)
chi_max = chi_h(Omegam, 1-Omegam-Omegak, w0, w1, zred_max)


xcart_min = n_x_orig*cell_size
ycart_min = n_y_orig*cell_size
zcart_min = n_z_orig*cell_size

xcart_max = xcart_min + n_x*cell_size
ycart_max = ycart_min + n_y*cell_size
zcart_max = zcart_min + n_z*cell_size

# Now create grids: on each cell we compute (1) the number of each tracer in each shell;
# (2) the mean weights of those tracers in that shell

# Grid binning
xbins = np.arange(xcart_min,xcart_max + cell_size/2. , cell_size)
ybins = np.arange(ycart_min,ycart_max + cell_size/2. , cell_size)
zbins = np.arange(zcart_min,zcart_max + cell_size/2. , cell_size)


print("Dimensions of the grids: n_x, n_y, n_z =",n_x,n_y,n_z)
print()
print("Origin (0,0,0) of box displaced from the observer @Earth by this # of cells:")
print("n_x_orig=" , n_x_orig)
print("n_y_orig=" , n_y_orig)
print("n_z_orig=" , n_z_orig)
print()
yesno = input(" Continue? Y or N :")
if yesno != "Y":
	print('Aborting now.')
	sys.exit(-1)
print()


# Now, loop over sets of catalogs and tracers
for nc in range(Ncats):
	print("Processing catalog #", nc)
	counts  = np.zeros((Ntracers,n_x,n_y,n_z))
	if do_weights:
		weights = np.zeros((Ntracers,n_x,n_y,n_z))
	for nt in range(Ntracers):
		print("Reading catalog for tracer",nt)
		try:
			h5map = h5py.File(filenames_catalogs[nc,nt], 'r')
			h5data = h5map.get(list(h5map.keys())[0])
			cat = np.asarray(h5data,dtype='float32')
			h5map.close
		except:
			print("Could not read file:" , filenames_catalogs[nc,nt])
		ntot = len(cat)
		print("Original catalog has", ntot,"objects")
		# Select the objects in the catalog that fall into this redshift
		if re_mask_redshift:
			redshift_mask = np.where( (cat[:,col_z] >= zred_min) & (cat[:,col_z] <= zred_max) )[0]
		else:
			redshift_mask = np.where( cat[:,col_z] >= 0 )[0]
		if len(redshift_mask) < 1:
			print("Warning! There are no tracers in the chosen redshift range. Try again. Aborting now...")
			sys.exit(-1)
		tracer_angles  = cat[redshift_mask][:,(col_theta,col_phi)]
		tracer_redshifts  = cat[redshift_mask][:,col_z]
		
		if do_weights:
			tracer_weights = cat[redshift_mask][:,col_w]
		del redshift_mask
		print(" ...converting redshifts and angles to Cartesian coordinates for given cosmological model...")
		tracer_chi = chi_h(Omegam, 1-Omegam-Omegak, w0, w1, tracer_redshifts)
		del tracer_redshifts
		print(" ...computing histograms inside the grid box...")
		tracer_x = tracer_chi*np.sin(tracer_angles[:,0])*np.cos(tracer_angles[:,1])
		tracer_y = tracer_chi*np.sin(tracer_angles[:,0])*np.sin(tracer_angles[:,1])
		tracer_z = tracer_chi*np.cos(tracer_angles[:,0])
		del tracer_chi, tracer_angles
		if do_weights:
			histo = np.histogramdd(np.array([tracer_x,tracer_y,tracer_z,tracer_weights]).T,bins=(xbins,ybins,zbins,wbins))[0]
		else:
			histo = np.histogramdd(np.array([tracer_x,tracer_y,tracer_z]).T,bins=(xbins,ybins,zbins))[0]
		#print("Min, max in Cartesian x, y, z for this tracer:")
		#print(np.min(tracer_x),np.max(tracer_x))
		#print(np.min(tracer_y),np.max(tracer_y))
		#print(np.min(tracer_z),np.max(tracer_z))
		del tracer_x, tracer_y, tracer_z
		if do_weights:
			counts[nt]  = np.sum(histo,axis=3)
			weights[nt] = 1./(0.00001 + counts[nt]) * np.sum(histo * wcent,axis=3)
		else:
			counts[nt]  = histo
		del histo
		print("... after placing objects in grid there are", np.int0(np.sum(counts[nt])), "objects.")
		print("Final/original number:", np.around(100.*np.sum(counts[nt])/ntot,2), "%")
	# Now write these catalogs into a single grid
	if len(str(nc))==1:
		map_num = '00' + str(nc)
	elif len(str(nc))==2:
		map_num = '0' + str(nc)
	else:
		map_num = str(nc)

	print("Saving grid of counts to file:",dir_out + filenames_out + "_grid_" + map_num + ".hdf5")
	h5f = h5py.File(dir_out + filenames_out + "_grid_" + map_num + ".hdf5",'w')
	h5f.create_dataset('grid', data=counts, dtype='int32',compression='gzip')
	h5f.close()
	if do_weights:
		print("Saving grid of weights to file:",dir_out + filenames_out + "_grid_weights_" + map_num + ".hdf5")
		h5f = h5py.File(dir_out + filenames_out + "_grid_weights_" + map_num + ".hdf5",'w')
		h5f.create_dataset('grid', data=weights, dtype='float32',compression='gzip')
		h5f.close()

	print()

print()
print("...done!")
print()
print("++++++++++++++++++++")
print()
