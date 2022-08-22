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
from scipy.ndimage import gaussian_filter

from mass_assign import build_grid , grid_pos_s , weights
import grid3D as gr

#if sys.platform == "darwin":
#	import pylab as pl
#	from matplotlib import cm
#else:
#	import matplotlib
#	matplotlib.use('Agg')
#	from matplotlib import pylab, mlab, pyplot
#	from matplotlib import cm
#	from IPython.display import display
#	from IPython.core.pylabtools import figsize, getfigs
#	pl=pyplot


####################################################################################################
#
# User definitions
#

####################
# Files containing the catalogs

# Or, use glob and read all catalogs in some directory 
#handle = "subbox"
#handle = "sub"

this_dir = os.getcwd()

#filenames_catalogs = np.sort(glob.glob(this_dir + '/catalogs/Uchuu_age/*' + handle + '*'))
#filenames_catalogs = np.sort(glob.glob('/Users/lrwa/trabalho/JPAS-LSS/Pinocchio/catalogs/Mass-12p5-15_redshift-0p1-1p0/Pin*/mass*.dat'))
#filenames_catalogs = np.sort(glob.glob('/Users/lrwa/trabalho/JPAS-LSS/stage2/halos_unitsim.txt'))
filenames_catalogs = np.sort(glob.glob('data/lightcone/Lightcone_RA_DEC_z.dat'))

if len(filenames_catalogs)==0 :
	print ('Files not found! Aborting now...')
	sys.exit(-1)

# Specify how many tracers, how many catalogs, and which files refer to which catalog/tracer
# Final format should be filenames(cat,tracer)

# Number of maps or subboxes
Ncats = 1

# Number of tracers FOR EACH CATALOG (choose 1 if there is a single catalog with many tracers)
ntracers_catalog = 1
# Create grids for those tracers from a single catalog
split_tracers = False
# Number of tracers that will be created from the catalog
ntracers_grid = 1
# Specify columns for Mass, if splitting catalog
col_tracer = 4
# Specify criterium to split tracers: e.g., mass bins
tracer_bins = [1.e11,1.e13,1.e15]

# This line should be rewritten if the order of the files doesn't match the required
# ordering: first index is catalog/sub-box, second index is tracer
filenames_catalogs = np.reshape(filenames_catalogs,(Ncats,ntracers_catalog))

col_r = 0
col_theta = 2
col_phi = 1

# Angles in radians or degrees? If in degrees, use factor
#angle_factor = 1.0
angle_factor = np.pi/180.

# min and max of coordinates for the catalogs
x_cat_min , y_cat_min , z_cat_min = -500.0 , -500.0 , -500.0
x_cat_max , y_cat_max , z_cat_max = 500.0 , 500.0 , 500.0

# Pinocchio:
#x_cat_min , y_cat_min , z_cat_min = -1560.0 , -1560.0 , 1440.0
#x_cat_max , y_cat_max , z_cat_max = 1560.0 , 1560.0 , 2208.0

# Mask out some redshift range?
mask_radial = False
#rmin = 0.0
#rmax = 400.0

# Mask out cells outside bounds of the box?
mask_spillover_cells = False


# Mass assignement method:
# Nearest Grid Point ("NGP")
# Clouds in Cell ("CIC")
# Triangular Shaped Cloud ("TSC")
# Piecewise Cubic Spline ("PCS")
#mas_method = "TSC"

# Batch size for mass assignement
batch_size = 1000000

# Which directory to write grids to:
#dir_out = this_dir + "/maps/sims/Uchuu_age/"
#dir_out = this_dir + "/maps/sims/Pinocchio_z09_NGP/"
dir_out = this_dir + "/maps/sims/Unit_LoS_TSC/"
filenames_out = "Data"


# File to save mean total number of tracers in maps
file_ntot = this_dir + "/catalogs/Unit_sims/Ntot_Unit_LoS_TSC.txt"


####################
# Input file with grid properties (cell size, box dimensions, cosmological model etc.) 
#input_filename = "Uchuu_age_416_CiC"
#input_filename = "Unit_sims_NGP"
#input_filename = "Unit_RSD_TSC"
input_filename = "Unit_LoS_TSC"

####
#
# Boundary method: wrap galaxies around (i.e., use periodic B.C.)?
wrap=False

#
# (End of user definitions)
#
####################################################################################################






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


# Read input file
this_dir = os.getcwd()
input_dir = this_dir + '/inputs'
sys.path.append(input_dir)

print("Using input file: ", input_filename)

try:
	exec ("from " + input_filename + " import *")
except:
	print()
	print("Failed importing inputs! Check file:", input_filename)
	print("Aborting now...")
	sys.exit(-1)


try:
	padding_length
except:
	padding_length = 0


# Now create grids: on each cell we compute the number of each tracer in each shell;

# Grid binning
xbins = cell_size * np.arange(n_x + 1)
ybins = cell_size * np.arange(n_y + 1)
zbins = cell_size * np.arange(n_z + 1)

print("Dimensions of the grids: n_x, n_y, n_z =",n_x,n_y,n_z)
print()
print("The actual catalog spans the ranges in x,y,z:")
print("x:", x_cat_min , "-->",x_cat_max)
print("y:", y_cat_min , "-->",y_cat_max)
print("z:", z_cat_min , "-->",z_cat_max)
print()
print()
print("With the padding length, of ", padding_length, "cells, the box will be filled with:")
print("x: (",padding_length, "* 0 ,", n_x-2*padding_length, ",", padding_length,"*0)")
print("y: (",padding_length, "* 0 ,", n_y-2*padding_length, ",", padding_length,"*0)")
print("z: (",padding_length, "* 0 ,", n_z-2*padding_length, ",", padding_length,"*0)")
print()
print("Check: given the padding, your catalog should end at cartesian positions:")
print("max(x) =", xbins[-1] - 2*padding_length*cell_size )
print("max(y) =", ybins[-1] - 2*padding_length*cell_size )
print("max(z) =", zbins[-1] - 2*padding_length*cell_size )
print()
print("Origin (0,0,0) of box will be considered to be displaced from the observer @Earth")
print("by these numbers of cells in each direction:    (This affects RSDs!)")
print("n_x_orig=" , n_x_orig)
print("n_y_orig=" , n_y_orig)
print("n_z_orig=" , n_z_orig)
print()

mystr=filenames_catalogs[0,0]

if mystr[-4:] == "hdf5":
	use_h5py = True
elif mystr[-4:] == ".dat":
	use_h5py = False
elif mystr[-4:] == ".cat":
	use_h5py = False
elif mystr[-4:] == ".txt":
	use_h5py = False
else:
	print('Sorry, I cannot recognize the format of the catalogs (should be .hdf5 , .dat , .cat , .txt)')
	print("Aborting now")
	sys.exit(-1)

if mas_method == "NGP":
	print()
	print("Mass assignement: Nearest Grid Point (NGP)...")
	print()
elif mas_method == "CIC":
	print()
	print("Mass assignement: Clouds in Cell (CIC)...")
	print()
elif mas_method == "TSC":
	print()
	print("Mass assignement: Triangular Shaped Cloud (TSC)...")
	print()
elif mas_method == "PCS":
	print()
	print("Mass assignement: Piecewise Cubic Spline (PCS)...")
	print()
else:
	print("Sorry, I don't recognize the mass assignement choise. Please check code preamble above.")
	print("Aborting now...")
	print()
	sys.exit(-1)

yesno = input(" Continue? Y or N :")
if yesno != "Y":
	print('Aborting now.')
	sys.exit(-1)
print()


zmin = np.around(zcentral - zbinwidth,5)
zmax = np.around(zcentral + zbinwidth,5)

mean_counts  = np.zeros((ntracers_grid,n_x,n_y,n_z))
box=(n_x,n_y,n_z)


# Make grid; may be useful later
L_x = n_x*cell_size ; L_y = n_y*cell_size ; L_z = n_z*cell_size
grid = gr.grid3d(n_x,n_y,n_z,L_x,L_y,L_z)

LX0 = n_x_orig
LY0 = n_y_orig
LZ0 = n_z_orig + 0.01

grid_radius=np.sqrt((LX0 + grid.RX)**2 + (LY0 + grid.RY)**2 + (LZ0 + grid.RZ)**2)



# Now, loop over sets of catalogs and tracers
for nc in range(Ncats):
	print("Processing catalog #", nc)
	counts  = np.zeros((ntracers_grid,n_x,n_y,n_z))
	for nt in range(ntracers_catalog):
		print("Reading tracer #",nt)
		try:
			#h5map = h5py.File(filenames_catalogs[nc,nt], 'r')
			#h5data = h5map.get(list(h5map.keys())[0])
			#cat = np.asarray(h5data,dtype='float32')
			#h5map.close
			if use_h5py:
				f = h5py.File(filenames_catalogs[nc,nt], 'r')
				tracer_r, tracer_theta, tracer_phi = np.array(f['r']), np.array(f['theta']), np.array(f['phi'])
				if split_tracers:
					tracer_type = np.array(f['m'])
				f.close()
				if mask_radial:
					mask = np.where( (tracer_r >= rmin) & (tracer_r <= rmax))[0]
					tracer_r = tracer_r[mask]
					tracer_theta = tracer_theta[mask]
					tracer_phi = tracer_phi[mask]
					if split_tracers:
						tracer_type = tracer_type[mask]

			else:
				f = filenames_catalogs[nc,nt]
				this_cat = np.loadtxt(f)
				# Fix dimensions if catalog has switched lines/columns
				if this_cat.shape[0] < this_cat.shape[1]:
					this_cat = this_cat.T
				tracer_r, tracer_theta, tracer_phi = this_cat[:,col_r] , this_cat[:,col_theta] , this_cat[:,col_phi]
				# Redshift mask
				if split_tracers:
					tracer_type = this_cat[:,col_tracer]
				if mask_radial:
					mask = np.where( (tracer_r >= rmin) & (tracer_r <= rmax))[0]
					tracer_r = tracer_r[mask]
					tracer_theta = tracer_theta[mask]
					tracer_phi = tracer_phi[mask]
					if split_tracers:
						tracer_type = tracer_type[mask]
		except:
			print("Could not read file:" , filenames_catalogs[nc,nt])
			sys.exit(-1)

		ntot = len(tracer_r)
		print("Original catalog has", ntot,"objects")

		tracer_x = tracer_r*np.sin(np.pi/2 - angle_factor*tracer_theta)*np.cos(angle_factor*tracer_phi)
		tracer_y = tracer_r*np.sin(np.pi/2 - angle_factor*tracer_theta)*np.sin(angle_factor*tracer_phi)
		tracer_z = tracer_r*np.cos(np.pi/2 - angle_factor*tracer_theta)

		tracer_x = tracer_x - x_cat_min + cell_size*padding_length
		tracer_y = tracer_y - y_cat_min + cell_size*padding_length
		tracer_z = tracer_z - z_cat_min + cell_size*padding_length

		# Mask out cells outside bounds of the box
		if mask_spillover_cells:
			posx = np.where( (tracer_x <= 0) | (tracer_x >= xbins[-1]) )[0]
			posy = np.where( (tracer_y <= 0) | (tracer_y >= ybins[-1]) )[0]
			posz = np.where( (tracer_z <= 0) | (tracer_z >= zbins[-1]) )[0]

			boxmask = np.union1d(posx,np.union1d(posy,posz))

			tracer_x=np.delete(tracer_x,boxmask)
			tracer_y=np.delete(tracer_y,boxmask)
			tracer_z=np.delete(tracer_z,boxmask)
			ntot = len(tracer_x)
			print("After trimming inside box, the catalog has", ntot,"objects")
			print(" Now building grid box...")

		if split_tracers:
			print("Splitting tracers in catalog and generating grids...")
			for ntg in range(ntracers_grid):
				print(" ... tracer",ntg)
				pos_tracers = np.where( (tracer_type > tracer_bins[ntg]) & (tracer_type <= tracer_bins[ntg+1]) )[0]
				tx, ty , tz = tracer_x[pos_tracers] , tracer_y[pos_tracers] , tracer_z[pos_tracers]
				ntot_tracer = len(pos_tracers)
				counts[ntg] = build_grid(np.array([tx,ty,tz]).T,cell_size,box,mas_method,batch_size,wrap)
				mean_counts[ntg] += counts[ntg]
				print("... after placing objects in grid there are", np.int0(np.sum(counts[ntg])), "objects.")
				print("Final/original number:", np.around(100.*np.sum(counts[ntg])/ntot_tracer,2), "%")
			del tracer_x, tracer_y, tracer_z
		else:
			counts[nt] = build_grid(np.array([tracer_x,tracer_y,tracer_z]).T,cell_size,box,mas_method,batch_size,wrap)
			mean_counts[nt] += counts[nt]
			del tracer_x, tracer_y, tracer_z
			print("... after placing objects in grid there are", np.int0(np.sum(counts[nt])), "objects.")
			print("Final/original number:", np.around(100.*np.sum(counts[nt])/ntot,2), "%")
			print()

	# Now write these catalogs into a single grid
	if len(str(nc))==1:
		map_num = '00' + str(nc)
	elif len(str(nc))==2:
		map_num = '0' + str(nc)
	else:
		map_num = str(nc)


	print("Saving grid of counts to file:",dir_out + filenames_out + "_grid_" + map_num + ".hdf5")
	h5f = h5py.File(dir_out + filenames_out + "_grid_" + map_num + ".hdf5",'w')
	h5f.create_dataset('grid', data=counts, dtype='float32', compression='gzip')
	h5f.close()

	print()


mean_counts = mean_counts/Ncats
tot_counts = np.sum(mean_counts,axis=0)

# Mean total number of tracers of each type in box
np.savetxt(file_ntot,np.sum(mean_counts,axis=(1,2,3)))

print("Number of tracers per cell:", np.sum(mean_counts,axis=(1,2,3))/n_x/n_y/n_z)

print()
print("...done!")
print()
print("++++++++++++++++++++")
print()
