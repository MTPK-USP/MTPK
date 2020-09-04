#! /usr/bin/env python
# -*- coding: utf-8 -*-
########################################
# This code creates catalogs from grids of cells of size cell_size, 
#
# The output has the format (theta,phi,redshift)
#   or (theta,phi,redshift,weight) if you also give the weights
#
# INPUTS:
#
#  - files "GRID_XX1.hdf5", "GRID_XX2.hdf5", ..., which are grids of format
#    (Ntracers,n_x,n_y,n_z) 
#
#  - standard input file containing grid properties, cosmological model, etc.
#
########################################
import numpy as np
import h5py
import sys
import os
import glob
from scipy import interpolate

from cosmo_funcs import *

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
# Input file with grid properties and cosmological model
input_filename = "Highz_z16_gals"


####################
# Directory and files containing the catalogs
dir_in = "maps/sims/Highz_z16_gals/"

#filenames_grids = ["TESTCAT_grid_weights0.hdf5","TESTCAT_grid_weights1.hdf5","TESTCAT_grid_weights2.hdf5","TESTCAT_grid0.hdf5","TESTCAT_grid1.hdf5","TESTCAT_grid2.hdf5"]
#filenames_weights = ["TESTCAT_grid_weights0.hdf5","TESTCAT_grid_weights1.hdf5","TESTCAT_grid_weights2.hdf5"]

# Or, use glob and read all catalogs in some directory 
handle = "rid_"


# Specify how many tracers and how many maps
Nmaps = 100
Ntracers = 2


# Which directory to write grids/weight to:
dir_out = "catalogs/Highz_z16_gals"

# Name to use for these catalogs
cat_handle = "CAT_Highz_z16_gals" 


####################################################################################################
# Start main code
#
#

this_dir = os.getcwd()

dir_in = this_dir + "/" + dir_in 
dir_out = this_dir + "/" + dir_out

filenames_grids = glob.glob(dir_in + '*' + handle + '*')
filenames_weights = glob.glob(dir_in + '*' + handle + '*weight*')


if len(filenames_grids)==0 :
    print ('Files not found! Aborting now...')
    sys.exit(-1)
filenames_grids = np.sort(np.array(filenames_grids))

if len(filenames_weights)!=0 :
	filenames_weights = np.sort(np.array(filenames_weights))
	filenames_grids = np.setdiff1d(filenames_grids,filenames_weights)
	do_weights = True
else:
	do_weights = False

# Read input file
input_dir = this_dir + '/inputs'
sys.path.append(input_dir)

try:
	exec ("from " + input_filename + " import *")
except:
	print("Failed importing inputs. Check file:", input_filename)


if not os.path.exists(dir_out):
    os.makedirs(dir_out)


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
xpos = np.arange(xcart_min + cell_size/2. , xcart_max , cell_size)
ypos = np.arange(ycart_min + cell_size/2. , ycart_max , cell_size)
zpos = np.arange(zcart_min + cell_size/2. , zcart_max , cell_size)


print("Assuming dimensions of the grids: n_x, n_y, n_z =",n_x,n_y,n_z)
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



# Build grid coordinates
r_x = xcart_min + cell_size*(0.5+np.arange(n_x))
r_y = ycart_min + cell_size*(0.5+np.arange(n_y))
r_z = zcart_min + cell_size*(0.5+np.arange(n_z))

identx = np.ones_like(r_x)
identy = np.ones_like(r_y)
identz = np.ones_like(r_z)

RX = np.einsum('i,j,k', r_x,identy,identz)
RY = np.einsum('i,j,k', identx,r_y,identz)
RZ = np.einsum('i,j,k', identx,identy,r_z)

RX2 = np.einsum('i,j,k', r_x*r_x,identy,identz)
RY2 = np.einsum('i,j,k', identx,r_y*r_y,identz)
RZ2 = np.einsum('i,j,k', identx,identy,r_z*r_z)

grid_r = np.sqrt(RX2 + RY2 + RZ2)

xflat = RX.flatten()
yflat = RY.flatten()
zflat = RZ.flatten()
rflat = grid_r.flatten()

del r_x, r_y, r_z, identx, identy, identz, RX, RY, RZ, grid_r

phiflat = np.arctan2(yflat,xflat)
thetaflat = np.arccos(zflat/rflat)

# Create an interpolation function for the redshift as a function of radius
dz=0.0001
zrange = np.arange(0.0,2*zred_max,dz)
rrange = chi_h(Omegam, 1-Omegam-Omegak, w0, w1, zrange)

zs_interp = interpolate.interp1d(rrange, zrange)

#These are the redshifts at each point in the grid, given the cosmological model
zred_flat = zs_interp(rflat)

# Now, loop over maps (grids)
for nm in range(Nmaps):
	print("Reading map/grid #", nm)
	try:
		h5map = h5py.File(filenames_grids[nm], 'r')
		h5data = h5map.get(list(h5map.keys())[0])
		grid = np.asarray(h5data,dtype='int32')
		h5map.close
	except:
		print("Could not read file with grid:" , filenames_grids[nm])
	if do_weights:
		try:
			h5map = h5py.File(filenames_weights[nm], 'r')
			h5data = h5map.get(list(h5map.keys())[0])
			weights = np.asarray(h5data,dtype='float32')
			h5map.close
		except:
			print("Could not read file with weights:" , filenames_weights[nm])
	# Now get a catalog of objects on each position x,y,z., for each tracer in grid
	for nt in range(Ntracers):
		print("  Processing tracer",nt,"...",)
		flat_counts = grid[nt].flatten()
		if do_weights:
			flat_weights = weights[nt].flatten()
		# Find the cells where there are non-zero number of objects
		mask = np.where( flat_counts != 0)[0]
		if len(mask) == 0:
			print("WARNING! There are no tracers of species", nt, "in this map! Aborting now...")
			sys.exit(-1)
		flat_counts = flat_counts[mask]
		print("  There are", np.sum(flat_counts), "tracers of this type in this grid")
		phi_mask = phiflat[mask]
		theta_mask = thetaflat[mask]
		zred_mask = zred_flat[mask]
		if do_weights:
			weights_mask = flat_weights[mask]
		# First, one line for each cell (since now every cell has at least one tracer)
		# I will use a flattened list/array, that will later be reshaped 
		if do_weights:
			cat = np.ndarray.tolist(np.array([theta_mask,phi_mask,zred_mask,weights_mask]).T.ravel())
		else:
			cat = np.ndarray.tolist(np.array([theta_mask,phi_mask,zred_mask]).T.ravel())
		# Now start adding objects corresponding to the cells where there are more than one tracer
		# Histogram of counts
		counts_bins = np.arange(1.5,np.max(flat_counts)+1,1)
		h=np.histogram(flat_counts,bins=counts_bins)[0]
		h_counts = np.int0(counts_bins[:-1] + 0.5 )
		# Now get the multiplicity of each line by the counts, and add lines corresponding to those
		c_counts = h_counts[h!=0]
		n_counts = h[h!=0]
		for i in np.arange(len(c_counts)):
			pos = np.where(flat_counts == c_counts[i])[0]
			sys.stdout.write(str(n_counts[i]))
			sys.stdout.write(" - ")
			sys.stdout.write(str(c_counts[i]))
			sys.stdout.write(" | ")
			sys.stdout.flush()
			repeat_theta = np.tile(theta_mask[pos],c_counts[i]-1)
			repeat_phi = np.tile(phi_mask[pos],c_counts[i]-1)
			repeat_z = np.tile(zred_mask[pos],c_counts[i]-1)
			if do_weights:
				repeat_w = np.tile(weights_mask[pos],c_counts[i]-1)
				repeat_array = np.ndarray.tolist(np.array([repeat_theta,repeat_phi,repeat_z,repeat_w]).T.ravel())
				cat.extend(repeat_array)
			else:
				repeat_array = np.ndarray.tolist(np.array([repeat_theta,repeat_phi,repeat_z]).T.ravel())
				cat.extend(repeat_array)
		# Now write out these catalogs
		sys.stdout.write("\n")

		cat= np.asarray(cat)
		catlen=len(cat)
		if do_weights:
			nobj=catlen//4
			cat=np.reshape(cat,(nobj,4))
		else:
			nobj=catlen//3
			cat=np.reshape(cat,(nobj,3))			
		print("  Generated catalog with",nobj, "tracers of this type.")

		if len(str(nm))==1:
			map_num = '00' + str(nm)
		elif len(str(nm))==2:
			map_num = '0' + str(nm)
		else:
			map_num = str(nm)
		print("  Saving catalog in file:", dir_out + "/" + cat_handle + "_Map_" + map_num + "_Tracer_" + str(nt) + ".hdf5")

		h5f = h5py.File(dir_out + "/" + cat_handle + "_Map_" + map_num + "_Tracer_" + str(nt) + ".hdf5",'w')
		h5f.create_dataset('catalog', data=cat, dtype='float32',compression='gzip')
		h5f.close()
		print("  ... done!")


print()
print("Success!")
print()
print("++++++++++++++++++++")
print()

