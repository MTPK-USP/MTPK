'''
This code creates a grid (nx , ny, nz) of cells of size cell_size, 
containing the selection functions of any number of tracers (Nt).

The output has the format (Nt, nx, ny, nz)

INPUTS:

 - files "catalog_XXX_YY.dat", ..., for XXX representing the seed and YY the bin, which are 
   table with any number of "points" with columns given by the cartesian coordinates:
   (x) (y) (z)

 - input properties containing the catalogs' properties
'''

import numpy as np
import h5py
import sys
import os
import glob
from scipy.ndimage import gaussian_filter
from mass_assign_beta import build_grid, grid_pos_s, weights

'''
-----------
Parameters
-----------

input_filename : string
 Name of the catalogs

filenames_catalogs : string
 Path to the catalogs using *

Yields
------

NameError
 Raised when an object could not be found

TypeError
 Raised when a function or operation is applied to an object of an incorrect type

ValueError 
 Raised when a function gets an argument of correct type but improper value

'''

def create_grids_from_xyz_cats(cat_specs, my_cosmology, my_code_options,
			       input_filename, filenames_catalogs, dir_out):
	
	'''
	Create grids from xyz cats
	'''
	
	filenames_catalogs = np.sort(glob.glob(filenames_catalogs))
	
	####################
	# Files containing the catalogs

	# Or, use glob and read all catalogs in some directory 

	if len(filenames_catalogs) == 0:
		raise NameError('Files not found!')

	ntracers_catalog = cat_specs.nhalos
	ntracers_grid = cat_specs.nhalos	
	Ncats = cat_specs.n_maps
	use_redshifts = my_code_options.use_redshifts
	col_tracer = cat_specs.col_m
	col_x = cat_specs.col_x
	col_y = cat_specs.col_y
	col_z = cat_specs.col_z
	x_cat_min = cat_specs.x_cat_min
	y_cat_min = cat_specs.y_cat_min
	z_cat_min = cat_specs.z_cat_min
	x_cat_max = cat_specs.x_cat_max
	y_cat_max = cat_specs.y_cat_max
	z_cat_max = cat_specs.z_cat_max
	mask_redshift = my_code_options.mask_redshift
	save_mask = my_code_options.save_mask
	save_mean_sel_fun = my_code_options.save_mean_sel_fun
	split_tracers = my_code_options.split_tracers
	tracer_bins = my_code_options.tracer_bins
	mask_spillover_cells = my_code_options.mask_spillover_cells
	batch_size = my_code_options.batch_size
	wrap = my_code_options.wrap

	# This line should be rewritten if the order of the files doesn't match the required
	# ordering: first index is catalog/sub-box, second index is tracer
	filenames_catalogs = np.reshape(filenames_catalogs, (Ncats, ntracers_catalog)) 

	# Which directory to write grids to:
	filenames_out = "Data"

	print()
	print("Will load maps stored in files:")
	print(filenames_catalogs)

	if not os.path.exists(dir_out):
		os.makedirs(dir_out)

	###################################################
	#Defining all the parameters from new classes here
	###################################################
	#Cosmological quantities
	zcentral = my_cosmology.zcentral
	#Code quantities
	cell_size = my_code_options.cell_size
	n_x = my_code_options.n_x
	n_y = my_code_options.n_y
	n_z = my_code_options.n_z
	V = n_x*n_y*n_z
	n_x_orig = my_code_options.n_x_orig
	n_y_orig = my_code_options.n_y_orig
	n_z_orig = my_code_options.n_z_orig

	mas_method = my_code_options.mas_method
	
	use_padding = my_code_options.use_padding
	if use_padding == True:
		padding_length = my_code_options.padding_length
	else:
		padding_length = 0

	zbinwidth = 0.1

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
		raise TypeError('Wrong format of the catalogs. They should be .hdf5, .dat, .cat, or .txt.')

	if mas_method == "NGP":
		print()
		print("Mass assignement: Nearest Grid Point (NGP)")
		print()
	elif mas_method == "CIC":
		print()
		print("Mass assignement: Clouds in Cell (CiC)")
		print()
	elif mas_method == "TSC":
		print()
		print("Mass assignement: Triangular Shaped Cloud (TSC)")
		print()
	elif mas_method == "PCS":
		print()
		print("Mass assignement: Piecewise Cubic Spline (PCS)")
		print()
	else:
		raise ValueError("Wrong gridding method (mas_method)")

	zmin = np.around(zcentral - zbinwidth,5)
	zmax = np.around(zcentral + zbinwidth,5)

	mean_counts  = np.zeros((ntracers_grid,n_x,n_y,n_z))
	box=(n_x,n_y,n_z)

	# Now, loop over sets of catalogs and tracers
	for nc in range(Ncats):
		print("Processing catalog #", nc)
		counts	= np.zeros((ntracers_grid,n_x,n_y,n_z))
		for nt in range(ntracers_catalog):
			print("Reading catalog for tracer",nt)
			try:
				if use_h5py:
					f = h5py.File(filenames_catalogs[nc,nt], 'r')
					if use_redshifts:
						tracer_redshift , tracer_x, tracer_y, tracer_z = np.array(f['redshift']), np.array(f['x']), np.array(f['y']), np.array(f['z'])
					else:
						tracer_x, tracer_y, tracer_z = np.array(f['x']), np.array(f['y']), np.array(f['z'])
					if split_tracers:
						tracer_type = np.array(f['m'])
					f.close()
					if mask_redshift:
						mask = np.where( (tracer_redshift >= zmin) & (tracer_redshift <= zmax))[0]
						tracer_x = tracer_x[mask]
						tracer_y = tracer_y[mask]
						tracer_z = tracer_z[mask]
						if split_tracers:
							tracer_type = tracer_type[mask]
				else:
					f = filenames_catalogs[nc,nt]
					this_cat = np.loadtxt(f)
					# Fix dimensions if catalog has switched lines/columns
					if this_cat.shape[0] < this_cat.shape[1]:
						this_cat = this_cat.T
					if use_redshifts:
						tracer_redshift, tracer_x, tracer_y, tracer_z = this_cat[:,col_redshift] , this_cat[:,col_x] , this_cat[:,col_y] , this_cat[:,col_z]
					else:
						tracer_x, tracer_y, tracer_z = this_cat[:,col_x] , this_cat[:,col_y] , this_cat[:,col_z]
					# Redshift mask
					if split_tracers:
						tracer_type = this_cat[:,col_tracer]
					if mask_redshift:
						mask = np.where( (tracer_redshift >= zmin) & (tracer_redshift <= zmax))[0]
						tracer_x = tracer_x[mask]
						tracer_y = tracer_y[mask]
						tracer_z = tracer_z[mask]
						if split_tracers:
							tracer_type = tracer_type[mask]
			except:
				raise AttributeError("Could not read file:" , filenames_catalogs[nc,nt])

			ntot = len(tracer_x)
			print("Original catalog has", ntot,"objects")

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

	print("Number of tracers per cell:", np.sum(mean_counts,axis=(1,2,3))/n_x/n_y/n_z)

	if save_mean_sel_fun:
		nf1=len(tot_counts[tot_counts!=0])

		# Smooth to catch non-zero cells
		sig_filter = 0.5
		tg=gaussian_filter(tot_counts,sigma=(sig_filter,sig_filter,sig_filter))

		# Number of cells inside mask
		nfull=len(tg[tg!=0])

		sel_fun = np.zeros((ntracers_grid,n_x,n_y,n_z))
		num_densities = np.zeros(ntracers_grid)
		print("Estimating APPROXIMATE number densities:")
		for i in range(ntracers_grid):
			num_densities[i] = np.sum(mean_counts[i])/nfull/cell_size**3
			print(np.around(num_densities[i],5))
			sel_fun[i] = num_densities[i]*np.sign(tg)

		print("Saving estimated selection function")
		print("(ATTENTION! USE WITH CARE. YOU SHOULD NOT RELY ON THIS ESTIMATE!)")


		h5f = h5py.File(dir_out + "Rough_sel_fun.hdf5",'w')
		h5f.create_dataset('grid', data=sel_fun, dtype='float32', compression='gzip')
		h5f.close()

	if save_mask:
		mask = np.ones((n_x,n_y,n_z))
		mask[:padding_length] = 0
		mask[:,:padding_length] = 0
		mask[:,:,:padding_length] = 0
		mask[-padding_length:] = 0
		mask[:,-padding_length:] = 0
		mask[:,:,-padding_length:] = 0

		h5f = h5py.File(dir_out + "mask.hdf5",'w')
		h5f.create_dataset('grid', data=mask, dtype='float32', compression='gzip')
		h5f.close()
	
	return print("Done!")
