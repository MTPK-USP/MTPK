'''
This code creates a grid (nx , ny, nz) of cells of size cell_size, 
containint the selection functions of any number of tracers (Nt).

The output has the format (Nt,nx,ny,nz)

INPUTS:

 - files "catalog_XXX_YY.dat", ..., for XXX representing the seed and YY the bin, which are 
   table with any number of "points" with columns given by the cartesian coordinates:
   (x) (y) (z)
'''

#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import h5py
import sys
import os
from cosmo_funcs import *
import glob
from scipy.ndimage import gaussian_filter
from cosmo import cosmo #definining the cosmological parameters
from code_options import code_parameters #defining the code options

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

def cic(value, x, nx, y = None, ny=1, z = None, nz=1, wraparound = False, average = True):
	""" 

        Clouds-in-Cells

        Interpolate an irregularly sampled field using Cloud in Cell
	method.
	This function interpolates an irregularly sampled field to a
	regular grid using Cloud In Cell (nearest grid point gets weight
	1-dngp, point on other side gets weight dngp, where dngp is the
	distance to the nearest grid point in units of the cell size).
	
	Inputs
	------
	value: array, shape (N,)
		Sample weights (field values). For a temperature field this
		would be the temperature and the keyword average should be
		True. For a density field this could be either the particle
		mass (average should be False) or the density (average should
		be True).
	x: array, shape (N,)
		X coordinates of field samples, unit indices: [0,NX>.
	nx: int
		Number of grid points in X-direction.
	y: array, shape (N,), optional
		Y coordinates of field samples, unit indices: [0,NY>.
	ny: int, optional
		Number of grid points in Y-direction.
	z: array, shape (N,), optional
		Z coordinates of field samples, unit indices: [0,NZ>.
	nz: int, optional
		Number of grid points in Z-direction.
	wraparound: bool (False)
		If True, then values past the first or last grid point can
		wrap around and contribute to the grid point on the opposite
		side (see the Notes section below).
	average: bool (False)
		If True, average the contributions of each value to a grid
		point instead of summing them.
	Returns
	-------
	dens: ndarray, shape (nx, ny, nz)
		The grid point values.
	Notes
	-----
	Example of default allocation of nearest grid points: nx = 4, * = gridpoint.
	  0   1   2   3     Index of gridpoints
	  *   *   *   *     Grid points
	|---|---|---|---|   Range allocated to gridpoints ([0.0,1.0> -> 0, etc.)
	0   1   2   3   4   posx
	Example of ngp allocation for wraparound=True: nx = 4, * = gridpoint.
	  0   1   2   3        Index of gridpoints
	  *   *   *   *        Grid points
	|---|---|---|---|--    Range allocated to gridpoints ([0.5,1.5> -> 1, etc.)
	  0   1   2   3   4=0  posx
	References
	----------
	R.W. Hockney and J.W. Eastwood, Computer Simulations Using Particles
		(New York: McGraw-Hill, 1981).
	Modification History
	--------------------
	IDL code written by Joop Schaye, Feb 1999.
	Avoid integer overflow for large dimensions P.Riley/W.Landsman Dec. 1999
	Translated to Python by Neil Crighton, July 2009.
	
	Examples
	--------
	>>> nx = 20
	>>> ny = 10
	>>> posx = np.random.rand(size=1000)
	>>> posy = np.random.rand(size=1000)
	>>> value = posx**2 + posy**2
	>>> field = cic(value, posx*nx, nx, posy*ny, ny)
	# plot surface
	"""

	def findweights(pos, ngrid):
		""" Calculate CIC weights.
		
		Coordinates of nearest grid point (ngp) to each value. """

		if wraparound:
			# grid points at integer values
			ngp = np.fix(pos + 0.5)
		else:
			# grid points are at half-integer values, starting at 0.5,
			# ending at len(grid) - 0.5
			ngp = np.fix(pos) + 0.5

		# Distance from sample to ngp.
		distngp = ngp - pos

		# weight for higher (right, w2) and lower (left, w1) ngp
		weight2 = np.abs(distngp)
		weight1 = 1.0 - weight2

		# indices of the nearest grid points
		if wraparound:
			ind1 = ngp
		else:
			ind1 = ngp - 0.5
		ind1 = ind1.astype(int)

		ind2 = ind1 - 1
		# Correct points where ngp < pos (ngp to the left).
		ind2[distngp < 0] += 2

		# Note that ind2 can be both -1 and ngrid at this point,
		# regardless of wraparound. This is because distngp can be
		# exactly zero.
		bad = (ind2 == -1)
		ind2[bad] = ngrid - 1
		if not wraparound:
			weight2[bad] = 0.
		bad = (ind2 == ngrid)
		ind2[bad] = 0
		if not wraparound:
			weight2[bad] = 0.

		if wraparound:
			ind1[ind1 == ngrid] = 0

		return dict(weight=weight1, ind=ind1), dict(weight=weight2, ind=ind2)


	def update_field_vals(field, totalweight, a, b, c, value, debug=True):
		""" This updates the field array (and the totweight array if
		average is True).
		The elements to update and their values are inferred from
		a,b,c and value.
		"""
		# indices for field - doesn't include all combinations
		indices = a['ind'] + b['ind'] * nx + c['ind'] * nxny
		# weight per coordinate
		weights = a['weight'] * b['weight'] * c['weight']
		# Don't modify the input value array, just rebind the name.
		value = weights * value 
		if average:
			for i,ind in enumerate(indices):
				field[ind] += value[i]
				totalweight[ind] += weights[i]
		else:
			for i,ind in enumerate(indices):
				field[ind] += value[i]

	nx, ny, nz = (int(i) for i in (nx, ny, nz))
	nxny = nx * ny
	value = np.asarray(value)

	x1, x2 = findweights(np.asarray(x), nx)
	y1 = z1 = dict(weight=1., ind=0)
	if y is not None:
		y1, y2 = findweights(np.asarray(y), ny)
		if z is not None:
			z1, z2 = findweights(np.asarray(z), nz)

	# float32 to save memory for big arrays (e.g. 256**3)
	field = np.zeros(nx * ny * nz, np.float32)

	if average:
		totalweight = np.zeros(nx * ny * nz, np.float32)
	else:
		totalweight = None

	update_field_vals(field, totalweight, x1, y1, z1, value)
	update_field_vals(field, totalweight, x2, y1, z1, value)
	if y is not None:
		update_field_vals(field, totalweight, x1, y2, z1, value)
		update_field_vals(field, totalweight, x2, y2, z1, value)
		if z is not None:
			update_field_vals(field, totalweight, x1, y1, z2, value)
			update_field_vals(field, totalweight, x2, y1, z2, value)
			update_field_vals(field, totalweight, x1, y2, z2, value)
			update_field_vals(field, totalweight, x2, y2, z2, value)

	if average:
		good = totalweight > 0
		field[good] /= totalweight[good]
	return field.reshape((nz, ny, nx)).transpose()

def create_grids_from_xyz_cats(my_cosmology, my_code_options, input_filename, filenames_catalogs):

        '''
        Create grids from xyz cats
        '''
        
        filenames_catalogs = np.sort(glob.glob(filenames_catalogs))
        
        ####################
        # Files containing the catalogs

        # Or, use glob and read all catalogs in some directory 

        this_dir = os.getcwd()

        if len(filenames_catalogs) == 0:
                raise NameError('Files not to create found!')
                
        Ncats = my_code_options.n_maps
        Ntracers = my_code_options.nhalos
        use_redshifts = my_code_options.use_redshifts
        col_x = my_code_options.col_x
        col_y = my_code_options.col_y
        col_z = my_code_options.col_z
        x_cat_min = my_code_options.x_cat_min
        y_cat_min = my_code_options.y_cat_min
        z_cat_min = my_code_options.z_cat_min
        x_cat_max = my_code_options.x_cat_max
        y_cat_max = my_code_options.y_cat_max
        z_cat_max = my_code_options.z_cat_max
        mask_redshift = my_code_options.mask_redshift
        save_mask = my_code_options.save_mask
        save_mean_sel_fun = my_code_options.save_mean_sel_fun

        # This line should be rewritten if the order of the files doesn't match the required
        # ordering: first index is catalog/sub-box, second index is tracer
        filenames_catalogs = np.reshape(filenames_catalogs, (Ncats, Ntracers))

        # Which directory to write grids to:
        dir_out = this_dir + "/maps/sims/" + input_filename + "/"
        filenames_out = "Data"

        # File to save mean total number of tracers in maps
        file_ntot = this_dir + "/inputs/Ntot_halos.txt"
        file_MF = this_dir + "/inputs/"+input_filename+"_MF.dat"

        print()
        print("Will load maps stored in files:")
        print(filenames_catalogs)

        if not os.path.exists(dir_out):
	        os.makedirs(dir_out)

        # Read input file
        this_dir = os.getcwd()
        input_dir = this_dir + '/inputs'
        sys.path.append(input_dir)

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

        grid_method = my_code_options.mas_method
        if grid_method == 'NGP':
                grid_method = 1
        elif grid_method == 'CIC':
                grid_method = 2
        
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

        if grid_method == 1:
	        print()
	        print("Mass assignement: Nearest Grid Point (NGP)...")
	        print()
        elif grid_method ==2:
	        print()
	        print("Mass assignement: Clouds in Cell (CiC)...")
	        print()
        else:
                raise ValueError("Wrong gridding method (mas_method)")

        zmin = np.around(zcentral - zbinwidth,5)
        zmax = np.around(zcentral + zbinwidth,5)

        mean_counts  = np.zeros((Ntracers,n_x,n_y,n_z))

        # Now, loop over sets of catalogs and tracers
        for nc in range(Ncats):
	        print("Processing catalog #", nc)
	        counts  = np.zeros((Ntracers,n_x,n_y,n_z))
	        for nt in range(Ntracers):
		        print("Reading catalog for tracer",nt)
		        try:
			        if use_h5py:
				        f = h5py.File(filenames_catalogs[nc,nt], 'r')
				        if use_redshifts:
					        tracer_redshift , tracer_x, tracer_y, tracer_z = np.array(f['redshift']), np.array(f['x']), np.array(f['y']), np.array(f['z'])
				        else:
					        tracer_x, tracer_y, tracer_z = np.array(f['x']), np.array(f['y']), np.array(f['z'])
				        f.close()
				        if mask_redshift:
					        mask = np.where( (tracer_redshift >= zmin) & (tracer_redshift <= zmax))[0]
					        tracer_x = tracer_x[mask]
					        tracer_y = tracer_y[mask]
					        tracer_z = tracer_z[mask]
			        else:
				        f = filenames_catalogs[nc,nt]
				        this_cat = np.loadtxt(f)
				        if use_redshifts:
					        tracer_redshift, tracer_x, tracer_y, tracer_z = this_cat[:,col_redshift] , this_cat[:,col_x] , this_cat[:,col_y] , this_cat[:,col_z]
				        else:
					        tracer_x, tracer_y, tracer_z = this_cat[:,col_x] , this_cat[:,col_y] , this_cat[:,col_z]
				        # Redshift mask
				        if mask_redshift:
					        mask = np.where( (tracer_redshift >= zmin) & (tracer_redshift <= zmax))[0]
					        tracer_x = tracer_x[mask]
					        tracer_y = tracer_y[mask]
					        tracer_z = tracer_z[mask]
		        except:
                                raise AttributeError("Could not read file:" , filenames_catalogs[nc,nt])

		        #ntot = len(cat)
		        ntot = len(tracer_x)
		        print("Original catalog has", ntot,"objects")

		        tracer_x = tracer_x - x_cat_min + cell_size*padding_length
		        tracer_y = tracer_y - y_cat_min + cell_size*padding_length
		        tracer_z = tracer_z - z_cat_min + cell_size*padding_length

		        # Mask out cells outside bounds of the box
		        posx = np.where( (tracer_x <= 0) | (tracer_x >= xbins[-1]) )[0]
		        posy = np.where( (tracer_y <= 0) | (tracer_y >= ybins[-1]) )[0]
		        posz = np.where( (tracer_z <= 0) | (tracer_z >= zbins[-1]) )[0]

		        boxmask = np.union1d(posx,np.union1d(posy,posz))

		        tracer_x=np.delete(tracer_x,boxmask)
		        tracer_y=np.delete(tracer_y,boxmask)
		        tracer_z=np.delete(tracer_z,boxmask)

		        ntot = len(tracer_x)
		        print("After trimming inside box, the catalog has", ntot,"objects")
		        print()
		        print(" Now building grid box...")

		        if grid_method == 1:
			        counts[nt] = np.histogramdd(np.array([tracer_x,tracer_y,tracer_z]).T,bins=(xbins,ybins,zbins))[0]
			        mean_counts[nt] += counts[nt]
		        elif grid_method ==2:
			        val = np.ones(ntot)
			        cic_grid = cic(val,tracer_x/cell_size,n_x,tracer_y/cell_size,n_y,tracer_z/cell_size,n_z,average=False)
			        counts[nt] = cic_grid
			        mean_counts[nt] += cic_grid
		        else:
                                raise ValueError("Wrong gridding method (mas_method)")

		        del tracer_x, tracer_y, tracer_z

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
	        if grid_method == 1:
		        h5f.create_dataset('grid', data=counts, dtype='int32',compression='gzip')
	        else:
		        h5f.create_dataset('grid', data=counts, dtype='float32', compression='gzip')
	        h5f.close()

	        print()

        mean_counts = mean_counts/Ncats
        tot_counts = np.sum(mean_counts,axis=0)

        # Mean total number of halos of each type in box
        np.savetxt(file_ntot, np.sum(mean_counts, axis=(1,2,3) ) )
        # Mass Function in box
        np.savetxt(file_MF, np.sum(mean_counts, axis=(1,2,3) )/V )

        if save_mean_sel_fun:
	        nf1=len(tot_counts[tot_counts!=0])

	        # Smooth to catch non-zero cells
	        sig_filter = 0.5
	        tg=gaussian_filter(tot_counts,sigma=(sig_filter,sig_filter,sig_filter))

	        # Number of cells inside mask
	        nfull=len(tg[tg!=0])

	        sel_fun = np.zeros((Ntracers,n_x,n_y,n_z))
	        num_densities = np.zeros(Ntracers)
	        print("Estimating APPROXIMATE number densities:")
	        for i in range(Ntracers):
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

        print()
        print("...done!")
        print()
        print("++++++++++++++++++++")
        print()
        
        return 0
