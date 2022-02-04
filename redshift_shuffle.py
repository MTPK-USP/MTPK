#! /usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import h5py
from scipy import interpolate
import sys
import glob
import os

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


###########################################################################
#
# User definitions. All files by default stored in directory /selection_functions
#

# Directory and files containing the catalogs
dir_in = "catalogs/Highz_z16_gals/"

# Use glob and read all catalogs in that directory containing handle:
handles = ["Tracer_0","Tracer_1"]

# Specify in which column you have the redshifts in the original catalogs
z_col = 2

# Directory where we store the 
dir_out = "catalogs/SHUFFLE_Highz_z16_gals/"

# min, max redshift of the catalog
zmin = 1.4
zmax = 1.8

###############
# Input file with photo-zs. 
# Format: zspec , zphot

zspecphot_filenames = ["inputs/zspec_zphot_sigma0.003.dat","inputs/Quasar_photoz_jpas_mock.cat"]

# Or, if using a different format...
zspec_col = 0
zphot_col = 1

# You may also change the effect of the photo-zs by: dz --> f * dz
dz_factor = 0.4


###########################################################################

this_dir = os.getcwd()

dir_in = this_dir + "/" + dir_in 
dir_out = this_dir + "/" + dir_out

if not os.path.exists(dir_out):
    os.makedirs(dir_out)


for nhand in range(len(handles)):

    print()
    print("Reading files with spectroscopic and photometric redshifts for handle/tracer:")
    print("  > ", handles[nhand])
    print()

    zspecphot_filename = this_dir + "/" + zspecphot_filenames[nhand]


    try:
        zsp = np.loadtxt(zspecphot_filename ,comments="#")
    except:
        print("Could not find file with photo-zs! Aborting now...")
        sys.exit(-1)

    nz = len(zsp)

    zspec=zsp[:,zspec_col]
    zphot=zsp[:,zphot_col]

    # We cannot have objects in the table at exactly the same redshifts, or the interpolation fails
    # So, we sum a very small number to the zspec file
    zspec = zspec + np.random.uniform(-0.00001,0.00001,len(zspec))

    # If z_spec not ordered, order them now
    ord_zs = np.argsort(zspec)

    zspec=zspec[ord_zs]
    zphot=zphot[ord_zs]

    # Redshift error
    dz = dz_factor*(zspec - zphot)
    snmad = 1.48*np.median(np.abs(dz)/(1+zspec))
    print("Photo-zs have sigma_NMAD",np.around(snmad,5))
    print()

    # Create function that interpolates position in zspec array given zspec
    zindx = np.arange(len(zspec))
    zindx_interp = interpolate.interp1d(zspec, zindx)


    print("Now attempting to find catalogs...")

    filenames_in = glob.glob(dir_in + '*' + handles[nhand] + '*')
    filenames_in = np.sort(np.array(filenames_in))

    ncats = len(filenames_in)

    if ncats ==0:
        print("Could not find any catalogs in directory", dir_in, "with handle", handles[nhand])
        print("Aborting now...")
        sys.exit(-1)

    print("Reading catalog with positions (angles x 2, redshifts), weights, etc. ...")


    for nc in range(ncats):
        # Read random/data catalog
        print("Processing catalog #",nc,":")
        print("  > ", filenames_in[nc])
        h5map = h5py.File(filenames_in[nc], 'r')
        h5data = h5map.get(list(h5map.keys())[0])
        cat = np.asarray(h5data,dtype='float32')
        h5map.close
        z_obj = cat[:,z_col]

        # Length of mask random
        Ntot = len(z_obj)

        print("   ...this catalog has ", Ntot, "objects.")

        if ( (np.max(z_obj) > np.max(zspec)) | (np.min(z_obj) < np.min(zspec)) ):
        	print("   Sorry, the objects in your catalog fall outside the range of spectroscopic redshifts given in your table.")
        	print("   Min(z), max(z) of catalog:", np.min(z_obj), np.max(z_obj) ) 
        	print("   Min(z), max(z) of z_spec table:", np.min(zspec), np.max(zspec) ) 

        # Given the z of an object, z_obj, the nearest zspec, z_near, will be at
        # position_nearest = interpolate.splev( z_obj , zindx_interp , der=0)
        indx_obj = np.int0( np.around( zindx_interp(z_obj) ) )

        # Create an array of random shifts from that "nearest" position
        random_shifts = np.int0(np.random.uniform(-5,5,Ntot))

        # Shift original indices, and make sure they all fall within range
        new_indx = indx_obj + random_shifts
        new_indx[new_indx<0] = 0
        new_indx[new_indx>zindx[-1]] = zindx[-1]

        # Add redshift fluctuation to objects
        z_obj_new = z_obj + dz[new_indx]

        # Redefine the redshifts
        newcat = np.copy(cat)
        newcat[:,z_col] = z_obj_new

        # Apply mask
        #redshift_mask = np.where( (newcat[:,z_col] >= zmin) & (newcat[:,z_col] <= zmax) )[0]
        #newcat = newcat[redshift_mask]

        print("   After redshift shuffle there are", len(cat),"objects remaining inside the catalog mask.")

        if len(str(nc))==1:
            map_num = '00' + str(nc)
        elif len(str(nc))==2:
            map_num = '0' + str(nc)
        else:
            map_num = str(nc)

        snmad_thismap = 1.48*np.median(np.abs(z_obj_new - z_obj)/(1+z_obj))
        print("   sigma_NMAD for this catalog and tracer:",np.around(snmad_thismap,5))
        print("   Saving shuffled catalog with file name:", dir_out + "SHUFFLED_Map_" + map_num + "_" + handles[nhand] + ".hdf5")

        h5f = h5py.File(dir_out + "SHUFFLED_Map_" + map_num + "_" + handles[nhand] + ".hdf5",'w')
        h5f.create_dataset('randoms', data=newcat , dtype='float32',compression='gzip')
        h5f.close()
        print()

print()
print("Done!")



