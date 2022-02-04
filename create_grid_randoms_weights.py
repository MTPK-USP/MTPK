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
#  - file "file_mask_weights_redshifts_randoms", in hdf5, which is a 
#    a table with Nran (number of "points" in random) lines, and columns
#    which are: 
#    0-theta  1-phi  
#    2-weight(1)  3-weight(2) ... 1+Nt-weight(Nt)
#    2+Nt-z(1)  3+Nt-z(2) ... 2+2Nt-z(Nt)
#
#  - standard input file containing grid properties, cosmological model, etc.
#
########################################
import numpy as np
import h5py
import sys
import os
from cosmo_funcs import *
from scipy.ndimage import gaussian_filter

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
# File with the randoms
# Format of the file (hdf5, float32): theta, phi, weight_1, weight_2, ..., weight_N, z_1, z_2, ..., z_N
file_mask_weights_redshifts_randoms = 'mask_weights_redshifts.hdf5'


####################
# Input file with grid properties and cosmological model
input_filename = "Highz_z24"


####################
# Apply smoothing kernel? If yes, what is FWHM of kernel, in cells?
smooth = True
smoothing_length = 1.0


####################
# Bins of weight -- make sure this is a fine grid that represents the values of the weights
wbins = np.arange(0.94,1.1,0.01)


####################
# Output files for grids containing counts and weights
output_filename_counts  = "sel_fun_highz_z2.4.hdf5"
output_filename_weights = "weights_highz_z2.4.hdf5"
####################




####################################################################################################
# Start main code
#
#

print()
print("++++++++++++++++++++")
print()
print("This is the Grid Selection Function code")
print()
print(" --> Will load randoms (theta, phi, weights, redshifts) from file:", file_mask_weights_redshifts_randoms)
print(" --> Will load input parameters from from file:", input_filename + ".py")
print(" --> Will export grids of counts to file:", output_filename_counts)
print(" --> Will export grids of weights to file:", output_filename_weights)
print()

# Read input file
this_dir = os.getcwd()
input_dir = this_dir + '/inputs'
sys.path.append(input_dir)

try:
	exec ("from " + input_filename + " import *")
except:
	print("Failed importing inputs. Check file:", input_filename)
	sys.exit(-1)

print("Starting now: reading random catalog... (may take a while if the file is very large...)")
print()

# Read randoms
try:
	h5map = h5py.File('selection_functions/' + file_mask_weights_redshifts_randoms, 'r')
	h5data = h5map.get(list(h5map.keys())[0])
	randoms = np.asarray(h5data,dtype='float32')
	h5map.close
except:
	print("Could not read file with the randoms," , file_mask_weights_redshifts_randoms , ". Please check the file name in the code")

# Find how many tracers
Nt = (len(randoms.T) - 2)//2
Nran = len(randoms)
print("Using ", Nran, " random points, for Nt=", Nt, " tracers")




# Determine boundaries of box in (x,y,z)
# By definition, the box STARTS in the Cartesian z direction,
# at the lowest value of z(cart) 
# By definition, the map is already centralized around the z axis, and is 

zred_min = zcentral - zbinwidth
zred_max = zcentral + zbinwidth

zred_min_cat = np.min(randoms[:,2+Nt:])
zred_max_cat = np.max(randoms[:,2+Nt:])

if ( (zred_min < zred_min_cat) | (zred_max > zred_max_cat) ):
	print("Your random catalog has redshift intervals that do not match your redshift slice:")
	print("       Randoms: min(z)=",np.around(zred_min_cat,2),"; max(z)=",np.around(zred_max_cat,2))
	print("    This slice: min(z)=",np.around(zred_min,2),"; max(z)=",np.around(zred_max,2))
	yesno = input(" Continue anyway? Y or N :")
	if yesno != "Y":
		print('Aborting now.')
		sys.exit(-1)
	print()


Omegam = Omegab + Omegac

chi_min = chi_h(Omegam, 1-Omegam-Omegak, w0, w1, zred_min)
chi_max = chi_h(Omegam, 1-Omegam-Omegak, w0, w1, zred_max)
chi_mean = chi_h(Omegam, 1-Omegam-Omegak, w0, w1, zcentral)

xhat=np.sin(randoms[:,0])*np.cos(randoms[:,1])
yhat=np.sin(randoms[:,0])*np.sin(randoms[:,1])
zhat=np.cos(randoms[:,0])

zcart_min = chi_min*np.min(zhat)
zcart_max = chi_max*np.max(zhat)
zcart_mean = chi_mean*np.mean(zhat)

xcart_min = chi_max * np.min(xhat)
xcart_max = chi_max * np.max(xhat)
xcart_mean = chi_mean * np.mean(xhat)

ycart_min = chi_max * np.min(yhat)
ycart_max = chi_max * np.max(yhat)
ycart_mean = chi_max * np.mean(yhat)

n_z_orig = zcart_mean/cell_size

xydata=np.array([xhat[::200],yhat[::200]]).T
x1, y1 = chi_max*xydata[:,0], chi_max*xydata[:,1]
x2, y2 = chi_min*xydata[:,0], chi_min*xydata[:,1]

pl.scatter(x1,y1,s=0.5,marker='o',c='r',alpha=0.5)
pl.scatter(x2,y2,s=0.5,marker='o',c='b',alpha=0.5)
pl.savefig("xy.png")
pl.close()

# THe displacement of the origin (0,0,0) of the box 
# with respect to the origin of the coordinate system (Earth) is defined in the input file
# (in cell units; doesn't need to be integer).

n_x_orig , n_y_orig , n_z_orig = xcart_min/cell_size , ycart_min/cell_size , zcart_min/cell_size


print()
print("Using the redshift range z_min, z_max:", zred_min, zred_max)



# Now create grids: on each cell we compute (1) the number of each tracer in each shell;
# (2) the mean weights of those tracers in that shell

# Grid binning
xbins = np.arange(xcart_min,xcart_max+cell_size,cell_size)
ybins = np.arange(ycart_min,ycart_max+cell_size,cell_size)
zbins = np.arange(zcart_min,zcart_max+cell_size,cell_size)

n_x = len(xbins) - 1
n_y = len(ybins) - 1
n_z = len(zbins) - 1

print("Dimensions of the grid: n_x, n_y, n_z =",n_x,n_y,n_z)
print()
print("Origin (0,0,0) of box displaced from the observer @Earth by this # of cells:")
print("n_x_orig=" , n_x_orig)
print("n_y_orig=" , n_y_orig)
print("n_z_orig=" , n_z_orig)
print()
print(" !!! Make sure that these are the values in the input file:", input_filename, "!!!")
yesno = input(" Continue? Y or N :")
if yesno != "Y":
	print('Aborting now.')
	sys.exit(-1)
print()

counts  = np.zeros((Nt,n_x,n_y,n_z))
weights = np.zeros((Nt,n_x,n_y,n_z))


# Weight bin centers
wcent = 0.5*(wbins[1:] + wbins[:-1])

print()
print("Using weights binned in the form:")
print(wbins)

print()
print("Now computing the grids for the tracers: one grid for the counts, one for the weights")
print()
for nt in range(Nt):
	print("Tracer" , nt, ". Selecting redshifts...")
	# Select the objects in the randoms that fall into this redshift
	redshift_mask = np.where( (randoms[:,2+Nt+nt] >= zred_min) & (randoms[:,2+Nt+nt] <= zred_max) )[0]
	if len(redshift_mask) < 1:
		print("Warning! There are no tracers in this redshift range. Try again. Aborting now...")
		sys.exit(-1)
	tracer_angles  = randoms[redshift_mask,:2]
	tracer_weights = randoms[redshift_mask,2+nt]
	tracer_redshifts  = randoms[redshift_mask,2+Nt+nt]
	del redshift_mask
	print("... converting redshifts to radii (xi), using chosen fiducial cosmological model...")
	tracer_chi = chi_h(Omegam, 1-Omegam-Omegak, w0, w1, tracer_redshifts)
	del tracer_redshifts
	print("...computing histograms inside the grid box...")
	tracer_x = tracer_chi*np.sin(tracer_angles[:,0])*np.cos(tracer_angles[:,1])
	tracer_y = tracer_chi*np.sin(tracer_angles[:,0])*np.sin(tracer_angles[:,1])
	tracer_z = tracer_chi*np.cos(tracer_angles[:,0])
	del tracer_chi, tracer_angles
	histo = np.histogramdd(np.array([tracer_x,tracer_y,tracer_z,tracer_weights]).T,bins=(xbins,ybins,zbins,wbins))[0]
	print("Min, max in Cartesian x, y, z for this tracer:")
	print(np.min(tracer_x),np.max(tracer_x))
	print(np.min(tracer_y),np.max(tracer_y))
	print(np.min(tracer_z),np.max(tracer_z))
	del tracer_x, tracer_y, tracer_z
	counts[nt]  = np.sum(histo,axis=3)
	weights[nt] = 1./(0.00001 + counts[nt]) * np.sum(histo * wcent,axis=3)
	# Smooth the counts & weights with given kernel
	if smooth == True:
		counts[nt]  = gaussian_filter(counts[nt],sigma=(smoothing_length,smoothing_length,smoothing_length))
		weights[nt] = gaussian_filter(weights[nt],sigma=(smoothing_length,smoothing_length,smoothing_length))
	# Normalize selection function to actual number of objects in dataset
	ntot = np.sum(counts[nt])
	norm = 1.0*Ntot_Gals[nt]/ntot
	counts[nt] = norm * counts[nt]
	del histo
	print("... done.")
	print()

# Finally, normalize the number of objects to the total number of actual galaxies in the map

# Save grids
print("Saving grids to output files...")

h5f = h5py.File("selection_functions/" + output_filename_counts,'w')
h5f.create_dataset('grid', data=counts, dtype='float32',compression='gzip')
h5f.close()

h5f = h5py.File("selection_functions/" + output_filename_weights,'w')
h5f.create_dataset('grid', data=weights, dtype='float32',compression='gzip')
h5f.close()

vol_box = n_x*n_y*n_z*cell_size**3*1.e-9

print()
print("Volume occupied by tracers:")
for nt in range(Nt):
	vol = len(counts[nt][counts[nt]!=0])*cell_size**3*1.e-9
	print("Tracer", nt, ", Volume (h^3 Gpc^-3)=", np.around(vol,4) , " (", np.around(vol/vol_box,3) , "of the full box)")

print()
print("Plotting slices (in Cartesian z) of the selection function and weights to /figures... ")
for nt in range(Nt):
	for i in range(20):
		zslice = np.int0((i+0.5)*(n_z/20.))
		pl.imshow(counts[nt,:,:,zslice])
		this_name = input_filename + "_tracer_" + str(nt) + "_slice_" + str(zslice) + ".png" 
		pl.savefig("selection_functions/figures/" + this_name,dpi=200)
		pl.close()

print()
print("...done!")
print()
print("++++++++++++++++++++")
print()
