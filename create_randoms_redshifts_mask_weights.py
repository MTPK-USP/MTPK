#! /usr/bin/env python
# -*- coding: utf-8 -*-
########################################
# This is the selection function module
# Use any definition of the selection function in terms of the grid
# The parameters must be defined in the input file
########################################
import numpy as np
import h5py
from scipy import interpolate
import sys

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


###############
# DEfine here OUTPUT file (where you will store angles, redshifts and weights)
filename = "mask_weights_redshifts.hdf5"
###############


###############
# Input file with angular mask and weights of N tracers.
# Format of the file (hdf5, float32): theta, phi, weight_1, weight_2, ..., weight_N
ang_mask_file = 'angular_mask_random_weights.hdf5'

###############
# Input file for radial (redshift) selection function. 
# Format: z , nbar_1 , nbar_2, ..., nbar_N
# Where the redshifts z are assumed to be in EVEN intervals (a cubic splice will be used to interpolate)
radial_sel_file = "radial_selection_function.dat"

print()
print("Reading angular footprint, mask & angular weights...")

# Read angular mask and angular weights
h5map = h5py.File('selection_functions/' + ang_mask_file, 'r')
h5data = h5map.get(list(h5map.keys())[0])
theta_phi_weights = np.asarray(h5data,dtype='float32')
h5map.close

# Length of mask random
Ntot = len(theta_phi_weights)

# Read radial (redshift) selection function
z_sel = np.loadtxt("selection_functions/" + radial_sel_file ,comments="#")


# Define a redshift interval smaller than the redshift error for your survey
dz=0.001
zrange=np.arange(np.min(z_sel[:,0])+dz/2,np.max(z_sel[:,0]),dz)

Ntracers=len(z_sel.T)-1
sel_fun_z = np.zeros((len(zrange),Ntracers))

# Create an interpolation function and a cubic spline for the radial selection function
for i in range(Ntracers):
	zs_interp = interpolate.interp1d(z_sel[:,0], z_sel[:,i+1])
	new_z = zs_interp(zrange)
	if new_z.any() < 0:
		print("WARNING: selecting function for tracer", i, "returned negative value.")
		print("Your selection function for this tracer may be poorly sampled. Using absolute value...")
	sel_fun_z[:,i] = np.abs(new_z)



sf_int = np.sum(sel_fun_z,axis=0)
sel_fun_norm = np.int0(np.round(Ntot*sel_fun_z/sf_int))

NewNtot = np.sum(sel_fun_norm,axis=0)
sel_fun_norm = np.int0(np.round(Ntot**2/NewNtot*sel_fun_z/sf_int))

NewNtot = np.sum(sel_fun_norm,axis=0)


radsel = np.zeros((Ntot,Ntracers))
# Now create a sequence of redshifts that is a concatenation of arrays of length sel_fun_norm
print("Building redshift selection functions...")


for nt in range(Ntracers):
	print("... tracer #",nt)
	seq=np.ones(NewNtot[nt])
	Ninit=0
	for i in range(len(zrange)):
		Nend = Ninit + sel_fun_norm[i,nt] 
		seq[Ninit:Nend] = zrange[i]
		Ninit = Nend
	# Add or subtract redshifts at the start of the redshift range to match number of objects in random
	if NewNtot[nt] > Ntot:
		dN = NewNtot[nt] - Ntot
		seq=seq[dN:NewNtot[nt]]
	elif NewNtot[nt] < Ntot:
		dN = Ntot - NewNtot[nt]
		seq=np.concatenate((zrange[0]*np.ones(dN),seq))
	# Now randomize the order so we don't end up with galaxies ordered in the same way in the same places'
	np.random.shuffle(seq)
	radsel[:,nt] = seq

print()
print("Plotting redshift selection functions: in the original table, and in the randoms")
zbins_edges=zrange[::10]
zbins_cetrs=0.5*(zbins_edges[1:] + zbins_edges[:-1])
for nt in range(Ntracers):
	pl.plot(z_sel[:,0],z_sel[:,nt+1])
	zhist = np.histogram(radsel[:,nt],bins=zbins_edges)[0]
	norm = z_sel[0,nt+1]/np.mean(zhist[:5])
	zhist = norm*zhist
	pl.plot(zbins_cetrs,zhist,'k--')
pl.savefig("selection_functions/redshift_sel_fun.png")
pl.close()


print()
# Now combine theta, phi, z, weights
print("Now saving angular mask + weights + redshifts ... (be patient... may take a while...)")


h5f = h5py.File("selection_functions/"+filename,'w')
h5f.create_dataset('randoms', data=np.hstack([ theta_phi_weights , radsel]), dtype='float32',compression='gzip')
h5f.close()

print()
print("Done!")



