#! /usr/bin/env python
# -*- coding: utf-8 -*-
########################################
# This is the selection function module
# Use any definition of the selection function in terms of the grid
# The parameters must be defined in the input file
########################################
import numpy as np
import h5py
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


###################
# Define how many points you want in your random catalogs (same will be used for all tracers)
# Watch out! Careful not to run out of memory...
# Total number of randoms in full footprint (actual number will be lower)
N_rand = np.int0(3 * 10**8)
# Number of randoms in each batch - depends on your RAM and CPU
N_rand_batch = np.int0(3 * 10**7)

###################
# Save center of footprint to this file:
file_mean_RA_dec = "mean_RA_dec.dat" 

###################
# Name of file where randoms will be saved:
file_angular_mask_random_weights = "angular_mask_random_weights.hdf5"
###################


# Load boundaries and exclusion zones of the mask
from parameters_mask_footprint import *
from parameters_mask_exclusion_zones import *

# Load selection function & weight parameters
from parameters_angular_weights import *


# Notice:
# RA, dec : 0 <= RA <= 360 ; -90 <= dec <= 90
# RA, dec : 0 <= RA <= 2*pi ; -pi/2 <= dec <= pi/2
# theta = pi/2 - dec 
# phi   = RA


# Boolean function to determine if a set of theta , phi (numpy arrays) are NOT masked (i.e., they ARE in the map)
def mask_bound(abc,theta,phi):
	dec = np.pi/2.  - theta
	ra  = phi
	d2 = np.outer(dec,np.ones_like(ra))
	r2 = np.outer(np.ones_like(dec),ra)
	cond_foot = (r2**2 >= 0.0)
	edges = len(abc.T)
	for i in range(edges):
		cond_foot = cond_foot * (abc[0,i]*r2 + abc[1,i]*d2 - abc[2,i] > 0)
	return cond_foot

def mask_exczones(trds,theta,phi):
	dec = np.pi/2.  - theta
	ra  = phi
	d2 = np.outer(dec,np.ones_like(ra))
	r2 = np.outer(np.ones_like(dec),ra)
	cond_exc = (r2**2 >= 0.0)
	exczones = len(trds.T)
	for i in range(exczones):
		if trds[0,i]==0:
			cond_exc = cond_exc * ( (r2-trds[1,i])**2 + (d2-trds[2,i])**2 - (trds[3,i])**2 >= 0 )
		if trds[0,i]==1:
			cond_exc = cond_exc * np.logical_not( ((r2-trds[1,i])**2 - (trds[3,i]/2.)**2 < 0) & ((d2-trds[2,i])**2 - (trds[3,i]/2.)**2 < 0) )
	return cond_exc

# Mask boundaries (footprint) and exclusion zones
abc = np.asarray([a_edge,b_edge,c_edge])
trds = np.asarray([exclusion_zones_types,exclusion_zones_RA,exclusion_zones_dec,exclusion_zones_sizes])

# Define *FULL* grid of angles to focus on main footprint
dmu=0.001
dphi=np.pi*0.001

mugrid    = np.arange(-1.0+dmu/2.0, 1.0,dmu)
thetagrid = np.arccos(mugrid)
phigrid   = np.arange(0, 2.*np.pi + dphi/4., dphi)

mask = mask_exczones(trds,thetagrid,phigrid)*mask_bound(abc,thetagrid,phigrid)
mask1 = 1*mask

# Plot mask area
pl.imshow(mask1)
pl.savefig("selection_functions/mask.png",dpi=300)
pl.close()

# Plot mask in polar coordinates
tflat=np.outer(thetagrid,np.ones_like(phigrid)).flatten()
fflat=np.outer(np.ones_like(thetagrid),phigrid).flatten()
mflat=mask.flatten()

tflat = tflat[mflat]
fflat = fflat[mflat]

fig = pl.figure()
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(fflat, tflat, c='b', s=0.01, alpha=0.75)
pl.savefig("selection_functions/mask_polar_orig.png",dpi=300)
pl.close()

# Compute area of mask, and centers of mask
area = dmu * dphi * np.sum(mask1)
areadeg = 41253./4/np.pi
print("Area inside mask (deg^2):",areadeg)

# Compute the mean mu and mean theta:
mean_mu = (dmu*dphi/area) * np.sum(np.outer(mugrid,np.ones_like(phigrid))*mask1)
mean_theta = np.arccos(mean_mu)

# Compute mean phi
mean_phi = (dmu*dphi/area) * np.sum(np.outer(np.ones_like(mugrid),phigrid)*mask1)

mean_dec = 90. - 180.*mean_theta/np.pi
mean_RA = 180.*mean_phi/np.pi

print("Mean theta, phi (rad):", mean_theta , mean_phi)
print("Mean theta, phi (deg):", 180.*mean_theta/np.pi, 180.*mean_phi/np.pi)
print("Mean RA, dec:", mean_RA,mean_dec)

np.savetxt("selection_functions/" + file_mean_RA_dec , np.array([mean_RA,mean_dec]) )


# Now focus on the main area by finding min, max of theta, phi
mintheta, maxtheta = np.min(tflat), np.max(tflat)
minmu, maxmu = np.cos(maxtheta), np.cos(mintheta)
minphi, maxphi = np.min(fflat), np.max(fflat)

print()
print("Min, max of theta:", np.around(mintheta,5), np.around(maxtheta,5))
print("Min, max of phi:", np.around(minphi,5), np.around(maxphi,5))


mmin = np.max((minmu-0.0001,-1.0))
mmax = np.min((maxmu+0.0001,1.0))


# Define NEW boolean function to determine if a set of theta , phi (numpy arrays) are NOT masked (i.e., they ARE in the map)
def mask_bound_rand(abc,theta,phi):
	dec = np.pi/2.  - theta
	ra  = phi
	cond_foot = (ra**2 >= 0.0)
	edges = len(abc.T)
	for i in range(edges):
		cond_foot = cond_foot * (abc[0,i]*ra + abc[1,i]*dec - abc[2,i] > 0)
	return cond_foot


def mask_exczones_rand(trds,theta,phi):
	dec = np.pi/2.  - theta
	ra  = phi
	cond_exc = (ra**2 >= 0.0)
	exczones = len(trds.T)
	for i in range(exczones):
		if trds[0,i]==0:
			cond_exc = cond_exc * ( (ra-trds[1,i])**2 + (dec-trds[2,i])**2 - (trds[3,i])**2 > 0 )
		if trds[0,i]==1:
			cond_exc = cond_exc * np.logical_not( ((ra-trds[1,i])**2 - (trds[3,i]/2.)**2 < 0) &  ((dec-trds[2,i])**2 - (trds[3,i]/2.)**2 < 0) )
	return cond_exc


# Run batches of randoms
n_batches = N_rand//N_rand_batch

phiprime=np.array([])
thetaprime=np.array([])
sel_fun=np.array([])

print()
print("Starting", n_batches,"batches:")
print()

for nb in range(n_batches):
	print("Batch #",nb)
	mu_rand = np.random.uniform(mmin,mmax,N_rand_batch)
	t_rand = np.arccos(mu_rand)
	# Free up memory
	del mu_rand

	fmin = np.max((minphi-0.0001,0.0))
	fmax = np.min((maxphi+0.0001,2.0*np.pi))
	f_rand = np.random.uniform(fmin,fmax,N_rand_batch)

	######################
	# OK, now we have angular randoms that obey the mask, but no "angular selection function" or weights.
	# We will add the selection + weights according to some formula.
	#
	# This formula for the selection and weights MUST BE GIVEN BY THE USER.
	# Here we assume that they are given in terms of the file sel_fun_parameters
	# (whose parameters have been imported at the start of this code)
	#

	print("Computing selection function weights...")
	selweights=np.zeros((len(t_rand),len(pivot)))
	for i in range(len(pivot)):
		p_RA, p_dec = pivot[i][0] , pivot[i][1]
		l_RA, l_dec = lin_par[i][0] , lin_par[i][1]
		q_RA, q_dec = quad_par[i][0] , quad_par[i][1]
		selweights[:,i] = 1.0 + l_RA*(f_rand - p_RA) + l_dec*(np.pi/2 - t_rand - p_dec) + 0.5*q_RA*(f_rand - p_RA)**2 + 0.5*q_dec*(np.pi/2 - t_rand - p_dec)**2

	# Apply mask to this random
	print("Applying mask to random...")
	maskran = mask_exczones_rand(trds,t_rand,f_rand)*mask_bound_rand(abc,t_rand,f_rand)

	t2=t_rand[maskran]
	f2=f_rand[maskran]

	del t_rand
	del f_rand

	# Restrict selection function to mask before deleting variable
	if len(sel_fun) == 0:
		sel_fun = selweights[maskran]
	else:
		sel_fun = np.concatenate((selweights[maskran],sel_fun))
	del selweights

	del maskran

	# Now, rotate the random maps so that the center of the mask aligns with the (new) z axis
	print("Rotate random...")

	xprime = np.sin(t2)*np.sin(mean_phi-f2)
	yprime = np.sin(t2)*np.cos(mean_theta) * np.cos(mean_phi-f2) - np.sin(mean_theta)*np.cos(t2) 
	zprime = np.cos(mean_theta)*np.cos(t2) + np.sin(mean_theta)*np.sin(t2) * np.cos(f2-mean_phi)

	# Check
	print("Check: mean projected x,y --> 0 ; mean z(cartesian) < 1")
	print(np.around(np.mean(xprime),4),np.around(np.mean(yprime),4),np.around(np.mean(zprime),4))

	del t2
	del f2

	# Rotate
	phiprime = np.concatenate((np.arctan2(yprime,xprime),phiprime))
	thetaprime = np.concatenate((np.arccos(zprime),thetaprime))

	del xprime
	del yprime
	del zprime

print()
print("Batches completed. Total number of points in random:", len(phiprime))
print()

print("Plotting footprint and mask...")

jump=len(phiprime)//5000

# Check that area before and after is the same by comparing number of points 
fig = pl.figure()
ax0 = fig.add_subplot(121, projection='polar')
ax0.grid(True)
c1 = ax0.scatter(phiprime[::jump], thetaprime[::jump], c=sel_fun[::jump,0], cmap="hsv", s=0.002, alpha=0.7)
ax1 = fig.add_subplot(122, projection='polar')
ax1.grid(True)
c2 = ax1.scatter(phiprime[::jump], thetaprime[::jump], c=sel_fun[::jump,1], cmap="hsv", s=0.002, alpha=0.7)
pl.savefig("selection_functions/mask_polar_rotated.png",dpi=300)
pl.close()

# Weight map
fig = pl.figure()
ax0 = fig.add_subplot(121, projection='mollweide')
ax0.grid(True)
c1 = ax0.scatter(phiprime[::jump], np.pi/2-thetaprime[::jump], c=sel_fun[::jump,0], cmap="hsv", s=0.002, alpha=0.7)
ax1 = fig.add_subplot(122, projection='mollweide')
ax1.grid(True)
c2 = ax1.scatter(phiprime[::jump], np.pi/2-thetaprime[::jump], c=sel_fun[::jump,1], cmap="hsv", s=0.002, alpha=0.7)
pl.savefig("selection_functions/mask_mollweide_rotated.png",dpi=300)
pl.close()


fig = pl.figure()
ax0 = fig.add_subplot(121, projection='lambert')
ax0.grid(True)
c1 = ax0.scatter(phiprime[::jump], np.pi/2 - thetaprime[::jump], c=sel_fun[::jump,0], cmap="hsv", s=0.002, alpha=0.7)
ax1 = fig.add_subplot(122, projection='lambert')
ax1.grid(True)
c2 = ax1.scatter(phiprime[::jump], np.pi/2 - thetaprime[::jump], c=sel_fun[::jump,1], cmap="hsv", s=0.002, alpha=0.7)
pl.savefig("selection_functions/mask_lambert_rotated.png",dpi=300)
pl.close()


print("Saving angular footprint & mask... (be patient: for large files it may take a while...)")

h5f = h5py.File("selection_functions/" + file_angular_mask_random_weights , 'w')
h5f.create_dataset('randoms', data=np.vstack([thetaprime,phiprime,sel_fun.T]).T, dtype='float32',compression='gzip')
h5f.close()

print("Done! Mask and weights saved to selection_functions/.")

