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



############################################################################
#
#  USER SPECIFICATIONS
#

# Original catalog of angles, redshifts, etc. -- MUST BE IN HDF5 !
# Columns of original file: 0 (RA)  1 (dec) ... N_c
# Columns of *output* file: 0 (RA)  1 (dec) ... N_c , weight
orig_cat_file = "test_cat.hdf5" 

nt = 1            # Specify which tracer this catalog refers to. This will choose which weights to use.
n_batches = 9    # number of batches (chunks) to process the data
radec_deg = True  # Specify if angles are in degrees or radians; True -> deg, False -> radians

jump = 2          # Use N/jump points in plots

#
############################################################################



############################################################################
# Begin code

orig_cat_root = orig_cat_file[:-5]

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

# Mask boundaries (footprint) and exclusion zones
abc = np.asarray([a_edge,b_edge,c_edge])
trds = np.asarray([exclusion_zones_types,exclusion_zones_RA,exclusion_zones_dec,exclusion_zones_sizes])


print()
print("Reading catalog with positions (angles x 2, redshifts, ... of objects...")


# Read random/data catalog
h5map = h5py.File(orig_cat_file, 'r')
h5data = h5map.get(list(h5map.keys())[0])
orig_cat = np.asarray(h5data,dtype='float32')
h5map.close

# Dimension of catalog
[ Nobj , Ncol ] =  orig_cat.shape

radec = orig_cat[:,:2]

# If original catalog is in degrees, convert to radians
if radec_deg:
	radec = np.pi/180.*radec

# Define angles theta, phi of spherical coordinates
thetaphi = np.array([ np.pi/2 - radec[:,1] , radec[:,0]]).T


# Define boolean function to determine if a set of theta , phi (numpy arrays) are NOT masked (i.e., they ARE in the map)
def mask_bound(abc, ras, decs):
	cond_foot = (ras**2 >= 0.0)
	edges = len(abc.T)
	for i in range(edges):
		cond_foot = cond_foot * (abc[0,i]*ras + abc[1,i]*decs - abc[2,i] > 0)
	return cond_foot


def mask_exczones(trds, ras, decs):
	cond_exc = (ras**2 >= 0.0)
	exczones = len(trds.T)
	for i in range(exczones):
		if trds[0,i]==0:
			cond_exc = cond_exc * ( (ras - trds[1,i])**2 + (decs - trds[2,i])**2 - (trds[3,i])**2 > 0 )
		if trds[0,i]==1:
			cond_exc = cond_exc * np.logical_not( ((ras - trds[1,i])**2 - (trds[3,i]/2.)**2 < 0) &  ((decs - trds[2,i])**2 - (trds[3,i]/2.)**2 < 0) )
	return cond_exc




# Run batches of randoms
nob_batch = Nobj//n_batches
ninit = Nobj - nob_batch * n_batches

ranges = np.append( ninit , ninit + np.cumsum(nob_batch*np.ones(n_batches)) )
ranges = np.int0( np.append( 0, ranges) )

# Define a mask that is initially all True
mask = np.ones(Nobj) > 0

# Define weights (assume all =1)
weights = np.ones(Nobj)


for nb in range(n_batches+1):
	print("Batch #",nb)
	nob = ranges[nb+1] - ranges[nb]
	ras , decs = radec[ranges[nb]:ranges[nb+1],0] , radec[ranges[nb]:ranges[nb+1],1]
	print("Computing selection function weights...")
	p_RA, p_dec = pivot[nt][0] , pivot[nt][1]
	l_RA, l_dec = lin_par[nt][0] , lin_par[nt][1]
	q_RA, q_dec = quad_par[nt][0] , quad_par[nt][1]
	weights[ranges[nb]:ranges[nb+1]] = 1.0 + l_RA*(ras - p_RA) + l_dec*(decs - p_dec) + 0.5*q_RA*(ras - p_RA)**2 + 0.5*q_dec*(decs - p_dec)**2

	# Apply mask to this random
	print("Applying mask to random...")
	mask[ranges[nb]:ranges[nb+1]] = mask_exczones(trds,ras,decs)*mask_bound(abc,ras,decs)

print()
print("Batches completed.")
print("Of initial", Nobj, "objects, those remaining inside mask are:", len(mask[mask]))
print()

masked_cat = orig_cat[mask]
masked_weights = weights[mask]

masked_radec = radec[mask]
masked_thetaphi = np.array([ np.pi/2 - masked_radec[:,1] , masked_radec[:,0]]).T

print("Plotting footprint/mask...")

# Plot original footprint of catalog
fig = pl.figure()
ax0 = fig.add_subplot(121, projection='polar')
ax0.grid(True)
c1 = ax0.scatter(thetaphi[::jump,1], thetaphi[::jump,0], marker="o", s=0.5, alpha=0.9 , c='b')
c2 = ax0.scatter(masked_thetaphi[:,1], masked_thetaphi[:,0], marker="o", s=0.5, alpha=0.9, c='r' )
pl.savefig("Polar_" + orig_cat_root +  ".png",dpi=300)
pl.close()

fig = pl.figure()
ax0 = fig.add_subplot(121, projection='mollweide')
ax0.grid(True)
c1 = ax0.scatter(thetaphi[::jump,1], np.pi/2. - thetaphi[::jump,0], marker="o", s=0.5, alpha=0.9 , c='b')
c2 = ax0.scatter(masked_thetaphi[::jump,1], np.pi/2. -  masked_thetaphi[::jump,0], marker="o", s=0.5, alpha=0.9 , c='r')
pl.savefig("Mollweide_" + orig_cat_root +  ".png",dpi=300)
pl.close()

fig = pl.figure()
ax0 = fig.add_subplot(121, projection='lambert')
ax0.grid(True)
c1 = ax0.scatter(thetaphi[::jump,1], np.pi/2. - thetaphi[::jump,0], marker="o", s=0.5, alpha=0.9 , c='b')
c2 = ax0.scatter(masked_thetaphi[::jump,1], np.pi/2. - masked_thetaphi[::jump,0], marker="o", s=0.5, alpha=0.9 , c='r')
pl.savefig("Lambert_" + orig_cat_root +  ".png",dpi=300)
pl.close()



print("Saving masked catalog... (be patient: for large files it may take a while...)")



h5f = h5py.File("selection_functions/Masked_" + orig_cat_file , 'w')
h5f.create_dataset('cat', data=np.vstack([masked_cat.T,masked_weights]).T, dtype='float32',compression='gzip')
h5f.close()

print("Done! Masked catalog saved to selection_functions/Masked_" + orig_cat_file)

