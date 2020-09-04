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


# This is the file where we have a catalog that will be rotated.
# ATTENTION: all entries must be floats or integers (no text allowed here)
cat_name = "testcat.hdf5"
print("Rotating catalog ", cat_name)

# You must also define the column where the RAs and decs -- in "Pythonesque": 0, 1, ...
col_RA  = 0
col_dec = 1

# Specify if these are in degrees or in radians
# If in deg, True , if in radians, False
angle_deg = False

# Rotate to align this position with the center:
central_RA_dec_file = 'mean_RA_dec.dat'

print("Reading center (RA, dec) of footprint on file", central_RA_dec_file)

# Read center of map that will be rotated to these positions
ra_dec_center = np.loadtxt(central_RA_dec_file)

# If RA and dec are given in degrees, correct this.
if ( (ra_dec_center[0] > 6.1) | (ra_dec_center[1] < -1.58 ) | (ra_dec_center[1] > 1.58)):
	mean_theta = np.pi/2 - np.pi*ra_dec_center[1]/180.
	mean_phi = np.pi*ra_dec_center[0]/180.
else:
	mean_theta = np.pi/2 - ra_dec_center[1]
	mean_phi = ra_dec_center[0]

print("RA, dec center:", ra_dec_center)
print("theta, phi center:", mean_theta,mean_phi)





#########################
#
# Start main code
#


# Type of file
str_ending = cat_name[-3:]

print("Reading catalog...")
# Read catalog
if ( (str_ending == "txt") |  (str_ending == "dat") ) :
	cat = np.loadtxt(cat_name,comments='#')
elif str_ending == "csv" :
	cat = np.loadtxt(cat_name,comments='#',sep=",")
elif str_ending == "df5" :
	h5f = h5py.File(cat_name, 'r')
	h5data = h5f.get(list(h5f.keys())[0])
	cat = np.asarray(h5data,dtype='float32')
	h5f.close

# TESTING:
cat=np.reshape(cat,(3,3489)).T

# Obtain theta and phi from ra, dec:
RA, dec = cat[:,col_RA] , cat[:,col_dec]

if angle_deg:
	theta = np.pi/2.  - np.pi*dec/180.0
	phi  = np.pi*RA/180.0
else:
	theta = np.pi/2.  - dec
	phi  = RA


# Number of objects in catalog
Nobj = len(cat)



# Now, rotate the catalog
print("Rotating ...")
xprime = np.sin(theta)*np.sin(mean_phi-phi)
yprime = np.sin(theta)*np.cos(mean_theta) * np.cos(mean_phi-phi) - np.sin(mean_theta)*np.cos(theta) 
zprime = np.cos(mean_theta)*np.cos(theta) + np.sin(mean_theta)*np.sin(theta) * np.cos(phi-mean_phi)

# Rotate
phiprime = np.arctan2(yprime,xprime)
thetaprime = np.arccos(zprime)

del xprime
del yprime
del zprime



print("Plotting original and rotated catalogs")
fig = pl.figure()
ax0 = fig.add_subplot(111, projection='polar')
ax0.grid(True)
c1 = ax0.scatter(phi, theta, s=0.002, alpha=0.7)
pl.savefig("selection_functions/cat_polar.png",dpi=300)
pl.close()

fig = pl.figure()
ax0 = fig.add_subplot(111, projection='polar')
ax0.grid(True)
c1 = ax0.scatter(phiprime, thetaprime, s=0.002, alpha=0.7)
pl.savefig("selection_functions/cat_polar_rotated.png",dpi=300)
pl.close()


fig = pl.figure()
ax0 = fig.add_subplot(111, projection='mollweide')
ax0.grid(True)
c1 = ax0.scatter(phi, theta, s=0.002, alpha=0.7)
pl.savefig("selection_functions/cat_mollweide.png",dpi=300)
pl.close()

fig = pl.figure()
ax0 = fig.add_subplot(111, projection='mollweide')
ax0.grid(True)
c1 = ax0.scatter(phiprime, thetaprime, s=0.002, alpha=0.7)
pl.savefig("selection_functions/cat_mollweide_rotated.png",dpi=300)
pl.close()


# Redefine catalog
cat[:,col_RA] = phiprime
cat[:,col_dec]= thetaprime

print("Saving rotated catalog...")

if ( (str_ending == "txt") | (str_ending == "dat") ) :
	np.savetxt("Rotated_"+ cat_name, cat)
elif str_ending == "csv" :
	np.savetxt("Rotated_"+ cat_name, cat, sep=",")
elif str_ending == "df5" :
	h5f = h5py.File("Rotated_" + cat_name,'w')
	h5f.create_dataset('catalog', data=cat, dtype='float32',compression='gzip')
	h5f.close()

print("Done!")

