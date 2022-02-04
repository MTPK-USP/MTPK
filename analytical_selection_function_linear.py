########################################
# This is the selection function module
# Use any definition of the selection function in terms of the grid
# The parameters must be defined in the input file
########################################
import numpy as np
from angular_mask_footprint import *
from angular_mask_exclusion_zones import *
def selection_func(xg,yg,zg,n0,ax,ay,az):
	"""
	number of parameters = 4
	n0 = mean number of galaxies/cell
	ax, ay, az = linear fits for variation the x, y, z directions
	"""
	x0, y0, z0 = np.mean(xg[:,0,0]) , np.mean(yg[0,:,0]) , np.mean(zg[0,0,:])
	return n0*(1 + ax*(xg-x0) + ay*(yg-y0) + az*(zg-z0) )
