#! /usr/bin/env python
# -*- coding: utf-8 -*-
########################################
# This is the selection function module
# Use any definition of the selection function in terms of the grid
# The parameters must be defined in the input file
########################################
import numpy as np

def selection_func_Gaussian(gridr,n0,b,r_bar):
	"""
	number of parameters = 3
	n0 = mean number of galaxies/cell
	c1=b
	c2=c2
	c3=k0
	"""
	return n0*np.exp(-((gridr-b)**2)/r_bar**2)

def selection_func_Linear(xg,yg,zg,n0,ax,ay,az):
	"""
	number of parameters = 4
	n0 = mean number of galaxies/cell
	ax, ay, az = linear fits for variation the x, y, z directions
	"""
	x0, y0, z0 = np.mean(xg[:,0,0]) , np.mean(yg[0,:,0]) , np.mean(zg[0,0,:])
	return n0*(1 + ax*(xg-x0) + ay*(yg-y0) + az*(zg-z0) )
