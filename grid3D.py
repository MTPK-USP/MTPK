#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Creates a 3D grid in Fourier space that obeys how the FFT behaves in python

	v0.1
	v1.0 - In 3D
	v1.5 - It can plot slices of the matrix
	v1.7 - Uses the side of the box 
	v2.0 - Uses Einsum to generate the grid
	Arthur E. da Mota Loureiro
		12/12/2013
"""
import numpy as np
#import pylab as pl
#from mpl_toolkits.mplot3d.axes3d import Axes3D
#from matplotlib import cm
####################################################
# Uncomment the line above and the last three lines 
# if you have matplotlib and want to see the grid
####################################################
from time import perf_counter
import numpy as np

class grid3d:
      '''
      The input is the size of the vectors k_x, k_y and k_z
      '''
      def __init__(self,n_x,n_y,n_z,L_x,L_y,L_z):
            self.size_x = n_x
            self.size_y = n_y
            self.size_z = n_z
            self.Lx = L_x
            self.Ly = L_y
            self.Lz = L_z

            ##################
            # grid in k space
            ##################
            self.k_x = np.fft.fftfreq(n_x)
            identx = np.ones_like(self.k_x)

            self.k_y = np.fft.fftfreq(n_y)
            identy = np.ones_like(self.k_y)

            self.k_z = np.fft.fftfreq(n_z)
            identz = np.ones_like(self.k_z)

            self.KX = np.einsum('i,j,k', self.k_x,identy,identz)
            self.KY = np.einsum('i,j,k', identx,self.k_y,identz)
            self.KZ = np.einsum('i,j,k', identx,identy,self.k_z)
            self.KX2 = np.einsum('i,j,k', self.k_x*self.k_x,identy,identz)
            self.KY2 = np.einsum('i,j,k', identx,self.k_y*self.k_y,identz)
            self.KZ2 = np.einsum('i,j,k', identx,identy,self.k_z*self.k_z)

            self.grid_k = np.sqrt(self.KX2 + self.KY2 + self.KZ2)


            ####################################################
            # Generating a grid a real space, uses grid unities
            ####################################################
            r_x = np.arange(n_x) #*(L_x/n_x)
            self.r_x = r_x
            r_y = np.arange(n_y) #*(L_y/n_y)
            self.r_y = r_y
            r_z = np.arange(n_z) #*(L_z/n_z)
            self.r_z = r_z

            self.RX = np.einsum('i,j,k', r_x,identy,identz)
            self.RY = np.einsum('i,j,k', identx,r_y,identz)
            self.RZ = np.einsum('i,j,k', identx,identy,r_z)

            self.RX2 = np.einsum('i,j,k', r_x*r_x,identy,identz)
            self.RY2 = np.einsum('i,j,k', identx,r_y*r_y,identz)
            self.RZ2 = np.einsum('i,j,k', identx,identy,r_z*r_z)
        
            self.grid_r = np.sqrt(self.RX2 + self.RY2 + self.RZ2)

#		pl.figure("Matriz de k")

#		self.plot = pl.imshow(self.matrix[3], cmap=cm.jet)
#		self.plot = pl.imshow(self.grid_r[3], cmap=cm.jet)#, interpolation="nearest")
#		pl.colorbar()
#           self.plothist = pl.imshow(self.hist[3], cmap=cm.jet)
#           pl.show()
