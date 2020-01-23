#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Class for the FKP code developed by Lucas F. Secco(IFUSP)
    Arthur E. da Mota Loureiro(IFUSP)
    04/11/2014
    Changed by R. Abramo - July-August/2015, Feb. 2016
"""
from time import clock
import numpy as np
from scipy.sparse import csc_matrix
import grid3D as gr

class fkp_init(object):
    '''
    This class contains the initial calculations for the FKP routine
    but this is the part to be called outside the N_realz loop, since w and N are
    the same for all realizations
    Enters num_bins, n_bar (matrix), bias 
    n_x,n_y, n_z and the bin_matrix
    '''
    def __init__(self,num_bins,n_bar_matrix,bias_single,cell_size,n_x,n_y,n_z,n_x_orig,n_y_orig,n_z_orig,bin_matrix,power0):
        # Here bin_matrix is the M-matrix
        self.num_bins = num_bins
        self.n_bar_matrix = n_bar_matrix
        self.bias_single = bias_single
        self.cell_size = cell_size
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.bin_matrix = bin_matrix
        self.phsize_x=float(cell_size*n_x)	#physical size of the side of the (assumed square) grid
        self.phsize_y=float(cell_size*n_y)
        self.phsize_z=float(cell_size*n_z)
        
        largenumber = 100000000.
        small = 1.0/largenumber
        alpha = small
        
        L_x = n_x*cell_size
        L_y = n_y*cell_size
        L_z = n_z*cell_size
        
        # Compute the grids for \hat{r}^i, \hat{k}^i, \hat{r}^i \hat{r}^j, and \hat{k}^i \hat{k}^j
        self.grid = gr.grid3d(n_x,n_y,n_z,L_x,L_y,L_z)

        # Shift origin of the grid so that the origin is in the right place
        # NOTE: units of cells
        LX0 = n_x_orig
        LY0 = n_y_orig
        LZ0 = n_z_orig
        
        rL0=np.sqrt((LX0 + self.grid.RX)**2 + (LY0 + self.grid.RY)**2 + (LZ0 + self.grid.RZ)**2)
        self.rxhat = (LX0 + self.grid.RX)/rL0
        self.ryhat = (LY0 + self.grid.RY)/rL0
        self.rzhat = (LZ0 + self.grid.RZ)/rL0
        
        self.kxhat = ((self.grid.KX)/(self.grid.grid_k+small))
        self.kyhat = ((self.grid.KY)/(self.grid.grid_k+small))
        self.kzhat = ((self.grid.KZ)/(self.grid.grid_k+small))

        # We only need these arrays in 1/2 of the Fourier space, since the underlying field is real
        self.kx_half = self.kxhat[:,:,:self.n_z//2+1]
        self.ky_half = self.kyhat[:,:,:self.n_z//2+1]
        self.kz_half = self.kzhat[:,:,:self.n_z//2+1]

        # self.rxxhat = self.rxhat**2
        # self.ryyhat = self.ryhat**2
        # self.rzzhat = self.rzhat**2
        # self.rxyhat = self.rxhat*self.ryhat
        # self.rxzhat = self.rxhat*self.rzhat
        # self.ryzhat = self.ryhat*self.rzhat

        # self.kxxhat = self.kxhat**2
        # self.kyyhat = self.kyhat**2
        # self.kzzhat = self.kzhat**2
        # self.kxyhat = self.kxhat*self.kyhat
        # self.kxzhat = self.kxhat*self.kzhat
        # self.kyzhat = self.kyhat*self.kzhat

        # This is the random catalog
        self.nr = np.random.poisson(n_bar_matrix*largenumber)

        # Definitions from Percival, Verde & Peacock 2004 (PVP)
        # The units in this subroutine are NOT physical: volume = (number of cells)^3, etc.

        self.Pi = power0*((1./self.cell_size)**3) # initial guess for the power spectrum
        # self.Pi = 20000.0*(n_x/self.phsize_x)*(n_y/self.phsize_y)*(n_z/self.phsize_z)
        
        #weights according to eq.28 (in PVP) : w = b^2 P/(1 + n b^2 P)
        self.w = ((bias_single**2)*self.Pi) / (1.0+n_bar_matrix*self.Pi*(bias_single**2))
        
        # Normalization N -- PVP ,Eq. 7:
        # N^2 = \int dV n^2 w^2 = \int dV (n b^2 P/(1 + n b^2 P))^2
        self.N = np.sqrt(np.sum((n_bar_matrix**2)*(self.w**2)))
        
        # Other normalizations used for sigmas
        self.N2 = np.sum((n_bar_matrix**4.)*(self.w**4.))               # \int dV n^4 w^4
        self.N3 = np.sum((n_bar_matrix**3.)*(self.w**4.)/(bias_single**2.))    # \int dV n^3 w^4/b^2
        self.N4 = np.sum((n_bar_matrix**2.)*(self.w**4.)/(bias_single**4.))    # \int dV n^2 w^4/b^4
        
        # Shot noise -- PVP, Eq. 16:
        # P_shot = (1+alpha)/N^2 * \int dV n w^2/b^2
        self.Pshot = ((1+alpha)/(self.N**2 + small)) * np.sum(n_bar_matrix*((self.w**2)/(bias_single**2 + small)))

        #########
        # Future: recast this in terms of our new definitions - ASL 2015
        #########

        

    def fkp(self,ng):
        '''
        This is the FKP function itself, only entry is the galaxy map
        '''
        largenumber=100000000.
        small = 1.0/largenumber
        alpha = small
        
        # Weighted overdensity field -- eq. 6 in PVP
        self.F=(self.w/(self.N*self.bias_single + small)) * (ng - alpha*self.nr)
        F_ret = self.F
        
        Fk=np.fft.rfftn(self.F)
        
        # This mu is only in order to introduce a multipole "by hand"
        # mu = 1.0/(np.sqrt(3.0))*(self.kxhat+self.kyhat+self.kzhat)
        # mu = 1.0/(np.sqrt(2.0))*(0*self.kxhat+self.kyhat+self.kzhat)
        # should be this:
        # mu = self.kzhat


        # Compute the multipoles
        #Frx_k=np.fft.fftn((self.F)*(self.rxhat))
        #Fry_k=np.fft.fftn((self.F)*(self.ryhat))
        #Frz_k=np.fft.fftn((self.F)*(self.rzhat))
        Frxx_k=np.fft.rfftn((self.F)*(self.rxhat)*(self.rxhat))
        Frxy_k=np.fft.rfftn((self.F)*(self.rxhat)*(self.ryhat))
        Frxz_k=np.fft.rfftn((self.F)*(self.rxhat)*(self.rzhat))
        Fryy_k=np.fft.rfftn((self.F)*(self.ryhat)*(self.ryhat))
        Fryz_k=np.fft.rfftn((self.F)*(self.ryhat)*(self.rzhat))
        Frzz_k=np.fft.rfftn((self.F)*(self.rzhat)*(self.rzhat))
        
        #numpy.fft is in the same Fourier convention of PVP - no extra normalization needed
        Fk=Fk                   #????????? WTF??... Seems to need these lines... Python bug?...
        #Frx_k=Frx_k
        #Fry_k=Fry_k
        #Frz_k=Frz_k
        Frxx_k=Frxx_k
        Frxy_k=Frxy_k
        Frxz_k=Frxz_k
        Fryy_k=Fryy_k
        Fryz_k=Fryz_k
        Frzz_k=Frzz_k
        
        # These are the multipoles of the F field:
        F0kflat = np.ndarray.flatten(Fk)
        
        # F1kflat = np.ndarray.flatten(kx_half*Frx_k[:,:,:self.n_z/2-1] + ky_half*Fry_k[:,:,:self.n_z/2-1] + kz_half*Frz_k[:,:,:self.n_z/2-1])

        # ATTENTION! Changed this definition wrt previous versions!!
        F2kflat = - F0kflat + 3.0*np.ndarray.flatten(\
                                     self.kx_half**2*Frxx_k
                                     + self.ky_half**2*Fryy_k
                                     + self.kz_half**2*Frzz_k
                                     + 2.0*self.kx_half*self.ky_half*Frxy_k
                                     + 2.0*self.kx_half*self.kz_half*Frxz_k
                                     + 2.0*self.ky_half*self.kz_half*Fryz_k)

        lenkf = len(F0kflat)
        # Monopole: F(0)*F(0)^*
        Fkf2=( F0kflat*(F0kflat.conj()) ).real
        
        # Quadrupole, standard: Re[F(2)*F(0)^*] = (1/2)[F(2)*F(0)^* + c.c.]
        F2akf2= 0.5*( F2kflat*(F0kflat.conj()) + F0kflat*(F2kflat.conj()) ).real
        # Dipole^2:
        # F1kf2= ( F1kflat*(F1kflat.conj()) ).real
        # Quadrupole, my way: (1/4) [F(2)*F(0)^* + c.c.] - (1/4) F(0)*F(0)^* + (3/4) F(1)*F(1)^*
        # F2bkf2= 0.25*( F2kflat*(F0kflat.conj()) + F0kflat*(F2kflat.conj()) ).real - 0.25*Fkf2 + 0.75*F1kf2
        

        ###############################################################
        #P_ret=np.zeros(self.num_bins) #initializing the Power Spectrum that will be the output of the external function
        counts=np.ones(self.num_bins) #initializing the vector that averages over modes within a bin 
        init=clock()

        # Here are the operations involving the M-matrix = bin_matrix)
        #self.counts2 = np.einsum("aijl->a", self.bin_matrix)
        #counts = np.einsum("aijl->a", self.bin_matrix)  # number of points in each bin a
        counts = (self.bin_matrix).dot(np.ones(lenkf))
        self.counts = counts
        
        # This is <|F(k)|^2> on bins [a]
        # Changed to sparse matrix format
        P_ret = ((self.bin_matrix).dot(Fkf2))/(self.counts + small)
        P2a_ret = ((self.bin_matrix).dot(F2akf2))/(self.counts + small)
        # P2b_ret = ((self.bin_matrix).dot(F2bkf2))/(self.counts + small)
        # P_ret = np.einsum("aijl,ijl,ijl->a", self.bin_matrix, Fk, np.conj(Fk))/(np.einsum("aijl->a", self.bin_matrix) + small)

        fin=clock()
        #print '---averaging over shells in k-space took',fin-init,'seconds'

        #P_ret = P_ret/counts - self.Pshot #mean power on each bin and shot noise correction
        P_ret = np.abs(P_ret - self.Pshot)
        #P_ret[np.where(P_ret < 0.0)] = 0.0
        #P2b_ret = P2b_ret + 0.25*(self.Pshot)
        # for i in range(len(P_ret)):
            #if np.sign(P_ret[i])==-1:
            #	P_ret[i]=0.0
        ###############################################################

        #print '\nCalculating error bars'

        init=clock()
        rel_var2=np.zeros(len(P_ret)) #initializing relative variance vector

        #nbarw2=(self.n_bar_matrix*self.w)**2
        #pifactor=((2*np.pi)**3)/(self.N**4 +small) #useful expressions
        pifactor=1./(self.N**4 +small)
        #nbarwb2=(self.n_bar_matrix)*((self.w/self.bias)**2)
        #for i in range(len(P_ret)):
        #	rel_var2[i]=( (pifactor) * np.sum( (nbarw2 + nbarwb2/P_ret[i])**2 )) #eq. 26 from PVP, except for the V_k term, which I include a few lines ahead
        #		rel_var = pifactor*(self.N2 + 2.*self.N3*np.power(P_ret,-1.) + self.N4*np.power(P_ret,-2.))
        #rel_var = pifactor*(self.N2 + 2.*self.N3/(P_ret+small) + self.N4/(P_ret**2 + small))
        rel_var = pifactor*(self.N2*(P_ret**2) + 2.*self.N3*P_ret + self.N4)
        #self.rel_var = rel_var
        #self.rel_var2 = rel_var2
        fin=clock()
        #print '---took',fin-init,'seconds'

        #V_k = counts/ ( (self.n_x/2.0)*self.n_x**2 + small) #this factor of volume is the fraction of modes that fell within each bin, makes more sense in this discrete case instead of 4*pi*(k**2)*(delta k)
        V_k = counts/((self.n_x*self.n_y)*(self.n_z/2.+1)+small) 
        rel_var=rel_var/(V_k + small)
        sigma=np.sqrt(rel_var).real #1-sigma error bars vector

        #changing to physical units
        #P_ret=P_ret*((self.phsize/self.n_x)**3) 
        P_ret=P_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
        P2a_ret=P2a_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
        # P2b_ret=P2b_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
        
        #P_ret2 = P_ret2*((self.phsize/self.n_x)**3)
        #self.Pshot_phys=self.Pshot*((self.phsize/self.n_x)**3) 
        self.Pshot_phys=self.Pshot*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
        #k=k*(2*np.pi*self.n_x/self.phsize)
        #sigma=sigma*((self.phsize/self.n_x)**3)
        sigma=sigma*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
        #eliminating the first 2 and last value, which are problematic, should be fixed
        #self.P_ret=np.abs(P_ret)
        self.P_ret=P_ret
        self.P2a_ret=P2a_ret
        # self.P2b_ret=P2b_ret
        self.F_ret=F_ret
        #self.P_ret2=P_ret2[1:]
        #self.kk=k
        self.sigma=sigma
        self.Fk = Fk
		
		
		
		
		
		
		
		
		
		
		
		
		
		
		
