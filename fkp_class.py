#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Class for the FKP code developed by Lucas F. Secco(IFUSP)
    Arthur E. da Mota Loureiro(IFUSP)
    04/11/2014
    Changed by R. Abramo - July-August/2015, Feb. 2016
"""
from time import perf_counter
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
    def __init__(self,num_bins,n_bar_matrix,bias,cell_size,n_x,n_y,n_z,n_x_orig,n_y_orig,n_z_orig,bin_matrix,power0):
        # Here bin_matrix is the M-matrix
        self.num_bins = num_bins
        self.n_bar_matrix = n_bar_matrix
        self.bias = bias
        self.cell_size = cell_size
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        self.bin_matrix = bin_matrix
        
        self.phsize_x=float(cell_size*n_x)	#physical size of the side of the (assumed square) grid
        self.phsize_y=float(cell_size*n_y)
        self.phsize_z=float(cell_size*n_z)
        
        self.number_tracers = len(bias)

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

        # Definitions from Percival, Verde & Peacock 2004 (PVP)
        # The units in this subroutine are NOT physical: volume = (number of cells)^3, etc.

        self.Pi = power0*((1./self.cell_size)**3) # initial guess for the power spectrum
        self.nr = np.zeros((self.number_tracers, n_x, n_y, n_z))
        self.w = np.zeros((self.number_tracers, n_x, n_y, n_z))
        
        self.N = np.zeros(self.number_tracers)
        self.N2 = np.zeros(self.number_tracers)
        self.N3 = np.zeros(self.number_tracers)
        self.N4 = np.zeros(self.number_tracers)
        
        self.Pshot = np.zeros(self.number_tracers)
        
        for i in range(self.number_tracers):
            # This is the random catalog
            self.nr[i] = np.random.poisson(n_bar_matrix[i]*largenumber)
        
            #weights according to eq.28 (in PVP) : w = b^2 P/(1 + n b^2 P)
            self.w[i] = ((bias[i]**2)*self.Pi) / (1.0+n_bar_matrix[i]*self.Pi*(bias[i]**2))
        
            # Normalization N -- PVP ,Eq. 7:
            self.N[i] = np.sqrt(np.sum((n_bar_matrix[i]**2)*(self.w[i]**2)))
            
            self.N2[i] = np.sum((n_bar_matrix[i]**4.)*(self.w[i]**4.))
            self.N3[i] = np.sum((n_bar_matrix[i]**3.)*(self.w[i]**4.)/(bias[i]**2.)) # \int dV n^3 w^4/b^2
            self.N4[i] = np.sum((n_bar_matrix[i]**2.)*(self.w[i]**4.)/(bias[i]**4.)) # \int dV n^2 w^4/b^4
    
            # Shot noise -- PVP, Eq. 16:
            # P_shot = (1+alpha)/N^2 * \int dV n w^2/b^2
            self.Pshot[i] = ((1+alpha)/(self.N[i]**2 + small)) * np.sum(n_bar_matrix[i]*((self.w[i]**2)/(bias[i]**2 + small)))

            #########
            # Future: recast this in terms of our new definitions - ASL 2015
            #########

    def fkp(self,ng):
        '''
        This is the FKP function itself, only entries are the galaxy maps for all tracers
        '''
        largenumber=100000000.
        small = 1.0/largenumber
        alpha = small
    
        lenkf = int(self.n_x*self.n_y*(self.n_z//2+1))
        
        # Weighted overdensity field -- eq. 6 in PVP
        self.F = np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z))
        Fk = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        
        Frxx_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        Frxy_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        Frxz_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        Fryy_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        Fryz_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        Frzz_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        
        F0kflat = (1.+0.0*1j)*np.zeros((self.number_tracers, lenkf))
        F2kflat = (1.+0.0*1j)*np.zeros((self.number_tracers, lenkf))

        for i in range(self.number_tracers):
            self.F[i] = (self.w[i]/(self.N[i]*self.bias[i] + small)) * (ng[i] - alpha*self.nr[i])
            Fk[i] = np.fft.rfftn(self.F[i])
            F0kflat[i] = np.ndarray.flatten(Fk[i])
    
            # Compute the multipoles
            Frxx_k[i]=np.fft.rfftn((self.F[i])*(self.rxhat)*(self.rxhat))
            Frxy_k[i]=np.fft.rfftn((self.F[i])*(self.rxhat)*(self.ryhat))
            Frxz_k[i]=np.fft.rfftn((self.F[i])*(self.rxhat)*(self.rzhat))
            Fryy_k[i]=np.fft.rfftn((self.F[i])*(self.ryhat)*(self.ryhat))
            Fryz_k[i]=np.fft.rfftn((self.F[i])*(self.ryhat)*(self.rzhat))
            Frzz_k[i]=np.fft.rfftn((self.F[i])*(self.rzhat)*(self.rzhat))
            
            F2kflat[i] = - F0kflat[i] + 3.0*np.ndarray.flatten(self.kx_half**2*Frxx_k[i]+ self.ky_half**2*Fryy_k[i]+ self.kz_half**2*Frzz_k[i]+ 2.0*self.kx_half*self.ky_half*Frxy_k[i]+ 2.0*self.kx_half*self.kz_half*Frxz_k[i]+ 2.0*self.ky_half*self.kz_half*Fryz_k[i])
        
        F_ret = self.F
        
        #numpy.fft is in the same Fourier convention of PVP - no extra normalization needed
        self.Fk=Fk               

        Frxx_k=Frxx_k
        Frxy_k=Frxy_k
        Frxz_k=Frxz_k
        Fryy_k=Fryy_k
        Fryz_k=Fryz_k
        Frzz_k=Frzz_k
        ntr = self.number_tracers
        Fkf2 = np.zeros((ntr,lenkf))
        F2akf2 = np.zeros((ntr,lenkf))
        for i in range(self.number_tracers):
            # Monopole: F(0)*F(0)^*
            Fkf2[i] = ( F0kflat[i]*(F0kflat[i].conj()) ).real
            # Quadrupole, standard: Re[F(2)*F(0)^*] = (1/2)[F(2)*F(0)^* + c.c.]
            F2akf2[i]= 0.5*( F2kflat[i]*(F0kflat[i].conj()) + F0kflat[i]*(F2kflat[i].conj()) ).real
            
        self.Fkf2 = Fkf2
        self.F2akf2 = F2akf2
        self.F0kflat = F0kflat
        ###############################################################
        counts=np.ones(self.num_bins) #initializing the vector that averages over modes within a bin 
    
        # Here are the operations involving the M-matrix = bin_matrix)
        counts = (self.bin_matrix).dot(np.ones(lenkf))
        self.counts = counts
        V_k = counts/((self.n_x*self.n_y)*(self.n_z/2.+1)+small)

        rel_var = np.zeros((self.number_tracers,self.num_bins)) #initializing relative variance vector
        P_ret = np.zeros((self.number_tracers,self.num_bins))
        P2a_ret = np.zeros((self.number_tracers,self.num_bins))
        self.Pshot_phys = np.zeros(self.number_tracers)
        sigma = np.zeros((self.number_tracers,self.num_bins))
            
        for i in range(self.number_tracers):
            # This is <|F(k)|^2> on bins [a]
            # Changed to sparse matrix format
            P_ret[i] = ((self.bin_matrix).dot(Fkf2[i]))/(self.counts + small)
            # Quadrupole with factor of 5/2
            P2a_ret[i] = 2.5*((self.bin_matrix).dot(F2akf2[i]))/(self.counts + small)
            
            P_ret[i] = np.abs(P_ret[i] - self.Pshot[i])

            ###############################################################
    
            pifactor=1./(self.N[i]**4 +small)
            rel_var[i] = pifactor*(self.N2[i]*(P_ret[i]**2) + 2.*self.N3[i]*P_ret[i] + self.N4[i])
    
            rel_var[i] = rel_var[i]/(V_k + small)
            sigma[i] = np.sqrt(rel_var[i]).real #1-sigma error bars vector

            #changing to physical units
            P_ret[i]=P_ret[i]*self.cell_size**3
            P2a_ret[i]=P2a_ret[i]*self.cell_size**3
        
            self.Pshot_phys[i]=self.Pshot[i]*self.cell_size**3
            # TESTING
            pshotret = (self.Pshot_phys[i])*(self.bias[i])**2
            print("   FKP shot noise for tracer",i," : ", pshotret)

            sigma[i] = sigma[i]*self.cell_size**3
            
        self.P_ret = P_ret
        self.P2a_ret = P2a_ret
        self.sigma = sigma
        self.F_ret = F_ret
    
        self.cross_spec = np.zeros(( ntr*(ntr-1)//2, self.num_bins ))
        index = 0
        for i in range(ntr):
            for j in range(i+1,ntr):
                cross_temp = np.real( F0kflat[i]*np.conj(F0kflat[j]) )
                self.cross_spec[index] = self.cell_size**3.*((self.bin_matrix).dot(cross_temp)/(self.counts + small)) 
                index += 1

        # Defining cross of the quadrupoles
        self.cross_spec2 = np.zeros(( ntr*(ntr-1)//2, self.num_bins ))
        index = 0
        for i in range(ntr):
            for j in range(i+1,ntr):
                # Quadrupole, standard: Re[F(2)*F(0)^*] = (1/2)[F(2)*F(0)^* + c.c.]
                cross_temp = 0.5*( F2kflat[i]*(F0kflat[j].conj()) + F0kflat[i]*(F2kflat[j].conj()) ).real
                self.cross_spec2[index] = 2.5*self.cell_size**3.*((self.bin_matrix).dot(cross_temp)/(self.counts + small)) 
                index += 1

	
		
		
		
		
		
		
		
		
		
		
		
		
		
