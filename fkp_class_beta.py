#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Class for the FKP code developed by Lucas F. Secco(IFUSP)
    Arthur E. da Mota Loureiro(IFUSP)
    04/11/2014
    Changed by R. Abramo - July-August/2015, Feb. 2016
"""
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
    def __init__(self,num_bins,n_bar_matrix,bias,cell_size,n_x,n_y,n_z,n_x_orig,n_y_orig,n_z_orig,bin_matrix,power0,mas_power, multipoles, do_cross_spectra, nhalos):
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

        self.multipoles = multipoles
        self.do_cross_spectra = do_cross_spectra
        self.nhalos = nhalos

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
        # OBS: displace origin in z direction by fraction of a cell, to avoid division by 0 for map around r=0
        LX0 = n_x_orig
        LY0 = n_y_orig
        LZ0 = n_z_orig + 0.01
        
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

        # Mass Assignement Scheme window function -- notice that the numpy function sinc has an implicit factor of PI!
        self.mas_wfunction = np.power( np.sinc(self.grid.KX[:,:,:self.n_z//2+1])*np.sinc(self.grid.KY[:,:,:self.n_z//2+1])*np.sinc(self.grid.KZ[:,:,:self.n_z//2+1]) , mas_power)

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

    def fkp(self, ng):
        '''
        This is the FKP function itself, only entries are the galaxy maps for all tracers
        '''
        largenumber=100000000.
        small = 1.0/largenumber
        alpha = small

        multipoles = self.multipoles
        do_cross_spectra = self.do_cross_spectra
        nhalos = self.nhalos
    
        lenkf = int(self.n_x*self.n_y*(self.n_z//2+1))
        
        # Weighted overdensity field -- eq. 6 in PVP
        self.F = np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z))
        self.F_bar = np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z))
        Fk = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))        
        
        Frxx_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        Frxy_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        Frxz_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        Fryy_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        Fryz_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))
        Frzz_k = (1.+ 0.0*1j)*np.zeros((self.number_tracers, self.n_x, self.n_y, self.n_z//2+1))

        if multipoles == 0: #Monopole
            F0kflat = (1.+0.0*1j)*np.zeros((self.number_tracers, lenkf))

            for i in range(self.number_tracers):                                                        
                self.F[i] = (self.w[i]/(self.N[i]*self.bias[i] + small)) * (ng[i])                      
                self.F_bar[i] = (self.w[i]/(self.N[i]*self.bias[i] + small)) * ( - alpha*self.nr[i])    
                Fk[i] = np.fft.rfftn(self.F[i]) / self.mas_wfunction + np.fft.rfftn(self.F_bar[i])      
                Fi = self.F[i] + self.F_bar[i]                                                          
                F0kflat[i] = np.ndarray.flatten(Fk[i])                                                  

                # Compute the multipoles                                                                
                Frxx_k[i]=np.fft.rfftn(Fi*(self.rxhat)*(self.rxhat))                                    
                Frxy_k[i]=np.fft.rfftn(Fi*(self.rxhat)*(self.ryhat))                                    
                Frxz_k[i]=np.fft.rfftn(Fi*(self.rxhat)*(self.rzhat))                                    
                Fryy_k[i]=np.fft.rfftn(Fi*(self.ryhat)*(self.ryhat))                                    
                Fryz_k[i]=np.fft.rfftn(Fi*(self.ryhat)*(self.rzhat))                                    
                Frzz_k[i]=np.fft.rfftn(Fi*(self.rzhat)*(self.rzhat))

            F_ret = self.F + self.F_bar

            self.Fk=Fk

            ntr = self.number_tracers
            Fkf2 = np.zeros((ntr,lenkf))
            for i in range(self.number_tracers):
                # Monopole: F(0)*F(0)^*
                Fkf2[i] = ( F0kflat[i]*(F0kflat[i].conj()) ).real

            self.F0kflat = F0kflat  
            ###############################################################
            counts=np.ones(self.num_bins) #initializing the vector that averages over modes within a bin     
            # Here are the operations involving the M-matrix = bin_matrix)
            counts = (self.bin_matrix).dot(np.ones(lenkf))
            self.counts = counts
            V_k = counts/((self.n_x*self.n_y)*(self.n_z/2.+1)+small)                                         
            rel_var = np.zeros((self.number_tracers,self.num_bins)) #initializing relative variance vector
            P_ret = np.zeros((self.number_tracers,self.num_bins))
            self.Pshot_phys = np.zeros(self.number_tracers)
            sigma = np.zeros((self.number_tracers,self.num_bins))

            for i in range(self.number_tracers):
                # This is <|F(k)|^2> on bins [a]
                # Changed to sparse matrix format
                P_ret[i] = ((self.bin_matrix).dot(Fkf2[i]))/(self.counts + small)

                P_ret[i] = np.abs(P_ret[i] - self.Pshot[i])
                ###############################################################

                pifactor=1./(self.N[i]**4 +small)
                rel_var[i] = pifactor*(self.N2[i]*(P_ret[i]**2) + 2.*self.N3[i]*P_ret[i] + self.N4[i])

                rel_var[i] = rel_var[i]/(V_k + small)
                sigma[i] = np.sqrt(rel_var[i]).real #1-sigma error bars vector

                #changing to physical units
                P_ret[i]=P_ret[i]*self.cell_size**3

                self.Pshot_phys[i]=self.Pshot[i]*self.cell_size**3
                # TESTING
                pshotret = (self.Pshot_phys[i])*(self.bias[i])**2
                print("   FKP shot noise for tracer",i," : ", pshotret)

                sigma[i] = sigma[i]*self.cell_size**3

            self.P_ret = P_ret                                                                          
            self.sigma = sigma                                                                          
            self.F_ret = F_ret

            #Cross spectra
            if do_cross_spectra == True and nhalos > 1:
                self.cross_spec = np.zeros(( ntr*(ntr-1)//2, self.num_bins ))
                index = 0
                for i in range(ntr):
                    for j in range(i+1,ntr):
                        cross_temp = np.real( F0kflat[i]*np.conj(F0kflat[j]) )
                        self.cross_spec[index] = self.cell_size**3.*((self.bin_matrix).dot(cross_temp)/(self.counts + small))
                        index += 1
            else:
                pass


        elif multipoles == 2: #Dipole
            F0kflat = (1.+0.0*1j)*np.zeros((self.number_tracers, lenkf))
            F2kflat = (1.+0.0*1j)*np.zeros((self.number_tracers, lenkf))

            for i in range(self.number_tracers):                                                        
                self.F[i] = (self.w[i]/(self.N[i]*self.bias[i] + small)) * (ng[i])                      
                self.F_bar[i] = (self.w[i]/(self.N[i]*self.bias[i] + small)) * ( - alpha*self.nr[i])    
                Fk[i] = np.fft.rfftn(self.F[i]) / self.mas_wfunction + np.fft.rfftn(self.F_bar[i])      
                Fi = self.F[i] + self.F_bar[i]                                                          
                F0kflat[i] = np.ndarray.flatten(Fk[i])                                                  

                # Compute the multipoles                                                                
                Frxx_k[i]=np.fft.rfftn(Fi*(self.rxhat)*(self.rxhat))                                    
                Frxy_k[i]=np.fft.rfftn(Fi*(self.rxhat)*(self.ryhat))                                    
                Frxz_k[i]=np.fft.rfftn(Fi*(self.rxhat)*(self.rzhat))                                    
                Fryy_k[i]=np.fft.rfftn(Fi*(self.ryhat)*(self.ryhat))                                    
                Fryz_k[i]=np.fft.rfftn(Fi*(self.ryhat)*(self.rzhat))                                    
                Frzz_k[i]=np.fft.rfftn(Fi*(self.rzhat)*(self.rzhat))

                F2 = (self.kx_half**2*Frxx_k[i] + \
                      self.ky_half**2*Fryy_k[i] + \
                      self.kz_half**2*Frzz_k[i] + \
                      2.0*self.kx_half*self.ky_half*Frxy_k[i] + \
                      2.0*self.kx_half*self.kz_half*Frxz_k[i] + \
                      2.0*self.ky_half*self.kz_half*Fryz_k[i] ) / self.mas_wfunction

                F2kflat[i] = - F0kflat[i] + 3.0*np.ndarray.flatten(F2)

            F_ret = self.F + self.F_bar

            self.Fk=Fk

            ntr = self.number_tracers
            Fkf2 = np.zeros((ntr,lenkf))
            F2kf2 = np.zeros((ntr,lenkf))
            for i in range(self.number_tracers):
                # Monopole: F(0)*F(0)^*
                Fkf2[i] = ( F0kflat[i]*(F0kflat[i].conj()) ).real
                # Quadrupole, standard: Re[F(2)*F(0)^*] = (1/2)[F(2)*F(0)^* + c.c.]
                F2kf2[i]= 0.5*( F2kflat[i]*(F0kflat[i].conj()) + F0kflat[i]*(F2kflat[i].conj()) ).real

            self.Fkf2 = Fkf2
            self.F2kf2 = F2kf2
            self.F0kflat = F0kflat  
            ###############################################################
            counts=np.ones(self.num_bins) #initializing the vector that averages over modes within a bin     
            # Here are the operations involving the M-matrix = bin_matrix)
            counts = (self.bin_matrix).dot(np.ones(lenkf))
            self.counts = counts
            V_k = counts/((self.n_x*self.n_y)*(self.n_z/2.+1)+small)                                         
            rel_var = np.zeros((self.number_tracers,self.num_bins)) #initializing relative variance vector
            P_ret = np.zeros((self.number_tracers,self.num_bins))
            P2_ret = np.zeros((self.number_tracers,self.num_bins))
            self.Pshot_phys = np.zeros(self.number_tracers)
            sigma = np.zeros((self.number_tracers,self.num_bins))

            for i in range(self.number_tracers):
                # This is <|F(k)|^2> on bins [a]
                # Changed to sparse matrix format
                P_ret[i] = ((self.bin_matrix).dot(Fkf2[i]))/(self.counts + small)
                # Quadrupole with factor of 5/2
                P2_ret[i] = 2.5*((self.bin_matrix).dot(F2kf2[i]))/(self.counts + small)

                P_ret[i] = np.abs(P_ret[i] - self.Pshot[i])
                ###############################################################

                pifactor=1./(self.N[i]**4 +small)
                rel_var[i] = pifactor*(self.N2[i]*(P_ret[i]**2) + 2.*self.N3[i]*P_ret[i] + self.N4[i])

                rel_var[i] = rel_var[i]/(V_k + small)
                sigma[i] = np.sqrt(rel_var[i]).real #1-sigma error bars vector

                #changing to physical units
                P_ret[i]=P_ret[i]*self.cell_size**3
                P2_ret[i]=P2_ret[i]*self.cell_size**3

                self.Pshot_phys[i]=self.Pshot[i]*self.cell_size**3
                # TESTING
                pshotret = (self.Pshot_phys[i])*(self.bias[i])**2
                print("   FKP shot noise for tracer",i," : ", pshotret)

                sigma[i] = sigma[i]*self.cell_size**3

            self.P_ret = P_ret                                                                          
            self.P2_ret = P2_ret                                                                        
            self.sigma = sigma                                                                          
            self.F_ret = F_ret

            #Cross spectra
            if do_cross_spectra == True and nhalos > 1:
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
            else:
                pass

        else: #Quadrupole
            F0kflat = (1.+0.0*1j)*np.zeros((self.number_tracers, lenkf))
            F2kflat = (1.+0.0*1j)*np.zeros((self.number_tracers, lenkf))
            F4kflat = (1.+0.0*1j)*np.zeros((self.number_tracers, lenkf))

            for i in range(self.number_tracers):                                                        
                self.F[i] = (self.w[i]/(self.N[i]*self.bias[i] + small)) * (ng[i])                      
                self.F_bar[i] = (self.w[i]/(self.N[i]*self.bias[i] + small)) * ( - alpha*self.nr[i])    
                Fk[i] = np.fft.rfftn(self.F[i]) / self.mas_wfunction + np.fft.rfftn(self.F_bar[i])      
                Fi = self.F[i] + self.F_bar[i]                                                          
                F0kflat[i] = np.ndarray.flatten(Fk[i])                                                  

                # Compute the multipoles                                                                
                Frxx_k[i]=np.fft.rfftn(Fi*(self.rxhat)*(self.rxhat))                                    
                Frxy_k[i]=np.fft.rfftn(Fi*(self.rxhat)*(self.ryhat))                                    
                Frxz_k[i]=np.fft.rfftn(Fi*(self.rxhat)*(self.rzhat))                                    
                Fryy_k[i]=np.fft.rfftn(Fi*(self.ryhat)*(self.ryhat))                                    
                Fryz_k[i]=np.fft.rfftn(Fi*(self.ryhat)*(self.rzhat))                                    
                Frzz_k[i]=np.fft.rfftn(Fi*(self.rzhat)*(self.rzhat))

                F2 = (self.kx_half**2*Frxx_k[i] + \
                      self.ky_half**2*Fryy_k[i] + \
                      self.kz_half**2*Frzz_k[i] + \
                      2.0*self.kx_half*self.ky_half*Frxy_k[i] + \
                      2.0*self.kx_half*self.kz_half*Frxz_k[i] + \
                      2.0*self.ky_half*self.kz_half*Fryz_k[i] ) / self.mas_wfunction

                # Now, the hexadecapole                                                                 
                F4 =  self.kx_half**4 *(np.fft.rfftn(Fi*(self.rxhat)**4))                               
                F4 += self.ky_half**4 *(np.fft.rfftn(Fi*(self.ryhat)**4))                               
                F4 += self.kz_half**4 *(np.fft.rfftn(Fi*(self.rzhat)**4))                               

                F4 += 4.0 * self.kx_half**3 * self.ky_half * (np.fft.rfftn(Fi*(self.rxhat)**3*(self.ryhat))) 
                F4 += 4.0 * self.kx_half**3 * self.kz_half * (np.fft.rfftn(Fi*(self.rxhat)**3*(self.rzhat))) 
                F4 += 4.0 * self.ky_half**3 * self.kx_half * (np.fft.rfftn(Fi*(self.ryhat)**3*(self.rxhat))) 
                F4 += 4.0 * self.ky_half**3 * self.kz_half * (np.fft.rfftn(Fi*(self.ryhat)**3*(self.rzhat))) 
                F4 += 4.0 * self.kz_half**3 * self.kx_half * (np.fft.rfftn(Fi*(self.rzhat)**3*(self.rxhat))) 
                F4 += 4.0 * self.kz_half**3 * self.ky_half * (np.fft.rfftn(Fi*(self.rzhat)**3*(self.ryhat))) 

                F4 += 6.0 * self.kx_half**2 * self.ky_half**2 * (np.fft.rfftn(Fi*(self.rxhat)**2*(self.ryhat)**2))
                F4 += 6.0 * self.kx_half**2 * self.kz_half**2 * (np.fft.rfftn(Fi*(self.rzhat)**2*(self.rxhat)**2))
                F4 += 6.0 * self.ky_half**2 * self.kz_half**2 * (np.fft.rfftn(Fi*(self.ryhat)**2*(self.rzhat)**2))
                
                F4 += 12.0 * self.kx_half**2 * self.ky_half * self.kz_half * (np.fft.rfftn(Fi*(self.rxhat)**2*(self.ryhat)*(self.rzhat)))                                                                         
                F4 += 12.0 * self.kz_half**2 * self.kx_half * self.ky_half * (np.fft.rfftn(Fi*(self.rzhat)**2*(self.rxhat)*(self.ryhat)))                                                                         
                F4 += 12.0 * self.ky_half**2 * self.kz_half * self.kx_half * (np.fft.rfftn(Fi*(self.ryhat)**2*(self.rxhat)*(self.rzhat)))                                                                         
                F4 = F4 / self.mas_wfunction

                F2kflat[i] = - F0kflat[i] + 3.0*np.ndarray.flatten(F2)
                F4kflat[i] = -7.0/4.0*F0kflat[i] - 2.5*F2kflat[i] + 35.0/4.0*np.ndarray.flatten(F4)

            F_ret = self.F + self.F_bar

            self.Fk=Fk

            ntr = self.number_tracers
            Fkf2 = np.zeros((ntr,lenkf))
            F2kf2 = np.zeros((ntr,lenkf))
            F4kf2 = np.zeros((ntr,lenkf))
            for i in range(self.number_tracers):
                # Monopole: F(0)*F(0)^*
                Fkf2[i] = ( F0kflat[i]*(F0kflat[i].conj()) ).real
                # Quadrupole, standard: Re[F(2)*F(0)^*] = (1/2)[F(2)*F(0)^* + c.c.]
                F2kf2[i]= 0.5*( F2kflat[i]*(F0kflat[i].conj()) + F0kflat[i]*(F2kflat[i].conj()) ).real
                # Hexadecapole
                F4kf2[i]= 0.5*( F4kflat[i]*(F0kflat[i].conj()) + F0kflat[i]*(F4kflat[i].conj()) ).real

            self.Fkf2 = Fkf2
            self.F2kf2 = F2kf2
            self.F4kf2 = F4kf2
            self.F0kflat = F0kflat  
            ###############################################################
            counts=np.ones(self.num_bins) #initializing the vector that averages over modes within a bin     
            # Here are the operations involving the M-matrix = bin_matrix)
            counts = (self.bin_matrix).dot(np.ones(lenkf))
            self.counts = counts
            V_k = counts/((self.n_x*self.n_y)*(self.n_z/2.+1)+small)                                         
            rel_var = np.zeros((self.number_tracers,self.num_bins)) #initializing relative variance vector
            P_ret = np.zeros((self.number_tracers,self.num_bins))
            P2_ret = np.zeros((self.number_tracers,self.num_bins))
            P4_ret = np.zeros((self.number_tracers,self.num_bins))
            self.Pshot_phys = np.zeros(self.number_tracers)
            sigma = np.zeros((self.number_tracers,self.num_bins))

            for i in range(self.number_tracers):
                # This is <|F(k)|^2> on bins [a]
                # Changed to sparse matrix format
                P_ret[i] = ((self.bin_matrix).dot(Fkf2[i]))/(self.counts + small)
                # Quadrupole with factor of 5/2
                P2_ret[i] = 2.5*((self.bin_matrix).dot(F2kf2[i]))/(self.counts + small)
                # Hexadecapole with factor of 9/2
                P4_ret[i] = 4.5*((self.bin_matrix).dot(F4kf2[i]))/(self.counts + small)

                P_ret[i] = np.abs(P_ret[i] - self.Pshot[i])
                ###############################################################

                pifactor=1./(self.N[i]**4 +small)
                rel_var[i] = pifactor*(self.N2[i]*(P_ret[i]**2) + 2.*self.N3[i]*P_ret[i] + self.N4[i])

                rel_var[i] = rel_var[i]/(V_k + small)
                sigma[i] = np.sqrt(rel_var[i]).real #1-sigma error bars vector

                #changing to physical units
                P_ret[i]=P_ret[i]*self.cell_size**3
                P2_ret[i]=P2_ret[i]*self.cell_size**3
                P4_ret[i]=P4_ret[i]*self.cell_size**3

                self.Pshot_phys[i]=self.Pshot[i]*self.cell_size**3
                # TESTING
                pshotret = (self.Pshot_phys[i])*(self.bias[i])**2
                print("   FKP shot noise for tracer",i," : ", pshotret)

                sigma[i] = sigma[i]*self.cell_size**3

            self.P_ret = P_ret                                                                          
            self.P2_ret = P2_ret                                                                        
            self.P4_ret = P4_ret                                                                        
            self.sigma = sigma                                                                          
            self.F_ret = F_ret

            #Cross spectra
            if do_cross_spectra == True and nhalos > 1:
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
                # Defining cross of the quadrupoles
                self.cross_spec4 = np.zeros(( ntr*(ntr-1)//2, self.num_bins ))
                index = 0
                for i in range(ntr):
                    for j in range(i+1,ntr):
                        cross_temp = 0.5*( F4kflat[i]*(F0kflat[j].conj()) + F0kflat[i]*(F4kflat[j].conj()) ).real
                        self.cross_spec4[index] = 4.5*self.cell_size**3.*((self.bin_matrix).dot(cross_temp)/(self.counts + small))
                        index += 1
            else:
                pass        
