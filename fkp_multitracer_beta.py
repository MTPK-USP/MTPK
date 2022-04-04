#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Class for the FKP code developed by Lucas F. Secco(IFUSP)
    Arthur E. da Mota Loureiro(IFUSP)
    04/11/2014
    Changed by R. Abramo - July-August/2015, Feb. 2016
    Multi-tracer stuff: RA, March 2016
"""
import numpy as np
from scipy.sparse import csc_matrix
import grid3D as gr
import sys

class fkp_init(object):
    '''
    This class contains the initial calculations for the FKP routine
    but this is the part to be called outside the N_realz loop, since w and N are
    the same for all realizations
    Enters num_bins, n_bar (matrix), bias 
    n_x,n_y, n_z and the bin_matrix
    '''
    def __init__(self,num_bins,n_bar_matrix,bias,cell_size,n_x,n_y,n_z,n_x_orig,n_y_orig,n_z_orig,bin_matrix,power0,mas_power, multipoles):

        self.bin_matrix = bin_matrix
        self.num_bins = num_bins
        self.n_bar_matrix = n_bar_matrix

        self.bias = bias
        
        number_tracers = len(bias)
        
        self.cell_size = cell_size
        self.n_x = n_x
        self.n_y = n_y
        self.n_z = n_z
        
        L_x = n_x*cell_size
        L_y = n_y*cell_size
        L_z = n_z*cell_size
        
        self.phsize_x=float(L_x)	#physical size of the sides of the grid
        self.phsize_y=float(L_y)
        self.phsize_z=float(L_z)

        # Corrected redundant definitions -- remember to kill Arthur
        # IF CHANGED HERE, MUST BE CHANGED EVERYWHERE!!
        # DON'T TOUCH IF YOU ARE NOT SURE WHAT YOU ARE DOING!
        self.largenumber = 100000000.0
        small = 1.0/self.largenumber
        self.alpha = 0.00000001
        
        # Compute the grids in real space and Fourier space
        self.grid = gr.grid3d(n_x,n_y,n_z,L_x,L_y,L_z)

        # Shift origin of the grid so that it's in the right place
        # NOTE: in units of cells
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
        
        # These are the random catalogs
        self.nr = np.zeros((number_tracers,n_x,n_y,n_z))
        for nt in range(number_tracers):
            self.nr[nt] = np.random.poisson(1./self.alpha * n_bar_matrix[nt])

        # We will need flattened arrays of the n_bar's
        self.n_bar_flat = np.reshape(n_bar_matrix,(number_tracers,n_x*n_y*n_z))
        nr_flat =  np.reshape(self.nr,(number_tracers,n_x*n_y*n_z))

        # Power spectrum in (almost) physical units
        self.Pi = power0*((1./self.cell_size)**3) # initial guess for the power spectrum
        
        ######################
        # Multi-tracer weights
        ######################
        
        # Regular P_mu = b_mu^2 * P(k)  -- add small number since will divide stuff by P_mu
        # Dim P_mu : [nt]
        self.P_mu = small + np.power(bias,2.0)*self.Pi

        # curved P_mu = cP_mu = nbar_mu P_mu  --  individual clustering strengths
        # Dim cPmu: [nt,nx,ny,nz]
        self.cP_mu = ((n_bar_matrix.T)*self.P_mu).T
        # Dim cPmu_flat: [nt,nx*ny*nz]
        self.cP_mu_flat = ((self.n_bar_flat.T)*self.P_mu).T

        # curved P = Total clustering strength ==> sum over nt
        # Dim cPtot = [nx,ny,nz]
        self.cP_tot = np.sum(self.cP_mu,axis=0)
        # Dim cPtot_flat = [nx*ny*nz]
        self.cP_tot_flat = np.sum(self.cP_mu_flat,axis=0)

        # Weights for *** n(x) ***  ( nt x nt matrices, functions of x )
        # f1_mu = b_mu (n_mu - alpha * n_mu_ran)
        # f1_tot = \sum f1_mu
        # f_mu = f1_mu - cP_\mu/(1 + cP_tot) * f_tot
        # ==>  f_tot = \sum_\mu f_mu = 1/(1+cP_tot) f1_tot
        #
        # Notice that w1 and wt have different dimensions:
        # wt is an array [nt,nx,ny,nz]
        self.wt = (self.cP_mu)/(1.0 + self.cP_tot)
        # wt is an array [nt,nx*ny*nz]
        #self.wt_flat = (self.cP_mu_flat)/(1.0 + self.cP_tot_flat)

        # Estimator bias, \Delta Q : [nt]
        # self.DeltaQ_mu = self.norm_Q * (0.5*(1.0 + self.alpha)*np.sum( self.n_bar_flat/np.power(1.0 + self.cP_tot_flat,2) , axis=1) )
        self.DeltaQ_mu = 0.5*(1.0 + self.alpha)*np.sum( self.n_bar_flat/np.power(1.0 + self.cP_tot_flat,2) , axis=1)

        #Multipoles
        self.multipoles = multipoles #AQUI

        # Multi-tracer Fisher matrix
        # I1_mu : [nt]
        I1_mu = np.sum(self.cP_mu_flat*self.cP_tot_flat/(1.0 + self.cP_tot_flat),axis=1)
        
        # I2_munu: [nt,nt]  (I have not found a better way to do this yet...)
        I2_munu = np.zeros((number_tracers,number_tracers))
        for imu in range(number_tracers):
            I2_munu[imu] = np.sum( self.cP_mu_flat * self.cP_mu_flat[imu] * (1.0 - self.cP_tot_flat)/np.power(1.0+self.cP_tot_flat,2) , axis=1)
        
        # Fisher matrix F_munu: [nt,nt]
        self.F_munu = 0.25*(np.eye(number_tracers)*I1_mu/self.P_mu/self.P_mu + \
                            ((I2_munu/self.P_mu).T)/self.P_mu)

        # Inverse of the Fisher matrix
        # N.B.: this is the matrix that goes in the estimation of the P_mu(k),
        # but when we compute the theoretical cov(P_mu,P_nu), we will use the
        # P_mu that we actually estimate -- at the end of the estimation procedure.
        self.F_inv_munu = np.linalg.inv(self.F_munu)

        # Print out the shapes of everything we will use later...
        #print('F_munu :',np.shape(self.F_munu))
        #print('F_inv :',np.shape(self.F_inv_munu))
        #print('wt :',np.shape(self.wt))
        #print('wt_flat :',np.shape(self.wt_flat))
        #print('DeltaQ_mu :',np.shape(self.DeltaQ_mu))

        ##################### OLD STUFF -- REMOVE LATER, AFTER FINAL TESTS
        # Normalization N -- PVP ,Eq. 7:
        # N^2 = \int dV n^2 w^2 = \int dV (n b^2 P/(1 + n b^2 P))^2
        # self.N = np.sqrt(np.sum((n_bar_matrix**2)*(self.w**2)))
        
        # Other normalizations used for sigmas
        # self.N2 = np.sum((n_bar_matrix**4.)*(self.w**4.))               # \int dV n^4 w^4
        # self.N3 = np.sum((n_bar_matrix**3.)*(self.w**4.)/(bias**2.))    # \int dV n^3 w^4/b^2
        # self.N4 = np.sum((n_bar_matrix**2.)*(self.w**4.)/(bias**4.))    # \int dV n^2 w^4/b^4
        
        # Shot noise -- PVP, Eq. 16:
        # P_shot = (1+alpha)/N^2 * \int dV n w^2/b^2
        # self.Pshot = ((1+self.alpha)/(self.N**2 + small)) * np.sum(n_bar_matrix*((self.w**2)/(bias**2 + small)))
        

    def fkp(self,ng):
        '''
        This is the FKP function itself, only entry is the galaxy map
        '''
        # This was defined in .init, above
        # largenumber = 10000.0
        small = 1.0/self.largenumber
        # alpha = small
        
        # Recover the number of tracers (no need to pass it on function FKP)
        number_tracers = np.shape(ng)[0]
        
        # Weighted overdensity fields
        # Dim: w1[nt] , ng[nt,nx,ny,nz], nr[nt,nx,ny,nz]
        # First, F1:
        #self.F1_mu = (self.bias*((ng - self.alpha*self.nr).T)).T
        self.F1_mu = (self.bias*((ng).T)).T
        self.F1_mu_bar = (self.bias*(( - self.alpha*self.nr).T)).T
        #print('F1_mu :',np.shape(self.F1_mu))
        # Now sum over tracers to get F1tot:

        #self.F1_tot = np.sum(self.F1_mu,axis=0)
        self.F1_tot = np.sum(self.F1_mu,axis=0)
        self.F1_tot_bar = np.sum(self.F1_mu_bar,axis=0)
        #print('F1_tot :',np.shape(self.F1_tot))
        # Now obtain F_mu
        # self.F_mu = self.F1_mu - ( self.cP_mu/(1.0 + self.cP_tot))*(self.F1_tot)
        
        #self.F_mu = self.F1_mu - self.wt * self.F1_tot
        self.F_mu = self.F1_mu - self.wt * self.F1_tot
        self.F_mu_bar = self.F1_mu_bar - self.wt * self.F1_tot_bar
        Fmu = self.F_mu + self.F_mu_bar
        
        #self.F_tot = np.sum(self.F_mu,axis=0)
        self.F_tot = np.sum(self.F_mu,axis=0)
        self.F_tot_bar = np.sum(self.F_mu_bar,axis=0)
        
        # No need to F_tot now -- will FFT the F_mu, then sum them

        # OLD FKP STUFF -- keep it for checking later
        # self.F=(self.w/(self.N*self.bias + small)) * (ng - self.alpha*self.nr)
        # ALSO KEEP THIS OLD STUFF TO CHECK LATER
        # This mu is only in order to introduce a multipole "by hand"
        # mu = 1.0/(np.sqrt(3.0))*(self.kxhat+self.kyhat+self.kzhat)
        # mu = 1.0/(np.sqrt(2.0))*(0*self.kxhat+self.kyhat+self.kzhat)
        # should be this:
        # mu = self.kzhat
        # mu = np.abs(mu)
        # Fk=Fk*(1.0 + 2.0*self.quad*(-0.5+1.5*mu**2))/np.sqrt(1 + (2.0*self.quad)**2/20.0)
        
        # FFts: will need F_tot and F_mu only
        # Need to carry out one for each tracer, independently!

        #Multipoles
        multipoles = self.multipoles#AQUI
        
        #F_mu_k = np.zeros((self.n_x,self.n_y,self.n_z))
        # These are the flattened arrays for the multipoles of F_mu field:
        lenkf = self.n_x*self.n_y*(self.n_z//2 + 1)
        if multipoles == 0: #Monopole
            F0_mu_k_flat = (0.0+0.0j)*np.zeros((number_tracers,lenkf))

        elif multipoles == 2: #Dipole
            F0_mu_k_flat = (0.0+0.0j)*np.zeros((number_tracers,lenkf))
            F2_mu_k_flat = (0.0+0.0j)*np.zeros((number_tracers,lenkf))

        else:
            F0_mu_k_flat = (0.0+0.0j)*np.zeros((number_tracers,lenkf))
            F2_mu_k_flat = (0.0+0.0j)*np.zeros((number_tracers,lenkf))
            F4_mu_k_flat = (0.0+0.0j)*np.zeros((number_tracers,lenkf))
        
        for nt in range(number_tracers):
            #F_mu_k = np.fft.rfftn(self.F_mu[nt])
            F_mu_k = np.fft.rfftn(self.F_mu[nt]) / self.mas_wfunction
            F_mu_k_bar = np.fft.rfftn(self.F_mu_bar[nt]) #/ self.mas_wfunction

            if multipoles == 0:#AQUI
                # Monopole
                F0_mu_k_flat[nt] = np.ndarray.flatten( F_mu_k + F_mu_k_bar )

            if multipoles == 2:#AQUI
                # Monopole
                F0_mu_k_flat[nt] = np.ndarray.flatten( F_mu_k + F_mu_k_bar )

                # Compute the physical quadrupole
                F2mu = (self.kx_half**2)*(np.fft.rfftn(Fmu[nt]*(self.rxhat)*(self.rxhat)))
                F2mu+= (self.ky_half**2)*(np.fft.rfftn(Fmu[nt]*(self.ryhat)*(self.ryhat)))
                F2mu+= (self.kz_half**2)*(np.fft.rfftn(Fmu[nt]*(self.rzhat)*(self.rzhat)))
                F2mu+= (2.0*self.kx_half*self.ky_half)*(np.fft.rfftn(Fmu[nt]*(self.rxhat)*(self.ryhat)))
                F2mu+= (2.0*self.kx_half*self.kz_half)*(np.fft.rfftn(Fmu[nt]*(self.rxhat)*(self.rzhat)))
                F2mu+= (2.0*self.ky_half*self.kz_half)*(np.fft.rfftn(Fmu[nt]*(self.ryhat)*(self.rzhat)))

                F2mu = F2mu/self.mas_wfunction

                F2_mu_k_flat[nt] = - F0_mu_k_flat[nt] + 3.0*np.ndarray.flatten( F2mu )

            if multipoles == 4:#AQUI
                # Monopole
                F0_mu_k_flat[nt] = np.ndarray.flatten( F_mu_k + F_mu_k_bar )

                # Compute the physical quadrupole
                F2mu = (self.kx_half**2)*(np.fft.rfftn(Fmu[nt]*(self.rxhat)*(self.rxhat)))
                F2mu+= (self.ky_half**2)*(np.fft.rfftn(Fmu[nt]*(self.ryhat)*(self.ryhat)))
                F2mu+= (self.kz_half**2)*(np.fft.rfftn(Fmu[nt]*(self.rzhat)*(self.rzhat)))
                F2mu+= (2.0*self.kx_half*self.ky_half)*(np.fft.rfftn(Fmu[nt]*(self.rxhat)*(self.ryhat)))
                F2mu+= (2.0*self.kx_half*self.kz_half)*(np.fft.rfftn(Fmu[nt]*(self.rxhat)*(self.rzhat)))
                F2mu+= (2.0*self.ky_half*self.kz_half)*(np.fft.rfftn(Fmu[nt]*(self.ryhat)*(self.rzhat)))

                F2mu = F2mu/self.mas_wfunction

                # Compute the physical hexadecapole
                F4mu =  self.kx_half**4 *(np.fft.rfftn(Fmu[nt]*(self.rxhat)**4))
                F4mu += self.ky_half**4 *(np.fft.rfftn(Fmu[nt]*(self.ryhat)**4))
                F4mu += self.kz_half**4 *(np.fft.rfftn(Fmu[nt]*(self.rzhat)**4))

                F4mu += 4.0 * self.kx_half**3 * self.ky_half * (np.fft.rfftn(Fmu[nt]*(self.rxhat)**3*(self.ryhat)))
                F4mu += 4.0 * self.kx_half**3 * self.kz_half * (np.fft.rfftn(Fmu[nt]*(self.rxhat)**3*(self.rzhat)))
                F4mu += 4.0 * self.ky_half**3 * self.kx_half * (np.fft.rfftn(Fmu[nt]*(self.ryhat)**3*(self.rxhat)))
                F4mu += 4.0 * self.ky_half**3 * self.kz_half * (np.fft.rfftn(Fmu[nt]*(self.ryhat)**3*(self.rzhat)))
                F4mu += 4.0 * self.kz_half**3 * self.kx_half * (np.fft.rfftn(Fmu[nt]*(self.rzhat)**3*(self.rxhat)))
                F4mu += 4.0 * self.kz_half**3 * self.ky_half * (np.fft.rfftn(Fmu[nt]*(self.rzhat)**3*(self.ryhat)))

                F4mu += 6.0 * self.kx_half**2 * self.ky_half**2 * (np.fft.rfftn(Fmu[nt]*(self.rxhat)**2*(self.ryhat)**2))
                F4mu += 6.0 * self.kx_half**2 * self.kz_half**2 * (np.fft.rfftn(Fmu[nt]*(self.rzhat)**2*(self.rxhat)**2))
                F4mu += 6.0 * self.ky_half**2 * self.kz_half**2 * (np.fft.rfftn(Fmu[nt]*(self.ryhat)**2*(self.rzhat)**2))

                F4mu += 12.0 * self.kx_half**2 * self.ky_half * self.kz_half * (np.fft.rfftn(Fmu[nt]*(self.rxhat)**2*(self.ryhat)*(self.rzhat)))
                F4mu += 12.0 * self.kz_half**2 * self.kx_half * self.ky_half * (np.fft.rfftn(Fmu[nt]*(self.rzhat)**2*(self.rxhat)*(self.ryhat)))
                F4mu += 12.0 * self.ky_half**2 * self.kz_half * self.kx_half * (np.fft.rfftn(Fmu[nt]*(self.ryhat)**2*(self.rxhat)*(self.rzhat)))

                F4mu = F4mu / self.mas_wfunction

                F2_mu_k_flat[nt] = - F0_mu_k_flat[nt] + 3.0*np.ndarray.flatten( F2mu )
                F4_mu_k_flat[nt] = - 7.0/4.0*F0_mu_k_flat[nt] - 2.5*F2_mu_k_flat[nt] + 35.0/4.0*np.ndarray.flatten(F4mu)

        if multipoles == 0:#AQUI

            #????????? WTF??... Seems to need these lines... Some python crazy shit?...
            F0_mu_k_flat = F0_mu_k_flat

            # Sum over nt to obtain F_tot
            F0_tot_k_flat = np.sum(F0_mu_k_flat,axis=0)

            # Now combine to obtain monopole and quadrupole of F^2
            # Monopole: F_mu F_tot^* + c.c.
            # Dim: F_mu_k_flat [nt,lenkf] , F_tot_k_flat [lenkf]
            FF0_mu_k_flat = 2.0*(F0_mu_k_flat*(F0_tot_k_flat.conj())).real
        
            # Here are the bin counts
            counts = (self.bin_matrix).dot(np.ones(lenkf))
            self.counts = counts
            nbinsout = len(counts)

            # Get volume of the k-bins
            # Dim V_k: [nbinsout]
            Vk = self.counts/((self.n_x*self.n_y)*(self.n_z//2+1) + small)
            Vfft_to_Vk = 1.0/((self.n_x*self.n_y)*(self.n_z//2+1) + small)
            Cov_munu = self.F_inv_munu
            # Here it is convenient to use the covariance per unit volume
            Cov_ret = np.reshape( np.kron( Cov_munu , 1.0*(Vk**0 + small) ),(number_tracers,number_tracers,nbinsout) )

            # These are the intermediate estimators Q .
            # Must average Q_mu for each tracer over the bins independently
            # Here I actually compute Q_mu / Vk , since that is easier to compare with calculations
            Q0_mu_flat = np.zeros((number_tracers,nbinsout))
        
            Pshot_mu_flat = np.zeros((number_tracers,nbinsout))
            # Notice that DeltaQ_mu is already in physical units, so Q0 must also be.
            # Here I compute the <Q>_k , which is easier to compare with other stuff
            for nt in range(number_tracers):
                F0k2 = (self.bin_matrix).dot(FF0_mu_k_flat[nt])
                Q0_mu_flat[nt] = ((0.25/((self.bias[nt])**2))*F0k2 - 1.0*(self.DeltaQ_mu)[nt]*self.counts)/(self.counts + small)/Vfft_to_Vk
            
                Pshot_mu_flat[nt] = 1.0*(self.DeltaQ_mu)[nt]*self.counts/(self.counts + small)/Vfft_to_Vk

            # Now, FINALLY, obtain the true multi-tracer estimator for the spectra...
            # Correcting for units of physical volume
            P0_mu_ret = np.sum(Cov_ret*Q0_mu_flat,axis=1)*Vfft_to_Vk
        
            Pshot_mu_ret = np.sum(Cov_ret*Pshot_mu_flat,axis=1)*Vfft_to_Vk

            # Testing: print shot noise
            print("   Multi-tracer shot noise:" , np.mean(Pshot_mu_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z),axis=1))

            ###############################################################
            # Now calculate the theoretical covariances
            ###############################################################

            # The idea is that Cov(P_mu,P_nu) = (F(P_mu,P_nu))^(-1) .
            # We have F_inv_munu from above, an nt x nt matrix
            # computed using the fiducial biases and P(k). 
            # To compute the actual covariance, with 
            # the actual P0_mu_ret obtained above, would be
            # prohibitive, numerically. We use, for now, F_inv_munu.

            ###############################################################
            ######### CONTINUE HERE                    ####################
            ######### CONTINUE HERE                    ####################
            ######### CONTINUE HERE                    ####################
            ######### CONTINUE HERE                    ####################
            ###############################################################

            # Changing to physical units
            P0_mu_ret = P0_mu_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
        
            # Now we need the covariance for each bin -- including the bin volumes
            Cov_ret = np.reshape( np.kron( Cov_munu , 1.0/(Vk + small) ),(number_tracers,number_tracers,nbinsout) )
            Cov_ret = Cov_ret*((self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)**2)

            #F0k2_ret = F0k2_flat 
            #F2k2_ret = F2k2_flat

            Q0_mu_ret = Q0_mu_flat 
        
            # Finalize output
            self.P0_mu_ret = P0_mu_ret

            self.Q0_mu_ret = Q0_mu_ret

            #self.F0k2_ret = F0k2_ret
            #self.F2k2_ret = F2k2_ret
        
            self.Cov_ret = Cov_ret
            self.Cov_munu = Cov_munu


        elif multipoles == 2:#AQUI

            #????????? WTF??... Seems to need these lines... Some python crazy shit?...
            F0_mu_k_flat = F0_mu_k_flat
            F2_mu_k_flat = F2_mu_k_flat

            # Sum over nt to obtain F_tot
            F0_tot_k_flat = np.sum(F0_mu_k_flat,axis=0)
            F2_tot_k_flat = np.sum(F2_mu_k_flat,axis=0)

            # Now combine to obtain monopole and quadrupole of F^2
            # Monopole: F_mu F_tot^* + c.c.
            # Dim: F_mu_k_flat [nt,lenkf] , F_tot_k_flat [lenkf]
            FF0_mu_k_flat = 2.0*(F0_mu_k_flat*(F0_tot_k_flat.conj())).real
        
            # Quadrupole: 1/2 * ( F2_mu F0_tot^* + F0_mu F2_tot^* + c.c.)
            FF2_mu_k_flat = ( F2_mu_k_flat*(F0_tot_k_flat.conj()) + F0_mu_k_flat*(F2_tot_k_flat.conj()) ).real

            # Here are the bin counts
            counts = (self.bin_matrix).dot(np.ones(lenkf))
            self.counts = counts
            nbinsout = len(counts)

            # Get volume of the k-bins
            # Dim V_k: [nbinsout]
            Vk = self.counts/((self.n_x*self.n_y)*(self.n_z//2+1) + small)
            Vfft_to_Vk = 1.0/((self.n_x*self.n_y)*(self.n_z//2+1) + small)
            Cov_munu = self.F_inv_munu
            # Here it is convenient to use the covariance per unit volume
            Cov_ret = np.reshape( np.kron( Cov_munu , 1.0*(Vk**0 + small) ),(number_tracers,number_tracers,nbinsout) )

            # These are the intermediate estimators Q .
            # Must average Q_mu for each tracer over the bins independently
            # Here I actually compute Q_mu / Vk , since that is easier to compare with calculations
            Q0_mu_flat = np.zeros((number_tracers,nbinsout))
            Q2_mu_flat = np.zeros((number_tracers,nbinsout))
        
            Pshot_mu_flat = np.zeros((number_tracers,nbinsout))
            # Notice that DeltaQ_mu is already in physical units, so Q0 must also be.
            # Here I compute the <Q>_k , which is easier to compare with other stuff
            for nt in range(number_tracers):
                F0k2 = (self.bin_matrix).dot(FF0_mu_k_flat[nt])
                Q0_mu_flat[nt] = ((0.25/((self.bias[nt])**2))*F0k2 - 1.0*(self.DeltaQ_mu)[nt]*self.counts)/(self.counts + small)/Vfft_to_Vk
            
                F2k2 = ((self.bin_matrix).dot(FF2_mu_k_flat[nt]))
                Q2_mu_flat[nt] = (0.25/((self.bias[nt])**2))*F2k2/(self.counts + small)/Vfft_to_Vk
            
                Pshot_mu_flat[nt] = 1.0*(self.DeltaQ_mu)[nt]*self.counts/(self.counts + small)/Vfft_to_Vk

            # Now, FINALLY, obtain the true multi-tracer estimator for the spectra...
            # Correcting for units of physical volume
            P0_mu_ret = np.sum(Cov_ret*Q0_mu_flat,axis=1)*Vfft_to_Vk
            # Quadrupole with factor of 5/2
            P2_mu_ret = 2.5*np.sum(Cov_ret*Q2_mu_flat,axis=1)*Vfft_to_Vk
        
            Pshot_mu_ret = np.sum(Cov_ret*Pshot_mu_flat,axis=1)*Vfft_to_Vk

            # Testing: print shot noise
            print("   Multi-tracer shot noise:" , np.mean(Pshot_mu_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z),axis=1))

            ###############################################################
            # Now calculate the theoretical covariances
            ###############################################################

            # The idea is that Cov(P_mu,P_nu) = (F(P_mu,P_nu))^(-1) .
            # We have F_inv_munu from above, an nt x nt matrix
            # computed using the fiducial biases and P(k). 
            # To compute the actual covariance, with 
            # the actual P0_mu_ret obtained above, would be
            # prohibitive, numerically. We use, for now, F_inv_munu.

            ###############################################################
            ######### CONTINUE HERE                    ####################
            ######### CONTINUE HERE                    ####################
            ######### CONTINUE HERE                    ####################
            ######### CONTINUE HERE                    ####################
            ###############################################################

            # Changing to physical units
            P0_mu_ret = P0_mu_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
            P2_mu_ret = P2_mu_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
        
            # Now we need the covariance for each bin -- including the bin volumes
            Cov_ret = np.reshape( np.kron( Cov_munu , 1.0/(Vk + small) ),(number_tracers,number_tracers,nbinsout) )
            Cov_ret = Cov_ret*((self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)**2)

            #F0k2_ret = F0k2_flat 
            #F2k2_ret = F2k2_flat

            Q0_mu_ret = Q0_mu_flat 
            Q2_mu_ret = Q2_mu_flat
        
            # Finalize output
            self.P0_mu_ret = P0_mu_ret
            self.P2_mu_ret = P2_mu_ret

            self.Q0_mu_ret = Q0_mu_ret
            self.Q2_mu_ret = Q2_mu_ret

            #self.F0k2_ret = F0k2_ret
            #self.F2k2_ret = F2k2_ret
        
            self.Cov_ret = Cov_ret
            self.Cov_munu = Cov_munu

        else:

            #????????? WTF??... Seems to need these lines... Some python crazy shit?...
            F0_mu_k_flat = F0_mu_k_flat
            F2_mu_k_flat = F2_mu_k_flat
            F4_mu_k_flat = F4_mu_k_flat

            # Sum over nt to obtain F_tot
            F0_tot_k_flat = np.sum(F0_mu_k_flat,axis=0)
            F2_tot_k_flat = np.sum(F2_mu_k_flat,axis=0)
            F4_tot_k_flat = np.sum(F4_mu_k_flat,axis=0)

            # Now combine to obtain monopole and quadrupole of F^2
            # Monopole: F_mu F_tot^* + c.c.
            # Dim: F_mu_k_flat [nt,lenkf] , F_tot_k_flat [lenkf]
            FF0_mu_k_flat = 2.0*(F0_mu_k_flat*(F0_tot_k_flat.conj())).real
        
            # Quadrupole: 1/2 * ( F2_mu F0_tot^* + F0_mu F2_tot^* + c.c.)
            FF2_mu_k_flat = ( F2_mu_k_flat*(F0_tot_k_flat.conj()) + F0_mu_k_flat*(F2_tot_k_flat.conj()) ).real

            # Hexadecapole:
            FF4_mu_k_flat = ( F4_mu_k_flat*(F0_tot_k_flat.conj()) + F0_mu_k_flat*(F4_tot_k_flat.conj()) ).real

            # Here are the bin counts
            counts = (self.bin_matrix).dot(np.ones(lenkf))
            self.counts = counts
            nbinsout = len(counts)

            # Get volume of the k-bins
            # Dim V_k: [nbinsout]
            Vk = self.counts/((self.n_x*self.n_y)*(self.n_z//2+1) + small)
            Vfft_to_Vk = 1.0/((self.n_x*self.n_y)*(self.n_z//2+1) + small)
            Cov_munu = self.F_inv_munu
            # Here it is convenient to use the covariance per unit volume
            Cov_ret = np.reshape( np.kron( Cov_munu , 1.0*(Vk**0 + small) ),(number_tracers,number_tracers,nbinsout) )

            # These are the intermediate estimators Q .
            # Must average Q_mu for each tracer over the bins independently
            # Here I actually compute Q_mu / Vk , since that is easier to compare with calculations
            Q0_mu_flat = np.zeros((number_tracers,nbinsout))
            Q2_mu_flat = np.zeros((number_tracers,nbinsout))
            Q4_mu_flat = np.zeros((number_tracers,nbinsout))
        
            Pshot_mu_flat = np.zeros((number_tracers,nbinsout))
            # Notice that DeltaQ_mu is already in physical units, so Q0 must also be.
            # Here I compute the <Q>_k , which is easier to compare with other stuff
            for nt in range(number_tracers):
                F0k2 = (self.bin_matrix).dot(FF0_mu_k_flat[nt])
                Q0_mu_flat[nt] = ((0.25/((self.bias[nt])**2))*F0k2 - 1.0*(self.DeltaQ_mu)[nt]*self.counts)/(self.counts + small)/Vfft_to_Vk
            
                F2k2 = ((self.bin_matrix).dot(FF2_mu_k_flat[nt]))
                Q2_mu_flat[nt] = (0.25/((self.bias[nt])**2))*F2k2/(self.counts + small)/Vfft_to_Vk

                F4k2 = ((self.bin_matrix).dot(FF4_mu_k_flat[nt]))
                Q4_mu_flat[nt] = (0.25/((self.bias[nt])**2))*F4k2/(self.counts + small)/Vfft_to_Vk
            
                Pshot_mu_flat[nt] = 1.0*(self.DeltaQ_mu)[nt]*self.counts/(self.counts + small)/Vfft_to_Vk

            # Now, FINALLY, obtain the true multi-tracer estimator for the spectra...
            # Correcting for units of physical volume
            P0_mu_ret = np.sum(Cov_ret*Q0_mu_flat,axis=1)*Vfft_to_Vk
            # Quadrupole with factor of 5/2
            P2_mu_ret = 2.5*np.sum(Cov_ret*Q2_mu_flat,axis=1)*Vfft_to_Vk
            # Hexadecapole with factor of 9/2
            P4_mu_ret = 4.5*np.sum(Cov_ret*Q4_mu_flat,axis=1)*Vfft_to_Vk
        
            Pshot_mu_ret = np.sum(Cov_ret*Pshot_mu_flat,axis=1)*Vfft_to_Vk

            # Testing: print shot noise
            print("   Multi-tracer shot noise:" , np.mean(Pshot_mu_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z),axis=1))

            ###############################################################
            # Now calculate the theoretical covariances
            ###############################################################

            # The idea is that Cov(P_mu,P_nu) = (F(P_mu,P_nu))^(-1) .
            # We have F_inv_munu from above, an nt x nt matrix
            # computed using the fiducial biases and P(k). 
            # To compute the actual covariance, with 
            # the actual P0_mu_ret obtained above, would be
            # prohibitive, numerically. We use, for now, F_inv_munu.

            ###############################################################
            ######### CONTINUE HERE                    ####################
            ######### CONTINUE HERE                    ####################
            ######### CONTINUE HERE                    ####################
            ######### CONTINUE HERE                    ####################
            ###############################################################

            # Changing to physical units
            P0_mu_ret = P0_mu_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
            P2_mu_ret = P2_mu_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
            P4_mu_ret = P4_mu_ret*(self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)
        
            # Now we need the covariance for each bin -- including the bin volumes
            Cov_ret = np.reshape( np.kron( Cov_munu , 1.0/(Vk + small) ),(number_tracers,number_tracers,nbinsout) )
            Cov_ret = Cov_ret*((self.phsize_x/self.n_x)*(self.phsize_y/self.n_y)*(self.phsize_z/self.n_z)**2)

            #F0k2_ret = F0k2_flat 
            #F2k2_ret = F2k2_flat

            Q0_mu_ret = Q0_mu_flat 
            Q2_mu_ret = Q2_mu_flat
            Q4_mu_ret = Q4_mu_flat
        
            # Finalize output
            self.P0_mu_ret = P0_mu_ret
            self.P2_mu_ret = P2_mu_ret
            self.P4_mu_ret = P4_mu_ret

            self.Q0_mu_ret = Q0_mu_ret
            self.Q2_mu_ret = Q2_mu_ret
            self.Q4_mu_ret = Q4_mu_ret

            #self.F0k2_ret = F0k2_ret
            #self.F2k2_ret = F2k2_ret
        
            self.Cov_ret = Cov_ret
            self.Cov_munu = Cov_munu
