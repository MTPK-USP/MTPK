#!usr/bin/env python
# -*- coding: utf-8 -*-
'''
    This class contains the initial calculations for the 
    multipoles of the power spectrum (monopole and quadrupole only, so far).
    Created by Raul Abramo, 2019

    Entries: 
    biases (1 x Ntracers array, adimensional) 
    doppler dipoles (1 x Ntracers array, adimensional) 
    matter growth rate (float, adimensional)
    k_phys (physical wavenumbers, in units of h/Mpc)
    velocity dispersion (1 x Ntracers array, in units of c)
    redshift error (1 x Ntracers array, adimensional)
    c*H^-1 (Hubble radius at the relevant redshift)
    z (redshift of the slice)

'''
import numpy as np
#from scipy import interpolate
from scipy import special
import sys

class pkmg(object):
    '''
    Only one generic class needed for both objects
    '''
    #def __init__(self,biases,dipoles,matgrowrate,kphys,vel_disp,sigma_z,cH,z):
    def __init__(self,biases,dipoles,matgrowrate,kphys,sigma_z,cH,z):

        self.biases = biases
        self.dipoles = dipoles
        self.matgrowrate = matgrowrate
        self.kphys = kphys
        #self.vel_disp = vel_disp
        self.sigma_z = sigma_z
        self.cH = cH
        self.z = z

        # This is to regularize the behaviour of the functions at k -> 0
        small = 0.005

        # Effective "alpha"-dipoles 
        self.adip = matgrowrate * np.outer( dipoles , 1./(0.0000001 + kphys))

        # k * sigma_z
        self.KZ = small + cH * np.outer( sigma_z , kphys )

        # k * vel_disp
        #self.KV = 1.5*small + cH * np.outer( vel_disp , kphys )

        # Auxiliary definitions
        #self.S1 = np.sqrt( self.KZ**2 + self.KV**2 )
        #self.S2 = np.sqrt( self.KZ**2 + 2.0 * self.KV**2 )
        self.S1 = np.sqrt( self.KZ**2 )
        self.S2 = np.sqrt( self.KZ**2 )

        self.EPS1_KZ = np.sqrt(np.pi)*special.erf(self.KZ)/(2.0*self.KZ)
        self.EPS1_S1 = np.sqrt(np.pi)*special.erf(self.S1)/(2.0*self.S1)
        self.EPS1_S2 = np.sqrt(np.pi)*special.erf(self.S2)/(2.0*self.S2)

        self.e_KZ = np.exp(-self.KZ**2)
        self.e_S1 = np.exp(-self.S1**2)
        self.e_S2 = np.exp(-self.S2**2)
        
        # Monopole
        self.M1 = self.adip**2 * (self.EPS1_KZ - self.e_KZ)/(2.0 * self.KZ**2)
        self.M2 = (self.biases**2 *  (self.EPS1_KZ).T ).T
        self.M3 = (self.biases * self.matgrowrate * ( (self.EPS1_S1 - self.e_S1)/(self.S1**2) ).T).T
        self.M4 = self.matgrowrate**2 * ( 3./4.*(self.EPS1_S2 - self.e_S2)/(self.S2**4) - 1./2. * self.e_S2/(self.S2**2) )
        self.mono = self.M1 + self.M2 + self.M3 + self.M4

        # Quadrupole
        self.Q1a = self.adip**2 * ( 5./8. * (9. - 2*self.KZ**2) * (self.EPS1_KZ - self.e_KZ) / (self.KZ**4) )
        self.Q1b = self.adip**2 * ( - 5./8. * (6. * self.e_KZ) / (self.KZ**2) )
        self.Q1 = self.Q1a + self.Q1b

        #self.Q2a = (self.biases**2 * ( 5./4. * (3. - 2*self.KZ**2) * self.EPS1_KZ / (self.KZ**2) ).T).T
        #self.Q2b = (self.biases**2 * ( - 15./4. * (self.e_KZ) / (self.KZ**2) ).T).T
        #self.Q2 = self.Q2a + self.Q2b

        self.Q2a = (self.biases**2 * ( 5./4. * (3. - 2*self.KZ**2) * self.EPS1_KZ ).T).T
        self.Q2b = (self.biases**2 * ( - 15./4. * (self.e_KZ) ).T).T
        self.Q2 = (self.Q2a + self.Q2b) / (self.KZ**2)

        self.Q3a = (self.matgrowrate * self.biases * ( 5./4. * (9. - 2*self.S1**2) * (self.EPS1_S1 - self.e_S1) / (self.S1**4) ).T).T
        self.Q3b = (self.matgrowrate * self.biases * ( - 5./4. * (6. * self.e_S1) / (self.S1**2) ).T).T
        self.Q3 = self.Q3a + self.Q3b

        self.Q4a = self.matgrowrate**2 * 15./16. * (15. - 2*self.S2**2) * self.EPS1_S2 / (self.S2**6) 
        self.Q4b = self.matgrowrate**2 * ( - 5./16.) * (45. + 24*self.S2**2 + 8.*self.S2**4) * self.e_S2 / (self.S2**6)
        self.Q4 = self.Q4a + self.Q4b

        self.quad = self.Q1 + self.Q2 + self.Q3 + self.Q4




