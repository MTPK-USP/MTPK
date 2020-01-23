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

class pkmg_cross(object):
    '''
    Only one generic class needed for both objects
    '''
    def __init__(self,biases,dipoles,matgrowrate,kphys,vel_disp,sigma_z,cH,z):

        self.biases = biases
        self.dipoles = dipoles
        self.matgrowrate = matgrowrate
        self.kphys = kphys
        self.vel_disp = vel_disp
        self.sigma_z = sigma_z
        self.cH = cH
        self.z = z

        f=matgrowrate

        ntracers = len(biases)
        nk = len(kphys)

        # This is to regularize the behaviour of the functions at k -> 0
        small = 0.005

        self.monos = np.zeros((ntracers*(ntracers-1)//2,nk))
        self.quads = np.zeros((ntracers*(ntracers-1)//2,nk))

        index=0
        for i in range(ntracers):
            for j in range(i+1,ntracers):
                b1 = biases[i]
                b2 = biases[j]

                # k * sigma_z
                self.KZ1 = small + cH * sigma_z[i] * kphys
                self.KZ2 = small + cH * sigma_z[j] * kphys
                self.KZ = np.sqrt(self.KZ1**2 + self.KZ2**2)

                self.ERF_KZ = np.sqrt(2.0*np.pi)*special.erf(self.KZ/np.sqrt(2.0))/(np.power(self.KZ,5))
                self.Exp_KZ = np.exp( -1.0 * np.power(self.KZ,2)/2.0 ) / (np.power(self.KZ,5))
                
                # Monopoles
                self.M1 = -2.0 * self.Exp_KZ * f * self.KZ * ( 3*f + self.KZ**2 *(b1 + b2 + f))
                self.M2 = self.ERF_KZ * ( 3.0*f**2 + (b1+b2)*f*self.KZ**2 + b1*b2*self.KZ**4 )
                self.monos[index] = 0.5*(self.M1 + self.M2)

                # Monopoles
                self.Q1 = -1.0 * self.Exp_KZ / self.KZ * ( 3*b1*b2*self.KZ**4 + (b1+b2)*f*(9+2*self.KZ**2)*self.KZ**2 + f**2 *(45 + 2*(6+self.KZ**2)*self.KZ**2) )
                self.Q2 = -0.5 * self.ERF_KZ * ( 3.0*f**2 * (-15 + self.KZ**2) + (b1+b2)*f*(-9+self.KZ**2)*self.KZ**2 + b1*b2*(-3+self.KZ**2)*self.KZ**4 ) / self.KZ**2
                self.quads[index] = 2.5*(self.Q1 + self.Q2)
                index += 1




