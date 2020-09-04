#!usr/bin/env python
# -*- coding: utf-8 -*-
"""
	Class to calculate the the gaussian power spectrum
	Arthur E. da Mota Loureiro (IFUSP)
	07/11/2014
"""
import numpy as np
from scipy import interpolate
import sys

class gauss_pk(object):
	"""
	This class is a part of the pk_fkp.
	It takes camb's power spectrum and transforms it in a gaussian P(k)
	The initial values are:
	k_camb, Pk_camb, the grid in k-space, cell_size and the maximum scale in the grid
	"""
	def __init__(self,k_camb,Pk_camb,grid_k,cell_size,L_max):
            corr_highk = np.exp(-(k_camb/2.0)**4)
            Pk_camb_corr = corr_highk*Pk_camb
            # low-k convergence
            k_max = 15.0
            self.k_max = k_max
            k_step = np.sqrt(grid_k[0,0,1]**2 + grid_k[0,0,1]**2 + grid_k[0,0,1]**2)/4.
            k_min = k_step/4.
            # interpolate camb's Power Spectrum (with convergence factors)
            self.Pk_camb_interp = interpolate.InterpolatedUnivariateSpline(k_camb,Pk_camb_corr)

            k_r = np.arange(k_min,k_max,k_step)
            ###########################
            #	Finding Xi_camb(r)
            ###########################
            ###########################################
            # Integral limits for the r-space integral
            ###########################################
            r_max = 0.25*np.pi/k_min
            r_step = 0.5*np.pi/k_max
            r_k = 1.0*np.arange(r_step/2.,r_max,r_step)
            dk_r=np.diff(k_r)                                      # makes the diff between k and k + dk
            dk_r=np.append(dk_r,[0.0])
            krk = np.outer(k_r,r_k)
            self.krk = krk
            #################################
            # Calculates the sin(kr)/kr term
            #################################
            sinkr=np.sin(krk)/krk
            ########################
            # calculates dk*k^2*P(k)
            ########################
            dkkPk=dk_r*(k_r**2)*self.Pk_camb_interp(k_r)
            ############################
            # The first integral itself
            ############################
            integral=np.einsum('i,ij->j',dkkPk,sinkr)
            #############################
            # Camb's correlation function
            #############################
            self.corr_ln=np.power(2.0*np.pi*np.pi,-1.0)*integral #*np.sum(integrando,axis=0)
            ####################################
            # The gaussian correlation function
            ####################################
            corr_g = np.log(1. + self.corr_ln)
            self.corr_g = corr_g
            ########################
            #	Calculating P_g(k)
            ########################
            dr = np.diff(r_k)
            dr = np.append(dr,[0.0])
            rkr = np.outer(r_k,k_r)
            ####################################
            # sin(rk)/rk and dr*r^2*Xi_gauss(r)
            ####################################
            sinrk2 = np.sin(rkr)/rkr
            drCorr = dr*r_k*r_k*corr_g
            drCorrln = dr*r_k*r_k*self.corr_ln
            ##############################################################
            # The second integral, resulting in a Gaussian Power Spectrum
            ##############################################################
            integralCorr = np.einsum('j,ji->i',drCorr,sinrk2)
            integralCorrln = np.einsum('j,ji->i',drCorrln,sinrk2)
            Pk_gauss = 4.*np.pi*integralCorr  #np.sum(integrando2, axis=0)
            Pk_rec = 4.*np.pi*integralCorrln  #np.sum(integrando2, axis=0)
            #Pk_gauss[0] = Pk_camb[1]
            ##############################################
            # The values that can be returned by the class
            ##############################################
            self.Pk_gauss_interp = interpolate.UnivariateSpline(k_r,Pk_gauss)
            self.Pk_rec_interp = interpolate.UnivariateSpline(k_r,Pk_rec)
            self.k_r = k_r
            self.r_k = r_k



