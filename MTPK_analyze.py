#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
------------
Initial code started by Arthur E. da Mota Loureiro, 04/2015
Additonal changes by R. Abramo 07/2015, 02/2016
Added multi-tracer generalization and other changes, 03-12/2016
Added Halo Model generalization and other changes, 01/2018
------------
"""

#from __future__ import print_function
import numpy as np
import sys
import os
import uuid
from time import time , strftime
from scipy import interpolate
from scipy.optimize import leastsq
import h5py
import glob


######
# Plotting:
if sys.platform == "darwin":
    import pylab as pl
    from matplotlib import cm
else:
    import matplotlib
    from matplotlib import pylab, mlab, pyplot
    from matplotlib import cm
    from IPython.display import display
    from IPython.core.pylabtools import figsize, getfigs
    pl=pyplot
######


small = 1.e-6

# Add path to /inputs directory in order to load inputs
# Change as necessary according to your installation
this_dir = os.getcwd()
input_dir = this_dir + '/inputs'
sys.path.append(input_dir)


np.set_printoptions(suppress=True)





#####################################################
#####################################################
#####################################################
# Load specs for this run of the analyze code
# Change name as needed
#####################################################

from ANALYZE_Vipers_W1_mocks import *

# Define a kmax up to which we will compute chi-2
kmax=0.4

#####################################################
#####################################################
#####################################################







# Save results here:
dir_results = 'spectra/' + dir_results
if not os.path.exists(dir_results):
    os.makedirs(dir_results)
#####################################################


# Load properties of the maps underlying spectra
ncomb = len(dirs)
biases = np.zeros((ncomb,ntracers))
nbars = np.zeros((ncomb,ntracers))
for nc in range(ncomb):
    this_dir = 'spectra/' + dirs[nc]
    sub_dir = glob.glob(this_dir + '/*k=*')[0]
    filename = glob.glob(sub_dir + '/*_bias.dat')[0]
    biases[nc] = np.asarray(np.loadtxt(filename))
    filename = glob.glob(sub_dir + '/*nbar*')[0]
    nbars[nc] = np.asarray(np.loadtxt(filename))


# Load spectra
all_ks = []
all_spectra_fkp = []
all_spectra_mt = []
all_spectra_theory = []
for nc in range(ncomb):
    this_dir = 'spectra/' + dirs[nc]
    print "Dataset", nc, this_dir
    sub_dir = glob.glob(this_dir + '/*k=*')[0]
    # ks
    filename = glob.glob(sub_dir + '/*vec_k*')[0]
    this_data = np.loadtxt(filename)
    print "# of k's:        :", this_data.shape
    all_ks.append(this_data)
    # fkp
    filename = glob.glob(sub_dir + '/*_k_data_FKP.dat')[0]
    this_data = np.loadtxt(filename)
    print "# of P(k)/FKP    :", this_data.shape
    all_spectra_fkp.append(this_data)
    # mt
    filename = glob.glob(sub_dir + '/*_k_data_MT.dat')[0]
    this_data = np.loadtxt(filename)
    print "# of P(k)/MT     :", this_data.shape
    all_spectra_mt.append(this_data)
    # theory
    filename = glob.glob(sub_dir + '/*_theory.dat')[0]
    this_data = np.loadtxt(filename)
    print "# of P(k)/Theory :", this_data.shape
    all_spectra_theory.append(this_data)

all_ks=np.asarray(all_ks)
all_spectra_fkp=np.asarray(all_spectra_fkp)
all_spectra_mt=np.asarray(all_spectra_mt)
all_spectra_theory=np.asarray(all_spectra_theory)

kph=np.mean(all_ks,axis=0)

#Dimensions
nk=len(kph)
nmaps=all_spectra_fkp.shape[-1]

# stack P0 and P2
fkp = np.transpose(all_spectra_fkp,(1,0,2))
fkp = np.reshape(fkp,((2*ntracers,nk,ncomb*nmaps)))
P0_fkp = fkp[::2]
P2_fkp = fkp[1::2]

mt = np.transpose(all_spectra_mt,(1,0,2))
mt = np.reshape(mt,((2*ntracers,nk,ncomb*nmaps)))
P0_mt = mt[::2]
P2_mt = mt[1::2]

P0_th = np.mean(all_spectra_theory[:,:,::2],axis=0).T
P2_th = np.mean(all_spectra_theory[:,:,1::2],axis=0).T


# Now, these are the positions of the data (estimated in slightly different ways) and of the mocks in these arrays
pos_mocks = np.arange(P0_mt.shape[-1])
pos_data  = pos_mocks[::nmaps]
pos_mocks = np.delete(pos_mocks,pos_data)


# We will compare data with means of the mocks, estimate stuff, compute chi^2.
kstop = np.argsort(np.abs(kmax-kph))[0]

# The fiducial bias was computed using the multi-tracer technique
# Hence, the "theory" for FKP is slightly different from the MT theory
fkp_norms = np.mean( np.mean(P0_fkp[:,3:kstop,pos_data],2)/np.mean(P0_fkp[:,3:kstop,pos_mocks],2) , axis=1)
mt_norms  = np.mean( np.mean(P0_mt[:,3:kstop,pos_data],2)/np.mean(P0_mt[:,3:kstop,pos_mocks],2) , axis=1)
# The relative normalization that should be used for the FKP MOCKS is:
norms = fkp_norms/mt_norms

# Now, normalized the P0 and P2 mocks with this normalization 
P0_fkp[:,:,pos_mocks] = (norms*(P0_fkp[:,:,pos_mocks].T)).T
P2_fkp[:,:,pos_mocks] = (norms*(P2_fkp[:,:,pos_mocks].T)).T


# Now build "total spectrum" and ratios as spherical coordinates
nbar = np.mean(nbars,axis=0)
bias = np.mean(biases,axis=0)

cP0_mt  = (nbar * P0_mt.T).T
cP2_mt  = (nbar * P2_mt.T).T
cP0_fkp = (nbar * P0_fkp.T).T
cP2_fkp = (nbar * P2_fkp.T).T
cP0_th  = (nbar * P0_th.T).T
cP2_th  = (nbar * P2_th.T).T

# Will only build ratios of monopoles to monopoles, and ratios of quadrupoles to monopoles
Sph_00_mt  = np.zeros_like(cP0_mt)
Sph_20_mt  = np.zeros_like(cP0_mt)
Sph_00_fkp = np.zeros_like(cP0_fkp)
Sph_20_fkp = np.zeros_like(cP0_fkp)
Sph_00_th  = np.zeros_like(cP0_th)
Sph_20_th  = np.zeros_like(cP0_th)
Sph_20_mix  = np.zeros_like(cP0_mt)

for nt in range(ntracers):
    # For quadrupole, only P2_i/P0_i for *same* tracers
    Sph_20_mt[nt]  = P2_mt[nt]/(small + P0_mt[nt])
    Sph_20_fkp[nt] = P2_fkp[nt]/(small + P0_fkp[nt])
    Sph_20_th[nt]  = P2_th[nt]/(small + P0_th[nt])
    Sph_20_mix[nt]  = P2_fkp[nt]/(small + P0_mt[nt])    
    # For monopoles, use "spherical" coordinates
    # nt=0: radial coordinate
    if nt==0:
        Sph_00_mt[nt]  = np.sum(cP0_mt,axis=0)
        Sph_00_fkp[nt] = np.sum(cP0_fkp,axis=0)
        Sph_00_th[nt]  = np.sum(cP0_th,axis=0)
    else:
        Rho_mt  = np.sum(cP0_mt[nt-1:],axis=0)
        Rho_fkp = np.sum(cP0_fkp[nt-1:],axis=0)
        Rho_th  = np.sum(cP0_th[nt-1:],axis=0)
        Sph_00_mt[nt]  = cP0_mt[nt-1]/(small + Rho_mt)
        Sph_00_fkp[nt] = cP0_fkp[nt-1]/(small + Rho_fkp)
        Sph_00_th[nt]  = cP0_th[nt-1]/(small + Rho_th)

# Data estimated with different techniques
Data_P0_mt  = np.mean(P0_mt[:,:,pos_data],axis=2)
Data_P0_fkp = np.mean(P0_fkp[:,:,pos_data],axis=2)

Data_P2_mt  = np.mean(P2_mt[:,:,pos_data],axis=2)
Data_P2_fkp = np.mean(P2_fkp[:,:,pos_data],axis=2)

Data_Sph_00_mt  = np.mean(Sph_00_mt[:,:,pos_data],axis=2)
Data_Sph_00_fkp = np.mean(Sph_00_fkp[:,:,pos_data],axis=2)

Data_Sph_20_mt  = np.mean(Sph_20_mt[:,:,pos_data],axis=2)
Data_Sph_20_fkp = np.mean(Sph_20_fkp[:,:,pos_data],axis=2)
Data_Sph_20_mix = np.mean(Sph_20_mix[:,:,pos_data],axis=2)


Mocks_P0_mt  = P0_mt[:,:,pos_mocks]
Mocks_P0_fkp = P0_fkp[:,:,pos_mocks]

Mocks_P2_mt  = P2_mt[:,:,pos_mocks]
Mocks_P2_fkp = P2_fkp[:,:,pos_mocks]

Mocks_Sph_00_mt  = Sph_00_mt[:,:,pos_mocks]
Mocks_Sph_00_fkp = Sph_00_fkp[:,:,pos_mocks]

Mocks_Sph_20_mt  = Sph_20_mt[:,:,pos_mocks]
Mocks_Sph_20_fkp = Sph_20_fkp[:,:,pos_mocks]
Mocks_Sph_20_mix = Sph_20_mix[:,:,pos_mocks]


# Means of the mocks == theory

Mean_P0_mt  = np.mean(Mocks_P0_mt,axis=2)
Mean_P0_fkp = np.mean(Mocks_P0_fkp,axis=2)

Mean_P2_mt  = np.mean(Mocks_P2_mt,axis=2)
Mean_P2_fkp = np.mean(Mocks_P2_fkp,axis=2)


Mean_Sph_00_mt  = np.mean(Mocks_Sph_00_mt,axis=2)
Mean_Sph_00_fkp = np.mean(Mocks_Sph_00_fkp,axis=2)

Mean_Sph_20_mt  = np.mean(Mocks_Sph_20_mt,axis=2)
Mean_Sph_20_fkp = np.mean(Mocks_Sph_20_fkp,axis=2)
Mean_Sph_20_mix = np.mean(Mocks_Sph_20_mix,axis=2)



# Covariances (including data)
monos_mt = np.reshape(P0_mt,((ntracers*nk,ncomb*nmaps)))
monos_fkp = np.reshape(P0_fkp,((ntracers*nk,ncomb*nmaps)))

quads_mt = np.reshape(P2_mt,((ntracers*nk,ncomb*nmaps)))
quads_fkp = np.reshape(P2_fkp,((ntracers*nk,ncomb*nmaps)))

Cov_P0_mt  = np.cov(monos_mt)
Cov_P0_fkp = np.cov(monos_fkp)

Cov_P2_mt  = np.cov(quads_mt)
Cov_P2_fkp = np.cov(quads_fkp)

Cov_P20_mt  = np.cov(np.vstack((monos_mt,quads_mt)))
Cov_P20_fkp  = np.cov(np.vstack((monos_fkp,quads_fkp)))
Cov_P20_mix  = np.cov(np.vstack((monos_mt,quads_fkp)))


# Relative covariances
rCov_P0_mt  = Cov_P0_mt/np.outer(Mean_P0_mt.flatten(),Mean_P0_mt.flatten())
rCov_P0_fkp = Cov_P0_fkp/np.outer(Mean_P0_fkp.flatten(),Mean_P0_fkp.flatten())

rCov_P2_mt  = Cov_P2_mt/np.outer(Mean_P2_mt.flatten(),Mean_P2_mt.flatten())
rCov_P2_fkp = Cov_P2_fkp/np.outer(Mean_P2_fkp.flatten(),Mean_P2_fkp.flatten())

mq_mt =np.vstack((Mean_P0_mt,Mean_P2_mt)).flatten()
mq_fkp=np.vstack((Mean_P0_fkp,Mean_P2_fkp)).flatten()
mq_mix=np.vstack((Mean_P0_mt,Mean_P2_fkp)).flatten()

rCov_P20_mt  = Cov_P20_mt/np.outer(mq_mt,mq_mt)
rCov_P20_fkp = Cov_P20_fkp/np.outer(mq_fkp,mq_fkp)
rCov_P20_mix = Cov_P20_mix/np.outer(mq_mix,mq_mix)


# Spherical coordinates
Cov_Sph_00_mt  = np.cov(np.reshape(Sph_00_mt,((ntracers*nk,ncomb*nmaps))))
Cov_Sph_00_fkp = np.cov(np.reshape(Sph_00_fkp,((ntracers*nk,ncomb*nmaps))))

Cov_Sph_20_mt  = np.cov(np.reshape(Sph_20_mt,((ntracers*nk,ncomb*nmaps))))
Cov_Sph_20_fkp = np.cov(np.reshape(Sph_20_fkp,((ntracers*nk,ncomb*nmaps))))
Cov_Sph_20_mix = np.cov(np.reshape(Sph_20_mix,((ntracers*nk,ncomb*nmaps))))


rCov_Sph_00_mt  = Cov_Sph_00_mt/np.outer(Mean_Sph_00_mt.flatten(),Mean_Sph_00_mt.flatten())
rCov_Sph_00_fkp = Cov_Sph_00_fkp/np.outer(Mean_Sph_00_fkp.flatten(),Mean_Sph_00_fkp.flatten())

rCov_Sph_20_mt  = Cov_Sph_20_mt/np.outer(Mean_Sph_20_mt.flatten(),Mean_Sph_20_mt.flatten())
rCov_Sph_20_fkp = Cov_Sph_20_fkp/np.outer(Mean_Sph_20_fkp.flatten(),Mean_Sph_20_fkp.flatten())
rCov_Sph_20_mix = Cov_Sph_20_mix/np.outer(Mean_Sph_20_mix.flatten(),Mean_Sph_20_mix.flatten())



# SQRT of Diagonals of covariances
sigma_P0_mt  = np.reshape(np.sqrt(np.diag(Cov_P0_mt)),((ntracers,nk)))
sigma_P0_fkp = np.reshape(np.sqrt(np.diag(Cov_P0_fkp)),((ntracers,nk)))

sigma_P2_mt  = np.reshape(np.sqrt(np.diag(Cov_P2_mt)),((ntracers,nk)))
sigma_P2_fkp = np.reshape(np.sqrt(np.diag(Cov_P2_fkp)),((ntracers,nk)))

sigma_P20_mt  = np.reshape(np.sqrt(np.diag(Cov_P20_mt)),((2*ntracers,nk)))
sigma_P20_fkp = np.reshape(np.sqrt(np.diag(Cov_P20_fkp)),((2*ntracers,nk)))
sigma_P20_mix = np.reshape(np.sqrt(np.diag(Cov_P20_mix)),((2*ntracers,nk)))


sigma_Sph_00_mt  = np.reshape(np.sqrt(np.diag(Cov_Sph_00_mt)),((ntracers,nk)))
sigma_Sph_00_fkp = np.reshape(np.sqrt(np.diag(Cov_Sph_00_fkp)),((ntracers,nk)))


sigma_Sph_00_mt  = np.reshape(np.sqrt(np.diag(Cov_Sph_00_mt)),((ntracers,nk)))
sigma_Sph_00_fkp = np.reshape(np.sqrt(np.diag(Cov_Sph_00_fkp)),((ntracers,nk)))

sigma_Sph_20_mt  = np.reshape(np.sqrt(np.diag(Cov_Sph_20_mt)),((ntracers,nk)))
sigma_Sph_20_fkp = np.reshape(np.sqrt(np.diag(Cov_Sph_20_fkp)),((ntracers,nk)))
sigma_Sph_20_mix = np.reshape(np.sqrt(np.diag(Cov_Sph_20_mix)),((ntracers,nk)))


# Correlation matrices
Corr_P0_mt  = Cov_P0_mt / np.outer(sigma_P0_mt.flatten(),sigma_P0_mt.flatten())
Corr_P0_fkp = Cov_P0_fkp/ np.outer(sigma_P0_fkp.flatten(),sigma_P0_fkp.flatten())

Corr_P2_mt  = Cov_P2_mt / np.outer(sigma_P2_mt.flatten(),sigma_P2_mt.flatten())
Corr_P2_fkp = Cov_P2_fkp/ np.outer(sigma_P2_fkp.flatten(),sigma_P2_fkp.flatten())

Corr_P20_mt  = Cov_P20_mt / np.outer(sigma_P20_mt.flatten(),sigma_P20_mt.flatten())
Corr_P20_fkp = Cov_P20_fkp/ np.outer(sigma_P20_fkp.flatten(),sigma_P20_fkp.flatten())
Corr_P20_mix = Cov_P20_mix/ np.outer(sigma_P20_mix.flatten(),sigma_P20_mix.flatten())


Corr_Sph_00_mt  = Cov_Sph_00_mt / np.outer(sigma_Sph_00_mt.flatten(),sigma_Sph_00_mt.flatten())
Corr_Sph_00_fkp = Cov_Sph_00_fkp/ np.outer(sigma_Sph_00_fkp.flatten(),sigma_Sph_00_fkp.flatten())

Corr_Sph_20_mt  = Cov_Sph_20_mt / np.outer(sigma_Sph_20_mt.flatten(),sigma_Sph_20_mt.flatten())
Corr_Sph_20_fkp = Cov_Sph_20_fkp/ np.outer(sigma_Sph_20_fkp.flatten(),sigma_Sph_20_fkp.flatten())
Corr_Sph_20_mix = Cov_Sph_20_mix/ np.outer(sigma_Sph_20_mix.flatten(),sigma_Sph_20_mix.flatten())


# plot corr. matrices
nameindex = ntracers*[str(0.001*np.round(1000*k)) for k in kph[np.int(nk/8.):-np.int(nk/10.):np.int(nk/4.)]]
indexcov = np.arange(nk/8., ntracers*nk - nk/10., nk/4.)
onesk = np.diag(np.ones((ntracers*nk)))

nameindex2 = 2*ntracers*[str(0.001*np.round(1000*k)) for k in kph[np.int(nk/8.):-np.int(nk/10.):np.int(nk/4.)]]
indexcov2 = np.arange(nk/8., 2*ntracers*nk - nk/10., nk/4.)
onesk2 = np.diag(np.ones((2*ntracers*nk)))


fullcov = np.tril(Corr_P0_fkp) + np.triu(Corr_P0_mt) - onesk
pl.imshow(fullcov,origin='lower',interpolation='none')
pl.title(r'Correlation matrix, ${P}^{(0)}_i$')
pl.xticks(indexcov,nameindex,size=6,name='monospace',rotation=45)
pl.yticks(indexcov,nameindex,size=8,name='monospace')
pl.annotate('FKP',(np.int(len(kph)/5.),2*len(kph)),fontsize=20)
pl.annotate('Multi-tracer',(2*len(kph),np.int(len(kph)/5.)),fontsize=20)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.colorbar()
figname= dir_results + '/P0_corr.pdf'
pl.savefig(figname)
pl.close('all')

fullcov = np.tril(Corr_P2_fkp) + np.triu(Corr_P2_mt) - onesk
pl.imshow(fullcov,origin='lower',interpolation='none')
pl.title(r'Correlation matrix, ${P}^{(2)}_i$')
pl.xticks(indexcov,nameindex,size=6,name='monospace',rotation=45)
pl.yticks(indexcov,nameindex,size=8,name='monospace')
pl.annotate('FKP',(np.int(len(kph)/5.),2*len(kph)),fontsize=20)
pl.annotate('Multi-tracer',(2*len(kph),np.int(len(kph)/5.)),fontsize=20)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.colorbar()
figname= dir_results + '/P2_corr.pdf'
pl.savefig(figname)
pl.close('all')

fullcov = np.tril(Corr_P20_fkp) + np.triu(Corr_P20_mix) - onesk2
pl.imshow(fullcov,origin='lower',interpolation='none')
pl.title(r'Correlation matrix, $ \{ {P}^{(0)}_i,{P}^{(2)}_i \}$')
pl.xticks(indexcov2,nameindex2,size=6,name='monospace',rotation=45)
pl.yticks(indexcov2,nameindex2,size=8,name='monospace')
pl.annotate('FKP',(np.int(len(kph)/5.),2*len(kph)),fontsize=20)
pl.annotate('MT (M) , FKP (Q)',(2*len(kph),np.int(len(kph)/5.)),fontsize=20)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.colorbar()
figname= dir_results + '/P20_mix_corr.pdf'
pl.savefig(figname)
pl.close('all')


fullcov = np.tril(Corr_Sph_00_fkp) + np.triu(Corr_Sph_00_mt) - onesk
pl.imshow(fullcov,origin='lower',interpolation='none')
pl.title(r'Correlation matrix, ${\cal{P}}^{(0)}$ and ratios')
pl.xticks(indexcov,nameindex,size=6,name='monospace',rotation=45)
pl.yticks(indexcov,nameindex,size=8,name='monospace')
pl.annotate('FKP',(np.int(len(kph)/5.),2*len(kph)),fontsize=20)
pl.annotate('Multi-tracer',(2*len(kph),np.int(len(kph)/5.)),fontsize=20)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.colorbar()
figname= dir_results + '/Sph_00_corr.pdf'
pl.savefig(figname)
pl.close('all')

fullcov = np.tril(Corr_Sph_20_fkp) + np.triu(Corr_Sph_20_mt) - onesk
pl.imshow(fullcov,origin='lower',interpolation='none')
pl.title(r'Correlation matrix, $P^{(2)}_i/P^{(0)}_i$')
pl.xticks(indexcov,nameindex,size=6,name='monospace',rotation=45)
pl.yticks(indexcov,nameindex,size=8,name='monospace')
pl.annotate('FKP',(np.int(len(kph)/5.),2*len(kph)),fontsize=20)
pl.annotate('Multi-tracer',(2*len(kph),np.int(len(kph)/5.)),fontsize=20)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.colorbar()
figname= dir_results + '/Sph_20_corr.pdf'
pl.savefig(figname)
pl.close('all')

fullcov = np.tril(Corr_Sph_20_fkp) + np.triu(Corr_Sph_20_mix) - onesk
pl.imshow(fullcov,origin='lower',interpolation='none')
pl.title(r'Correlation matrix, $P^{(2)}_i/P^{(0)}_i$')
pl.xticks(indexcov,nameindex,size=6,name='monospace',rotation=45)
pl.yticks(indexcov,nameindex,size=8,name='monospace')
pl.annotate('FKP',(np.int(len(kph)/5.),2*len(kph)),fontsize=20)
pl.annotate('Mix',(2*len(kph),np.int(len(kph)/5.)),fontsize=20)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.colorbar()
figname= dir_results + '/Sph_20_mix_corr.pdf'
pl.savefig(figname)
pl.close('all')


# plot RELATIVE covariance matrices
fullcov = np.tril(rCov_P0_fkp) + np.triu(rCov_P0_mt)
pl.imshow(fullcov,origin='lower',interpolation='none')
pl.title(r'Relative covariance matrix of  $P^{(0)}_i$')
pl.xticks(indexcov,nameindex,size=6,name='monospace',rotation=45)
pl.yticks(indexcov,nameindex,size=8,name='monospace')
pl.annotate('FKP',(np.int(len(kph)/5.),2*len(kph)),fontsize=20)
pl.annotate('Multi-tracer',(2*len(kph),np.int(len(kph)/5.)),fontsize=20)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.colorbar()
figname= dir_results + '/P0_covariances.pdf'
pl.savefig(figname)
pl.close('all')

fullcov = np.tril(rCov_P2_fkp) + np.triu(rCov_P2_mt)
pl.imshow(fullcov,origin='lower',interpolation='none')
pl.title(r'Relative covariance matrix of  $P^{(2)}_i$')
pl.xticks(indexcov,nameindex,size=6,name='monospace',rotation=45)
pl.yticks(indexcov,nameindex,size=8,name='monospace')
pl.annotate('FKP',(np.int(len(kph)/5.),2*len(kph)),fontsize=20)
pl.annotate('Multi-tracer',(2*len(kph),np.int(len(kph)/5.)),fontsize=20)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.colorbar()
figname= dir_results + '/P2_covariances.pdf'
pl.savefig(figname)
pl.close('all')

fullcov = np.tril(rCov_P20_fkp) + np.triu(rCov_P20_mix)
pl.imshow(fullcov,origin='lower',interpolation='none')
pl.title(r'Relative covariance matrix of  $\{ P^{(0)}_i , P^{(2)}_i \} $')
pl.xticks(indexcov2,nameindex2,size=6,name='monospace',rotation=45)
pl.yticks(indexcov2,nameindex2,size=8,name='monospace')
pl.annotate('FKP',(np.int(len(kph)/5.),2*len(kph)),fontsize=20)
pl.annotate('MT (M) , FKP (Q)',(2*len(kph),np.int(len(kph)/5.)),fontsize=20)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.colorbar()
figname= dir_results + '/P20_covariances.pdf'
pl.savefig(figname)
pl.close('all')


fullcov = np.tril(rCov_Sph_00_fkp) + np.triu(rCov_Sph_00_mt)
pl.imshow(fullcov,origin='lower',interpolation='none')
pl.title(r'Relative covariance matrix of  ${\cal{P}}^{(0)}$ and ratios')
pl.xticks(indexcov,nameindex,size=6,name='monospace',rotation=45)
pl.yticks(indexcov,nameindex,size=8,name='monospace')
pl.annotate('FKP',(np.int(len(kph)/5.),2*len(kph)),fontsize=20)
pl.annotate('Multi-tracer',(2*len(kph),np.int(len(kph)/5.)),fontsize=20)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.colorbar()
figname= dir_results + '/Sph_00_covariances.pdf'
pl.savefig(figname)
pl.close('all')

fullcov = np.tril(rCov_Sph_20_fkp) + np.triu(rCov_Sph_20_mt)
pl.imshow(fullcov,origin='lower',interpolation='none')
pl.title(r'Relative covariance matrix, $P^{(2)}_i/P^{(0)}_i$')
pl.xticks(indexcov,nameindex,size=6,name='monospace',rotation=45)
pl.yticks(indexcov,nameindex,size=8,name='monospace')
pl.annotate('FKP',(np.int(len(kph)/5.),2*len(kph)),fontsize=20)
pl.annotate('Multi-tracer',(2*len(kph),np.int(len(kph)/5.)),fontsize=20)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=10)
pl.colorbar()
figname= dir_results + '/Sph_20_covariances.pdf'
pl.savefig(figname)
pl.close('all')



# Compare errors (diagonal)
pl.rcParams["axes.titlesize"] = 8
cm_subsection = np.linspace(0, 0.8, ntracers)
mycolor = [ cm.jet(x) for x in cm_subsection ]

p0_labels = [r'${P}^{(0)}_1$',r'${P}^{(0)}_2$',r'${P}^{(0)}_3$',r'${P}^{(0)}_4$']
p2_labels = [r'${P}^{(2)}_1$',r'${P}^{(2)}_2$',r'${P}^{(2)}_3$',r'${P}^{(2)}_4$']

y_labels = [r'${\cal{P}}_{Tot}$',r'$\arccos^2 (\theta_1)$',r'$\arccos^2 (\theta_2)$',r'$\arccos^2 (\theta_3)$']
p20_labels = [r'${\cal{P}}^{(2)}_1/{\cal{P}}^{(0)}_1$',r'${\cal{P}}^{(2)}_2/{\cal{P}}^{(0)}_2$',r'${\cal{P}}^{(2)}_3/{\cal{P}}^{(0)}_3$',r'${\cal{P}}^{(2)}_4/{\cal{P}}^{(0)}_4$']


for nt in range(ntracers):
    dd=sigma_P0_fkp[nt]/Mean_P0_fkp[nt]
    pl.semilogy(kph,dd,color=mycolor[nt],linestyle=(0, (1, 2)),linewidth=0.8)
    dd=sigma_P0_mt[nt]/Mean_P0_mt[nt]
    pl.semilogy(kph,dd,color=mycolor[nt],linewidth=1.5,label=p0_labels[nt])

pl.legend(loc='lower right', shadow=None, fontsize='small')
pl.xlim([0.95*kph[0],1.01*kph[-1]])
pl.ylim([0.01,10])
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$\sigma(P^{(0)}_i)/P^{(0)}_i$',fontsize=12)
pl.title(r'Relative variances of $P^{(0)}_i$',fontsize=16)
pl.savefig(dir_results + '/' + 'sigmas_P0.pdf')
pl.close('all')



for nt in range(ntracers):
    dd=sigma_P2_fkp[nt]/Mean_P2_fkp[nt]
    pl.semilogy(kph,dd,color=mycolor[nt],linestyle=(0, (1, 2)),linewidth=0.8)
    dd=sigma_P2_mt[nt]/Mean_P2_mt[nt]
    pl.semilogy(kph,dd,color=mycolor[nt],linewidth=1.5,label=p2_labels[nt])

pl.legend(loc='lower right', shadow=None, fontsize='small')
pl.xlim([0.95*kph[0],1.01*kph[-1]])
pl.ylim([0.01,10])
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$\sigma(P^{(0)}_i)/P^{(2)}_i$',fontsize=12)
pl.title(r'Relative variances of $P^{(2)}_i$',fontsize=16)
pl.savefig(dir_results + '/' + 'sigmas_P2.pdf')
pl.close('all')




for nt in range(ntracers):
    dd=sigma_Sph_00_fkp[nt]/Mean_Sph_00_fkp[nt]
    pl.semilogy(kph,dd,color=mycolor[nt],linestyle=(0, (1, 2)),linewidth=0.8,label=y_labels[nt])
    dd=sigma_Sph_00_mt[nt]/Mean_Sph_00_mt[nt]
    pl.semilogy(kph,dd,color=mycolor[nt],linewidth=1.5)

pl.legend(loc='lower right', shadow=None, fontsize='small')
pl.xlim([0.95*kph[0],1.01*kph[-1]])
pl.ylim([0.01,10])
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$\sigma(Y_\mu)/Y_\mu$',fontsize=12)
pl.title(r'Relative variances of ${\cal{P}}_{Tot}$ and angle variables',fontsize=16)
pl.savefig(dir_results + '/' + 'sigmas_00.pdf')
pl.close('all')


for nt in range(ntracers):
    dd=sigma_Sph_20_fkp[nt]/Mean_Sph_20_fkp[nt]
    pl.semilogy(kph,np.abs(dd),color=mycolor[nt],linestyle=(0, (1, 2)),linewidth=0.5)
    dd=sigma_Sph_20_mt[nt]/Mean_Sph_20_mt[nt]
    pl.semilogy(kph,np.abs(dd),color=mycolor[nt],linewidth=1.5,label=p20_labels[nt])
    dd=sigma_Sph_20_mix[nt]/Mean_Sph_20_mix[nt]
    pl.semilogy(kph,np.abs(dd),color=mycolor[nt],linestyle=(0, (2, 2)),linewidth=0.5)
pl.legend(loc='lower right', shadow=None, fontsize='small')
pl.xlim([0.95*kph[0],1.01*kph[-1]])
pl.ylim([0.01,10])
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$\sigma(Y_\mu)/Y_\mu$',fontsize=12)
pl.title(r'Relative variances of ${\cal{P}}^{(2)}_i/{\cal{P}}^{(0)}_i$',fontsize=16)
pl.savefig(dir_results + '/' + 'sigmas_20.pdf')
pl.close('all')




print 
print "P_tot and angular variables: FKP v. MT"
# First, for FKP
full_chi2 = np.sum(np.power((Data_Sph_00_fkp - Mean_Sph_00_fkp)/sigma_Sph_00_fkp,2.0),axis=1)/len(kph)
rest_chi2 = np.sum(np.power((Data_Sph_00_fkp - Mean_Sph_00_fkp)/sigma_Sph_00_fkp,2.0)[:,:kstop],axis=1)/len(kph[:kstop])
print "FKP -- Chi^2, full range of k:", np.around(full_chi2,4)
print "FKP -- Chi^2,  k<", str(kmax), ":", np.around(rest_chi2,4)
fig, axs = pl.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0)
kpk_data = np.power(kph,1.5)*Data_Sph_00_fkp[0]
kpk_mean = np.power(kph,1.5)*Mean_Sph_00_fkp[0]
# Errors multiplied by 2 for visualization purposes
axs[0].errorbar(kph,kpk_mean,np.power(kph,1.5)*2*sigma_Sph_00_fkp[0],color=mycolor[0],linestyle=(0, (1, 2)),linewidth=0.8)
axs[0].plot(kph,kpk_data,color=mycolor[0],marker="x",linestyle='None',markersize=4.0)
axs[0].set_ylabel(r'${\cal{P}}_{Tot}$',fontsize=12)
axs[0].set_ylim([0.7*np.min(kpk_mean),1.3*np.max(kpk_mean)])
axs[0].set_title(r'$k^{1.5} \times {\cal{P}}_{Tot}$ & angle var., FKP',fontsize=16)
axs[0].axvline(x=kmax, c='k',linewidth=0.3)
#axs[0].annotate(r'${\cal{P}}_{Tot}$: $\chi^2$/dof, all $k$:' + str(np.around(full_chi2[0],3)),xy=(80.0,370.0),xycoords="figure pixels",fontsize=8)
axs[0].annotate(r'$\chi^2$/dof, $k<k_{max}$:' + str(np.around(rest_chi2[0],3)),xy=(80.0,370.0),xycoords="figure pixels",fontsize=8)
for nt in range(1,ntracers):
    # Shift slightly the k's for each tracer, for better visualization
    axs[1].errorbar( (1+0.01*(nt-1))*kph , Mean_Sph_00_fkp[nt], 2*sigma_Sph_00_fkp[nt],color=mycolor[nt],linestyle=(0, (1, 2)),linewidth=0.8)
    axs[1].plot( (1+0.01*(nt-1))*kph , Data_Sph_00_fkp[nt],color=mycolor[nt],marker="x",linestyle='None', markersize=4.0)
    ypos=190-50*(nt-1)
    #axs[1].annotate(r'$\chi^2$/dof, all $k$:' + str(np.around(full_chi2[nt],3)),xy=(80,ypos),xycoords="figure pixels",fontsize=8)
    #axs[1].annotate(r'$\chi^2$/dof, $k<k_{max}$:' + str(np.around(rest_chi2[nt],3)),xy=(80,ypos),xycoords="figure pixels",fontsize=8)
axs[1].annotate(r'$\chi^2$/dof, $k<k_{max}$:' + str(np.around(rest_chi2[1:],3)),xy=(80,199),xycoords="figure pixels",fontsize=8)
axs[1].set_ylabel(r'angle var.',fontsize=12)
axs[1].set_ylim([0.01,1.2])
axs[1].set_yscale('log')
axs[1].axvline(x=kmax, label='k max', c='k',linewidth=0.3)
pl.xlim([0.95*kph[0],1.01*kph[-1]])
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.savefig(dir_results + '/' + 'Data_var_00_FKP.pdf')
pl.close('all')

print 

# Now, MT
full_chi2 = np.sum(np.power((Data_Sph_00_mt - Mean_Sph_00_mt)/sigma_Sph_00_mt,2.0),axis=1)/len(kph)
rest_chi2 = np.sum(np.power((Data_Sph_00_mt - Mean_Sph_00_mt)/sigma_Sph_00_mt,2.0)[:,:kstop],axis=1)/len(kph[:kstop])
print " MT -- Chi^2, full range of k:", np.around(full_chi2,4)
print " MT -- Chi^2,  k<", str(kmax), ":", np.around(rest_chi2,4)
fig, axs = pl.subplots(2, 1, sharex=True)
fig.subplots_adjust(hspace=0)
kpk_data=np.power(kph,1.5)*Data_Sph_00_mt[0]
kpk_mean=np.power(kph,1.5)*Mean_Sph_00_mt[0]
axs[0].plot(kph,kpk_data,color=mycolor[0],marker=".",linestyle='None',markersize=4.0)
# Errors multiplied by 2 for visualization purposes
axs[0].errorbar(kph,kpk_mean,np.power(kph,1.5)*2*sigma_Sph_00_mt[0],color=mycolor[0],linestyle=(0, (1, 2)),linewidth=0.8)
axs[0].set_ylabel(r'${\cal{P}}_{Tot}$',fontsize=12)
axs[0].set_ylim([0.7*np.min(kpk_mean),1.3*np.max(kpk_mean)])
axs[0].set_title(r'$k^{1.5} \times {\cal{P}}_{Tot}$ & angle var., MT',fontsize=16)
#axs[0].annotate(r'${\cal{P}}_{Tot}$: $\chi^2$/dof, all $k$:' + str(np.around(full_chi2[0],3)),xy=(80.0,370.0),xycoords="figure pixels",fontsize=8)
axs[0].annotate(r'$\chi^2$/dof, $k<k_{max}$:' + str(np.around(rest_chi2[0],3)),xy=(80.0,370.0),xycoords="figure pixels",fontsize=8)
axs[0].axvline(x=kmax, label='k max', c='k',linewidth=0.3)
for nt in range(1,ntracers):
    axs[1].plot( (1+0.01*(nt-1))*kph , Data_Sph_00_mt[nt],color=mycolor[nt],marker=".",linestyle='None',markersize=4.0)
    axs[1].errorbar( (1+0.01*(nt-1))*kph ,Mean_Sph_00_mt[nt], 2*sigma_Sph_00_mt[nt],color=mycolor[nt],linestyle=(0, (1, 2)),linewidth=0.8)
    ypos=195-45*(nt-1)
    #axs[1].annotate(r'$\chi^2$/dof, all $k$:' + str(np.around(full_chi2[nt],3)),xy=(80,ypos),xycoords="figure pixels",fontsize=8)
    #axs[1].annotate(r'$\chi^2$/dof, $k<k_{max}$:' + str(np.around(rest_chi2[nt],3)),xy=(80,ypos),xycoords="figure pixels",fontsize=8)
axs[1].annotate(r'$\chi^2$/dof, $k<k_{max}$:' + str(np.around(rest_chi2[1:],3)),xy=(80,199),xycoords="figure pixels",fontsize=8)
axs[1].set_ylabel(r'angle var.',fontsize=12)
axs[1].set_ylim([0.01,1.2])
axs[1].set_yscale('log')
axs[1].axvline(x=kmax, label='k max', c='k',linewidth=0.3)
pl.xlim([0.95*kph[0],1.01*kph[-1]])
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.savefig(dir_results + '/' + 'Data_var_00_MT.pdf')
pl.close('all')


print 
print "P^2/P^0: FKP v. MT"
# First, for FKP
full_chi2 = np.sum(np.power((Data_Sph_20_fkp - Mean_Sph_20_fkp)/sigma_Sph_20_fkp,2.0),axis=1)/len(kph)
rest_chi2 = np.sum(np.power((Data_Sph_20_fkp - Mean_Sph_20_fkp)/sigma_Sph_20_fkp,2.0)[:,:kstop],axis=1)/len(kph[:kstop])
print "FKP -- Chi^2, full range of k:", np.around(full_chi2,4)
print "FKP -- Chi^2,  k<", str(kmax), ":", np.around(rest_chi2,4)
for nt in range(ntracers):
    pl.errorbar( (1+0.01*(nt-1))*kph ,2*nt+Mean_Sph_20_fkp[nt],sigma_Sph_20_fkp[nt],color=mycolor[nt],linestyle=(0, (1, 2)),linewidth=0.8)
    pl.plot( (1+0.01*(nt-1))*kph ,2*nt+Data_Sph_20_fkp[nt],color=mycolor[nt],marker="x",linestyle='None',markersize=4.0)
pl.ylabel(r'$P^{(2)}/P^{(0)}$',fontsize=12)
pl.ylim([-2,10])
#pl.yscale('log')
pl.axvline(x=kmax, label='k max', c='k',linewidth=0.3)
pl.xlim([0.95*kph[0],1.05*kph[-1]])
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.title(r'${\cal{P}}^{(2)}/{\cal{P}}^{(0)}$ , FKP',fontsize=16)
pl.savefig(dir_results + '/' + 'Data_var_20_FKP.pdf')
pl.close('all')

print 

# Now, MT
full_chi2 = np.sum(np.power((Data_Sph_20_mt - Mean_Sph_20_mt)/sigma_Sph_20_mt,2.0),axis=1)/len(kph)
rest_chi2 = np.sum(np.power((Data_Sph_20_mt - Mean_Sph_20_mt)/sigma_Sph_20_mt,2.0)[:,:kstop],axis=1)/len(kph[:kstop])
print " MT -- Chi^2, full range of k:", np.around(full_chi2,4)
print " MT -- Chi^2,  k<", str(kmax), ":", np.around(rest_chi2,4)
for nt in range(ntracers):
    pl.errorbar( (1+0.01*(nt-1))*kph , 2*nt+Mean_Sph_20_mt[nt],sigma_Sph_20_mt[nt],color=mycolor[nt],linestyle=(0, (1, 2)),linewidth=0.8)
    pl.plot( (1+0.01*(nt-1))*kph , 2*nt+Data_Sph_20_mt[nt],color=mycolor[nt],marker=".",linestyle='None',markersize=4.0)
pl.ylabel(r'$P^{(2)}/P^{(0)}$',fontsize=12)
pl.ylim([-2,10])
#pl.yscale('log')
pl.axvline(x=kmax, label='k max', c='k',linewidth=0.3)
pl.xlim([0.95*kph[0],1.05*kph[-1]])
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.title(r'${\cal{P}}^{(2)}/{\cal{P}}^{(0)}$ , MT',fontsize=16)
pl.savefig(dir_results + '/' + 'Data_var_20_MT.pdf')
pl.close('all')

print 

# Now, Mix
full_chi2 = np.sum(np.power((Data_Sph_20_mix - Mean_Sph_20_mix)/sigma_Sph_20_mix,2.0),axis=1)/len(kph)
rest_chi2 = np.sum(np.power((Data_Sph_20_mix - Mean_Sph_20_mix)/sigma_Sph_20_mix,2.0)[:,:kstop],axis=1)/len(kph[:kstop])
print "Mix -- Chi^2, full range of k:", np.around(full_chi2,4)
print "Mix -- Chi^2,  k<", str(kmax), ":", np.around(rest_chi2,4)
for nt in range(ntracers):
    pl.errorbar( (1+0.01*(nt-1))*kph , 2*nt+Mean_Sph_20_mix[nt],sigma_Sph_20_mix[nt],color=mycolor[nt],linestyle=(0, (1, 2)),linewidth=0.8)
    pl.plot( (1+0.01*(nt-1))*kph , 2*nt+Data_Sph_20_mix[nt],color=mycolor[nt],marker=".",linestyle='None',markersize=4.0)
pl.ylabel(r'$P^{(2)}/P^{(0)}$',fontsize=12)
pl.ylim([-2,10])
#pl.yscale('log')
pl.axvline(x=kmax, label='k max', c='k',linewidth=0.3)
pl.xlim([0.95*kph[0],1.05*kph[-1]])
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.title(r'${\cal{P}}^{(2)}/{\cal{P}}^{(0)}$ , Mix',fontsize=16)
pl.savefig(dir_results + '/' + 'Data_var_20_Mix.pdf')
pl.close('all')

print
print "Total information from Inverse(Relative covariance)"
print "Power spectra:"
print "               P^0_i, FKP: ", np.sum(np.linalg.inv(rCov_P0_fkp))
print "               P^0_i,  MT: ", np.sum(np.linalg.inv(rCov_P0_mt))
print "               P^2_i, FKP: ", np.sum(np.linalg.inv(rCov_P2_fkp))
print "               P^2_i,  MT: ", np.sum(np.linalg.inv(rCov_P2_mt))
print "P^0_i (FKP) + P^2_i (FKP): ", np.sum(np.linalg.inv(rCov_P20_fkp))
print "P^0_i (MT)  + P^2_i (FKP): ", np.sum(np.linalg.inv(rCov_P20_mix))
print 
print "Spherical variables:"
print "             Sph_00, FKP: ", np.sum(np.linalg.inv(rCov_Sph_00_fkp))
print "             Sph_00,  MT: ", np.sum(np.linalg.inv(rCov_Sph_00_mt))
print "             Sph_20, FKP: ", np.sum(np.linalg.inv(rCov_Sph_20_fkp))
print "             Sph_20,  MT: ", np.sum(np.linalg.inv(rCov_Sph_20_mt))
print "             Sph_20, MIX: ", np.sum(np.linalg.inv(rCov_Sph_20_mix))




# Finally, save covariance matrices for Sph. and P2/P0
np.savetxt(dir_results + '/' + 'Cov_P0_fkp.dat',Cov_P0_fkp)
np.savetxt(dir_results + '/' + 'Cov_P2_fkp.dat',Cov_P2_fkp)
np.savetxt(dir_results + '/' + 'Cov_P0_mt.dat',Cov_P0_mt)
np.savetxt(dir_results + '/' + 'Cov_P2_mt.dat',Cov_P2_mt)
np.savetxt(dir_results + '/' + 'Cov_P20_mt.dat',Cov_P20_mt)
np.savetxt(dir_results + '/' + 'Cov_P20_fkp.dat',Cov_P20_fkp)
np.savetxt(dir_results + '/' + 'Cov_P20_mix.dat',Cov_P20_mix)

np.savetxt(dir_results + '/' + 'Cov_Sph_00_fkp.dat',Cov_Sph_00_fkp)
np.savetxt(dir_results + '/' + 'Cov_Sph_20_fkp.dat',Cov_Sph_20_fkp)
np.savetxt(dir_results + '/' + 'Cov_Sph_00_mt.dat',Cov_Sph_00_mt)
np.savetxt(dir_results + '/' + 'Cov_Sph_20_mt.dat',Cov_Sph_20_mt)
np.savetxt(dir_results + '/' + 'Cov_Sph_20_mix.dat',Cov_Sph_20_mix)


# Save means of the mocks
np.savetxt(dir_results + '/' + 'Means_P0_fkp.dat',Mean_P0_fkp)
np.savetxt(dir_results + '/' + 'Means_P2_fkp.dat',Mean_P2_fkp)
np.savetxt(dir_results + '/' + 'Means_P0_mt.dat',Mean_P0_mt)
np.savetxt(dir_results + '/' + 'Means_P2_mt.dat',Mean_P2_mt)

np.savetxt(dir_results + '/' + 'Means_Sph_00_fkp.dat',Mean_Sph_00_fkp.T)
np.savetxt(dir_results + '/' + 'Means_Sph_20_fkp.dat',Mean_Sph_20_fkp.T)
np.savetxt(dir_results + '/' + 'Means_Sph_00_mt.dat',Mean_Sph_00_mt.T)
np.savetxt(dir_results + '/' + 'Means_Sph_20_mt.dat',Mean_Sph_20_mt.T)
np.savetxt(dir_results + '/' + 'Means_Sph_20_mix.dat',Mean_Sph_20_mix.T)


# Save datapoints
np.savetxt(dir_results + '/' + 'Data_P0_fkp.dat',Data_P0_fkp)
np.savetxt(dir_results + '/' + 'Data_P2_fkp.dat',Data_P2_fkp)
np.savetxt(dir_results + '/' + 'Data_P0_mt.dat',Data_P0_mt)
np.savetxt(dir_results + '/' + 'Data_P2_mt.dat',Data_P2_mt)

np.savetxt(dir_results + '/' + 'Data_Sph_00_fkp.dat',Data_Sph_00_fkp.T)
np.savetxt(dir_results + '/' + 'Data_Sph_20_fkp.dat',Data_Sph_20_fkp.T)
np.savetxt(dir_results + '/' + 'Data_Sph_00_mt.dat',Data_Sph_00_mt.T)
np.savetxt(dir_results + '/' + 'Data_Sph_20_mt.dat',Data_Sph_20_mt.T)
np.savetxt(dir_results + '/' + 'Data_Sph_20_mix.dat',Data_Sph_20_mix.T)


# Save basic properties of the maps, bandpowers and fiducial bias
np.savetxt(dir_results + '/' + 'k_phys.dat',kph)
np.savetxt(dir_results + '/' + 'nbar_mean.dat',nbar)
np.savetxt(dir_results + '/' + 'bias.dat',bias)


# Save one of the mocks for testing
np.savetxt(dir_results + '/' + 'Mockdata_P0_fkp.dat',P0_fkp[:,:,1])
np.savetxt(dir_results + '/' + 'Mockdata_P2_fkp.dat',P2_fkp[:,:,1])
np.savetxt(dir_results + '/' + 'Mockdata_P0_mt.dat',P0_mt[:,:,1])
np.savetxt(dir_results + '/' + 'Mockdata_P2_mt.dat',P2_mt[:,:,1])

np.savetxt(dir_results + '/' + 'Mockdata_Sph_00_fkp.dat',Sph_00_fkp[:,:,1].T)
np.savetxt(dir_results + '/' + 'Mockdata_Sph_20_fkp.dat',Sph_20_fkp[:,:,1].T)
np.savetxt(dir_results + '/' + 'Mockdata_Sph_00_mt.dat',Sph_00_mt[:,:,1].T)
np.savetxt(dir_results + '/' + 'Mockdata_Sph_20_mt.dat',Sph_20_mt[:,:,1].T)
np.savetxt(dir_results + '/' + 'Mockdata_Sph_20_mix.dat',Sph_20_mix[:,:,1].T)
