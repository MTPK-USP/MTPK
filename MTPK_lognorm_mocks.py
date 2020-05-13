#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
------------
Based on code initiated by Arthur E. da Mota Loureiro 04/2015
1st version by R. Abramo: 03/2016
02/2018 -- Last modified to include * spectral corrections
* estimation of monopole and quadrupoles in the sim box


This code create homogeneous halo catalogs.
The estimation tool can create galaxy catalogs from these homogeneous
halo catalogs, given an HOD.
------------
"""
#from __future__ import print_function
import numpy as np
import sys
import glob
import os
import uuid
import h5py
from time import time
from scipy import interpolate, special, ndimage

# Add path to /inputs directory in order to load inputs
this_dir = os.getcwd()
input_dir = this_dir + '/inputs'
sys.path.append(input_dir)

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

from camb_spec import camb_spectrum
from cosmo_funcs import matgrow, H
from multipoles import delta_r

import grid3D as gr
import gauss_pk_class as pkgauss


small=1.e-8


#####################################################
# Load name of the run
#####################################################
from MTPK import *

print()
print ('Multi-tracer lognormal simulations tool')
print ('Handle of this run of sims: ', handle_sims)
print()

input_filename = glob.glob('inputs/' + handle_sims + '.py')
if len(input_filename)==0 :
    print ('Input file not found! Check handle name on /inputs.')
    print ('Exiting program...')
    sys.exit(-1)

# Check if the handles defined in the input have already been used in a another run
dir_maps = 'maps/sims/' + handle_sims

## Check if data directory exists
#dir_data = 'maps/data/' + handle_data

if not os.path.exists(dir_maps):
    os.makedirs(dir_maps)
elif len(os.listdir(dir_maps)) != 0:
    print()
    print ('Simulated maps were already created with this handle/name, on:')
    print( dir_maps)
    print()
    answer = input('Continuing will overwrite those files. Proceed? y/n  ')
    if answer != 'y':
        print( 'Aborting now...')
        print()
        sys.exit(-1)

dir_specs = 'spectra/' + handle_sims
if not os.path.exists(dir_specs):
    os.makedirs(dir_specs)
elif len(os.listdir(dir_specs)) != 0:
    print( 'WARNING: another run with the same handle/name was already performed!')
    answer = input('Check specs/ . Continue anyway? y/n  ')
    if answer!='y':
        print ('Aborting now...')
        sys.exit(-1)

dir_figs = 'figures/' + handle_sims
if not os.path.exists(dir_figs):
    os.makedirs(dir_figs)
if len(os.listdir(dir_figs)) != 0:
    print( 'WARNING: another run with the same handle/name was already performed!')
    answer = input('Check figures/ . Continue anyway? y/n  ')
    if answer!='y':
        print ('Aborting now...')
        sys.exit(-1)

# Import inputs
# (the path to /inputs was added in the beginning of the code)
exec("from " + handle_sims + " import *")


# Growth rate and its derivative
Omegam = Omegac + Omegab
OmegaDE = 1. - Omegam - Omegak
h = H0/100.

try:
    gamma
except:
    gamma = 0.55


try:
    matgrowcentral
except:
    matgrowcentral = matgrow(Omegam, OmegaDE, w0, w1, zcentral, gamma)
else:
    print('ATTENTION: pre-defined (on input) matter growth rate =',matgrowcentral)

f_grow = np.ones(nhalos)*matgrowcentral

# Velocity dispersion. vdisp is defined on inputs with units of km/s
vdisp = np.asarray(vdisp)
sigma_v = vdisp/H(100, Omegam, OmegaDE, -1, 0.0, zcentral) #Mpc/h
a_vdisp = vdisp/c  #Adimensional vdisp

# Redshift errors. sigz_est is defined on inputs, and is adimensional
sigz_est = np.asarray(sigz_est)
sigma_z = sigz_est*c/H(100, Omegam, OmegaDE, -1, 0.0, zcentral) #Mpc/h

# Joint factor considering v dispersion and z error
sig_tot = np.sqrt(sigma_z**2 + sigma_v**2) #Mpc/h
a_sig_tot = np.sqrt(sigz_est**2 + a_vdisp**2) #Adimensional sig_tot

############# Calling CAMB for calculations of the spectra #################
print('Beggining CAMB calculations\n')

# It is strongly encouraged to use k_min >= 1e-4, since it is a lot faster
k_min_camb = 1.0e-4
k_max_camb = 1.5e1

# Where to evaluate spectra:
nklist = 2000
k_camb = np.logspace(-4.0,1.5,nklist)

kc, pkc = camb_spectrum(H0, Omegab, Omegac, w0, w1, z_re, zcentral, n_SA, k_min_camb, k_max_camb, whichspec)
Pk_camb = np.asarray( np.interp(k_camb, kc, pkc) )


try:
	power_low
except:
	pass
else:
	Pk_camb = power_low*np.power(Pk_camb,pk_power)


# Make sure this spectrum decays sufficiently rapidly for very high k's:
k_camb = np.append(k_camb,np.array([2*k_camb[-1],4*k_camb[-1],8*k_camb[-1],16*k_camb[-1]]))
Pk_camb = np.append(Pk_camb,np.array([1./4.*Pk_camb[-1],1./16*k_camb[-1],1./64*k_camb[-1],1./256*k_camb[-1]]))


# This tool generates only "halos", with densities and biases given by the
# halo mass function and halo bias models (Tinker, or something else).
# Import tables with HOD and halo model props
# Attention: mass function in units of h^3 Mpc^-3!
try:
    hod = np.loadtxt("inputs/" + hod_file)
    mass_fun = np.loadtxt("inputs/" + mass_fun_file)
    halo_bias = np.loadtxt("inputs/" + halo_bias_file)
except:
    print( "Something's wrong... did not find HOD, mass function and/or halo bias files!")
    print( "Check in the /inputs directory. Aborting now...")
    sys.exit(-1)

mass_fun = mult_sel_fun*mass_fun

# This is the mean number of halos of each bin in each cell
# It is valid for the whole box. The mass function and halo bias
# are given in the files above, on the dir /inputs
nbar = np.asarray(mass_fun)*cell_size**3
# This is the bias of the halos
bias = np.asarray(halo_bias)


# Physical sizes of boxes
L_x = n_x*cell_size ; L_y = n_y*cell_size ; L_z = n_z*cell_size     # size of the box
box_vol = L_x*L_y*L_z			# Box's volume
L_max = np.sqrt(L_x*L_x + L_y*L_y + L_z*L_z)	



#####################################################
# Generate real- and Fourier-space grids for FFTs
#####################################################
grid = gr.grid3d(n_x,n_y,n_z,L_x,L_y,L_z)		# generates the grid


#R Now fetch the fiducial values, for a LambdaCDM model, for:
#R   - Power spectrum at mean k,
#R   - Growth function at mean redshift,
#R   - Matter growth rate at mean redshift

beta = f_grow*np.power(bias,-1)
#quad = 2*beta/(3.0 + beta)

# Define numpy arrays
adip = np.asarray(adip)


# Here, the tracers are the halos
ntracers = nhalos

print('Will generate maps for ', ntracers, ' tracers,')

# Print out which spectrum is used
if whichspec == 0:
    print('using LINEAR power spectrum from CAMB ')
elif whichspec == 1:
    print('using power spectrum from CAMB + HaloFit ')
else:
    print('using power spectrum from CAMB + HaloFit with PkEqual')

print('Creating ' + str(n_maps) + ' simulated map(s) for each tracer, of (nx,ny,nz) = (' +str(n_x)+','+str(n_y)+','+str(n_z)+'), of cell_size=' + str(cell_size))
print()

print('Halo biases =' + str(bias))
#print('Gaussian Photo-z errors (added in Fourier space)='+str(sigz_map))
print('Assuming map centered at z='+str(zcentral))


#################################
# Calculating the P_Gauss(k) grid
# I will add the RSDs later, at the level of the *Gaussian* spectrum 
#################################

# This k is in physical units! Need this to interpolate from CAMB spectrum
k_flat = grid.grid_k.flatten()*(2.*np.pi/cell_size)

# Define spectra where we will evaluate P^G_ij and its inverse
stop = np.max(np.where(k_camb < 2.0))
start = np.min(np.where(k_camb > 0.0001))
kspec = np.append(0.,0.75*k_camb[start:stop])
#kspec = np.append(0.,0.75*k_camb[start:stop:2])
nk = (kspec.shape)[0]


# Displacements of the box -- used for RSDs
LX0 = n_x_orig
LY0 = n_y_orig
LZ0 = n_z_orig

rL0=np.sqrt((LX0 + grid.RX)**2 + (LY0 + grid.RY)**2 + (LZ0 + grid.RZ)**2)

rxhat = (LX0 + grid.RX)/rL0
ryhat = (LY0 + grid.RY)/rL0
rzhat = (LZ0 + grid.RZ)/rL0

kxhat = ((grid.KX)/(grid.grid_k+small))
kyhat = ((grid.KY)/(grid.grid_k+small))
kzhat = ((grid.KZ)/(grid.grid_k+small))

# Sometimes only need 1/2 of the modes for real-valued functions
kxhat_half = kxhat[:,:,:n_z//2+1]
kyhat_half = kyhat[:,:,:n_z//2+1]
kzhat_half = kzhat[:,:,:n_z//2+1]
k_half  = grid.grid_k[:,:,:n_z//2+1]


p_mono = np.zeros((ntracers,ntracers,nk))
# p_mono == P_ij represents the monopole of the power spectra, evaluated at kspec
# We now invert the P_ij for each kspec .
# In order to GUARANTEE that the inverse is well-defined,
# I will add an additional cross-correlation "fudge factor" between the tracers.
# My tests show that a cross-correlation factor of the order of 1%
# of the diagonal terms is sufficient to ensure that the inverse covariance is positive-definite

print('Generating matrix of fiducial power spectra...')

for nt in range(ntracers):
    # NEW PROCEDURE FOR LOGNORMAL MAPS
    effbiasnt = bias[nt]
    monoij = effbiasnt*effbiasnt
    pkg = pkgauss.gauss_pk(k_camb,monoij*Pk_camb,grid.grid_k,cell_size,L_max)
    p_mono[nt,nt] = np.abs(pkg.Pk_gauss_interp(kspec))
    corr1 = p_mono[nt,nt]
    for ntp in range(nt+1,ntracers):
        effbiasntp = bias[ntp]
        monocross = effbiasnt*effbiasntp
        corr2 = p_mono[ntp,ntp]
        pkg = pkgauss.gauss_pk(k_camb,monocross*Pk_camb,grid.grid_k,cell_size,L_max)
        c12 = np.abs(pkg.Pk_gauss_interp(kspec))
        p_mono[nt,ntp] = c12
        p_mono[ntp,nt] = c12


# I will construct the random variables by starting with a grid of "white noise" random variables, X,
# of 0 mean and variance = 1 . Then, the modes, f, will be constructed using the Cholesky decomposition,
# in the following way: if Cov == Cov(f) , then Cov = Ch . Transpose(Ch) , where Ch is the
# Cholesky matrix for Cov . Then, with Cov(X) = 1, it follows that f = Ch . X .
diagmat = np.diag(np.ones(ntracers))
offdiag = np.ones((ntracers,ntracers)) - diagmat
pch = np.zeros((ntracers,ntracers,nk))
pch[:,:,0] = np.diag(np.ones(ntracers))

print('Diagonalizing d.o.f. ...')

normbox=2.0*box_vol
for k in range(1,nk):
    l , v = np.linalg.eigh(p_mono[:,:,k])
    posl = 0.5*(l + np.abs(l.real))
    newl = np.diag(posl + fudge * (fudge + np.random.random(ntracers))*np.mean(posl))
    p_corr = normbox * np.dot( v , np.dot(newl,np.conjugate(v).T))
    pch[:,:,k] = np.linalg.cholesky(p_corr)

print('Interpolating ...')

# Now interpolate the elements of pch at every value on k_flat
pch_kgrid = np.zeros((ntracers,ntracers,n_x,n_y,n_z),dtype=np.float32)
for nt in range(ntracers):
    for ntp in range(ntracers):
        interpfun = interpolate.PchipInterpolator(kspec,pch[nt,ntp])
        pch_kgrid[nt,ntp] = interpfun(k_flat).reshape((n_x,n_y,n_z))

# These are the matrices such that Pch . Transpose(Pch) = P_ij (k_x,k_y,k_z)

###########################################
# Define lognormal mapping
# This definition regularizes high-variance fields
def delta_x_ln(d_,sigma2_,bias_):
    ###############################
    # The log-normal density field
    ###############################
    mean_delta2 = np.sqrt(np.var(d_))
    dlim = min(3.0*mean_delta2,7.0)
    d_[d_ > dlim] = dlim * np.power((d_[d_ > dlim]/dlim),0.75)
    dln = np.exp(d_)
    mln = np.mean(dln)
    return dln/mln - 1.0
###########################################


print('Now generating the maps...')

# Now we create the maps -- ntracers * n_maps in total
maps=np.zeros((ntracers,n_maps,n_x,n_y,n_z))
lnmaps=np.zeros((ntracers,n_maps,n_x,n_y,n_z))

lnmaps_fourier=(0.+1j*0.0)*np.zeros((ntracers,n_maps,n_x,n_y,n_z//2+1))
lnmaps_fourier_rsds=(0.+1j*0.0)*np.zeros((ntracers,n_maps,n_x,n_y,n_z//2+1))
lnmaps_Bterm_fourier=(0.+1j*0.0)*np.zeros((ntracers,n_maps,n_x,n_y,n_z//2+1))
lnmaps_halos_fourier=(0.+1j*0.0)*np.zeros((ntracers,n_maps,n_x,n_y,n_z//2+1))

delta_k_tracer = (0.+1j*0.0)*np.zeros((ntracers,n_maps,n_x,n_y,n_z))

# Always remember that 'k' in grid_k are the frequencies, not physical k's
# Notice that rL0 is given in units of cell

# To correctly implement the FOGs we'll have to use physical k
kgrid_phys = 2*np.pi*grid.grid_k/cell_size

kr0 = 2.*np.pi*np.mean(grid.grid_k)*np.mean(rL0)

# Create sets of Gaussian modes, and maps of tracers for those modes
for i in range(n_maps):
    print( 'Map #', i) 
    phases = np.exp(1j*2.0*np.pi*np.random.rand(n_x,n_y,n_z) )
    # Testing
    X_mu_k = phases*np.random.normal(0.,np.ones((ntracers,n_x,n_y,n_z)))
    #X_mu_k = np.random.normal(0.,np.ones((ntracers,n_x,n_y,n_z)))
    halodensity = np.zeros(ntracers,dtype=np.float16)
    sigma_gauss = np.zeros(ntracers)
    sigma_ln = np.zeros(ntracers)
    compare_B_term = np.zeros(ntracers)
    #print '           Var X = ', np.mean(np.abs(X_mu_k)**2)
    # compare_D_term = np.zeros(ntracers)
    B_tracer = np.zeros((ntracers,n_x,n_y,n_z))
    ksize = delta_k_tracer[0,0].size
    for nt in range(ntracers):
        delta_k_tracer[nt,i] = np.sum(pch_kgrid[nt]*X_mu_k,axis=0)
        ksize = delta_k_tracer[nt,i].size
        
        ###!!! Originally, the RSDs were inserted here, in the Gaussian field
        # delta_k_tracer = corr_factors[nt]*delta_k_tracer
        # New method using direct RSDs

        delta_x_gaus = ((ksize)/box_vol)*np.fft.ifftn(delta_k_tracer[nt,i])
        delta_xr_g = delta_x_gaus.real
        var_gr = np.var(delta_xr_g)
        sigma_gauss[nt] = np.sqrt(var_gr)
        if var_gr > 10000. :
            print('Sorry, the code is unstable when the variance of the density field is too large: tracer', nt, ' sigma = ', '%1.4f'% np.sqrt(var_gr) )
            print('Try again with a larger cell size (see input file).  Exiting now...')
            sys.exit(-1)
        
        ###########################
        # Log-Normal Density Field
        ###########################
        delta_xr = delta_x_ln(delta_xr_g, var_gr, 1.0)


        # TESTING
        mxr = np.mean(delta_xr)
        vxr = np.var(delta_xr)
        sigma_ln[nt] = np.sqrt(vxr)
        print('    <delta_LN> , <(delta_LN)^2> =', '%1.5f'% mxr , '%1.5f'% vxr )

        delta_ln_k = np.fft.rfftn(delta_xr)
        lnmaps_fourier[nt,i] = delta_ln_k
        
        # RSDs
        
        # Dipole term -- not defined for halos!
        # Remember that grid_k is in grid frequency units -- i.e., without the 2 pi factor
        #Dx  = np.fft.irfftn( 1j*kxhat_half / (small + 2.*np.pi*k_half) * delta_ln_k, s=[n_x,n_y,n_z] )
        #Dy  = np.fft.irfftn( 1j*kyhat_half / (small + 2.*np.pi*k_half) * delta_ln_k, s=[n_x,n_y,n_z] )
        #Dz  = np.fft.irfftn( 1j*kzhat_half / (small + 2.*np.pi*k_half) * delta_ln_k, s=[n_x,n_y,n_z] )
        #Dterm = adip[nt]*( rxhat*Dx + ryhat*Dy + rzhat*Dz)/rL0
    
        # #Dterm = Dterm.real

        delta_rsds = delta_r(kgrid_phys,kxhat,kyhat,kzhat,rxhat,ryhat,rzhat,delta_ln_k,sig_tot[nt],bias[nt],matgrowcentral)
        # Real part only
        delta_rsds = delta_rsds.real
        # clip values < -1
        delta_rsds [ delta_rsds < -1.0 ] = -1.0
        
        lnmaps_fourier_rsds[nt,i] = np.fft.rfftn(delta_rsds)

        # compare_B_term[nt] = np.sqrt(np.mean(Bterm**2))/np.sqrt(np.mean(delta_xr**2))
        # compare_D_term[nt] = np.sqrt(np.mean(Dterm.real**2))/np.sqrt(np.mean(delta_xr**2))
        
        
        # Estimate quadrupole for this map
        iBxx = np.fft.rfftn(rxhat*rxhat*delta_rsds)
        iByy = np.fft.rfftn(ryhat*ryhat*delta_rsds)
        iBzz = np.fft.rfftn(rzhat*rzhat*delta_rsds)
        iBxy = np.fft.rfftn(rxhat*ryhat*delta_rsds)
        iBxz = np.fft.rfftn(rxhat*rzhat*delta_rsds)
        iByz = np.fft.rfftn(ryhat*rzhat*delta_rsds)
        iBterm = 3.0*(kxhat_half**2*iBxx + kyhat_half**2*iByy + kzhat_half**2*iBzz + 2.0*kxhat_half*kyhat_half*iBxy + 2.0*kxhat_half*kzhat_half*iBxz + 2.0*kyhat_half*kzhat_half*iByz)
        
        lnmaps_Bterm_fourier[nt,i] = iBterm
        
        #######################
        # Poissonian realization
        # This is the final galaxy Map
        #######################
        thismap = np.random.poisson(nbar[nt]*(1. + delta_rsds)/(1 + np.mean(delta_rsds)))
        #thismap = np.random.poisson(nbar[nt]*(1. + delta_rsds))
        Nhalos = np.sum(thismap)
        
        # Halo density field v. matter density field
        delta_halos = thismap/nbar[nt] - 1.0
        lnmaps_halos_fourier[nt] = np.fft.rfftn(delta_halos)

        # Some final checks
        halodensity[nt] = Nhalos/box_vol
        maps[nt,i] = thismap
        lnmaps[nt,i] = delta_xr

    frac = halodensity/nbar*cell_size**3
    print( '    densities true/fiducial = ', [ "{:1.4f}".format(x) for x in frac ])
    print()

pch = None
del pch

pch_kgrid = None
del pch_kgrid

print()
print('Computing spectral corrections to lognormal maps...')
# Compare input spectrum with spectra of lognormal maps (before RSDs!)
# normalization from Fourier transform to power spectra
Fnorm = cell_size**3/(n_x*n_y*n_z)

# Spectrum in redshift space, <delta_k^2>:
Pk_ln_halos_Mono_norm = Fnorm * np.average( np.power(np.abs(lnmaps_halos_fourier),2) , axis=1 )

lnmaps_halos_fourier = None
del lnmaps_halos_fourier

# Spectrum in real (not redshift) space, <delta_k^2>:
Pk_ln_norm = Fnorm * np.average( np.power(np.abs(lnmaps_fourier),2) , axis=1 )

Cross_Pk_ln_norm = np.zeros((ntracers*(ntracers-1)//2,n_x,n_y,n_z//2+1))
index=0
for i in range(ntracers):
	for j in range(i+1,ntracers):
		tmp = Fnorm * np.average( lnmaps_fourier[i]*np.conjugate(lnmaps_fourier[j]) , axis=0 )
		Cross_Pk_ln_norm[index] = 0.5*np.real( tmp + np.conjugate(tmp) )
		index += 1

lnmaps_fourier = None
del lnmaps_fourier

# Monopole:
# Angle average < ( 1 + beta * mu^2 ) delta_k ( 1 + beta * mu^2 ) delta_k >
#              --> (b^2 + 2/3*b*f + 1/5*f^2) * P(k)
Pk_ln_Mono_norm = Fnorm * np.average( np.abs(lnmaps_fourier_rsds)**2 , axis=1 )

Cross_Pk_ln_Mono_norm = np.zeros((ntracers*(ntracers-1)//2,n_x,n_y,n_z//2+1))
index=0
for i in range(ntracers):
	for j in range(i+1,ntracers):
		Cross_Pk_ln_Mono_norm[index] = 0.5 * Fnorm * np.real( np.average( lnmaps_fourier_rsds[i]*np.conjugate(lnmaps_fourier_rsds[j]) + lnmaps_fourier_rsds[j]*np.conjugate(lnmaps_fourier_rsds[i]) , axis=0 ) )
		index += 1

# Quadrupole:
# Angle average < (-1 + 3*mu^2) * ( 1 + beta * mu^2 )^2 * delta_k^2 >
#              --> 8/5*(b*f/3 + f^2/7) * P(k)
Pk_ln_Quad_norm = Fnorm * np.average( (lnmaps_fourier_rsds*np.conjugate(lnmaps_Bterm_fourier)).real , axis=1 ) - Pk_ln_Mono_norm

Cross_Pk_ln_Quad_norm = np.zeros((ntracers*(ntracers-1)//2,n_x,n_y,n_z//2+1))

index=0
for i in range(ntracers):
	for j in range(i+1,ntracers):
		Cross_Pk_ln_Quad_norm[index] = 0.5 * Fnorm * np.real( np.average( lnmaps_fourier_rsds[i]*np.conjugate(lnmaps_Bterm_fourier[j]) + lnmaps_fourier_rsds[j]*np.conjugate(lnmaps_Bterm_fourier[i]) , axis=0 ) )
		index += 1


lnmaps_fourier_rsds = None
del lnmaps_fourier_rsds
lnmaps_Bterm_fourier = None
del lnmaps_Bterm_fourier



##############################

# Order the k's for interpolation -- here, k in physical units (h/Mpc)
k_half_flat = 2*np.pi/cell_size*k_half.flatten()
k_order = np.argsort(k_half_flat)
k_sort = k_half_flat[k_order]
k_max = k_sort[-1]
k_min = k_sort[1]

# Bin k_sort in bins of dkph_bin (defined on input)
dk = dkph_bin
k_bins = np.arange(k_min,k_max + dk/2.0,dk)
k_sort_bins = np.digitize(k_sort,k_bins)
k_max_index = np.max(k_sort_bins)
# Will use these k's for the spectral corrections
k_interp = k_bins[:k_max_index] + dk/2.0

Pk_camb_interp_fun = interpolate.interp1d(k_camb,Pk_camb)
Pk_camb_interp = Pk_camb_interp_fun(k_interp)


# The angle averages are NOT equivalent to a true spherical mean,
# which forces us to compute the effective averages, such as:
# < \mu^2 > --> \Sum_i < (k^hat_i r^hat_i)^2 >
rx2 = np.mean(rxhat**2)
ry2 = np.mean(ryhat**2)
rz2 = np.mean(rzhat**2)

kxhat2_half_flat = ((kxhat_half**2).flatten())[k_order]
kyhat2_half_flat = ((kyhat_half**2).flatten())[k_order]
kzhat2_half_flat = ((kzhat_half**2).flatten())[k_order]

kxhat4_half_flat = kxhat2_half_flat**2
kyhat4_half_flat = kyhat2_half_flat**2
kzhat4_half_flat = kzhat2_half_flat**2

kxhat6_half_flat = kxhat2_half_flat**3
kyhat6_half_flat = kyhat2_half_flat**3
kzhat6_half_flat = kzhat2_half_flat**3


# We use the CAMB spectra to weight the angle averages
kxhat2_interp = np.asarray( [ kxhat2_half_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
kyhat2_interp = np.asarray( [ kyhat2_half_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
kzhat2_interp = np.asarray( [ kzhat2_half_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )

kxhat4_interp = np.asarray( [ kxhat4_half_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
kyhat4_interp = np.asarray( [ kyhat4_half_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
kzhat4_interp = np.asarray( [ kzhat4_half_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )

kxhat6_interp = np.asarray( [ kxhat6_half_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
kyhat6_interp = np.asarray( [ kyhat6_half_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
kzhat6_interp = np.asarray( [ kzhat6_half_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )

# Now compute the "means on the box":
mu2_interp = rx2*kxhat2_interp + ry2*kyhat2_interp + rz2*kzhat2_interp
mu2 = np.mean(mu2_interp)
# Should be ~1/3.

mu4_interp = rx2**2 * kxhat4_interp + ry2**2 * kyhat4_interp + rz2**2 * kzhat4_interp
mu4 = np.mean(mu4_interp)
# Should be ~1/5.

mu6_interp = rx2**3 * kxhat6_interp + ry2**3 * kyhat6_interp + rz2**3 * kzhat6_interp
mu6 = np.mean(mu6_interp)
# Should be ~1/7.

b = np.asarray(bias)
f = matgrowcentral

# These are the "theory multipoles on the box"
mono_box_interp = np.zeros((ntracers,len(mu2_interp)))
quad_box_interp = np.zeros((ntracers,len(mu2_interp)))
for nt in range(ntracers):
    mono_box_interp[nt] = b[nt]**2 + 2*b[nt]*f * mu2_interp + f**2 * mu4_interp
    quad_box_interp[nt] = b[nt]**2 * (3*mu2_interp - 1.0 + 2*beta[nt]*( 3*mu4_interp - mu2_interp ) + beta[nt]**2*(3*mu6_interp - mu4_interp))

# Cross spectra -- used for the "theory" values on the box
crossmono_box_interp = np.zeros((ntracers*(ntracers-1)//2,len(mu2_interp)))
crossquad_box_interp = np.zeros((ntracers*(ntracers-1)//2,len(mu2_interp)))
index=0
for nt in range(ntracers):
	for ntp in range(nt+1,ntracers):
		crossmono_box_interp[index] = b[nt]*b[ntp] + (b[nt]+b[ntp])*f * mu2_interp + f**2 * mu4_interp
		crossquad_box_interp[index] = b[nt]*b[ntp] * (3*mu2_interp - 1.0 + (beta[nt]+beta[ntp])*( 3*mu4_interp - mu2_interp ) + beta[nt]*beta[ntp]*(3*mu6_interp - mu4_interp))
		index += 1

# Take means in an interval that is typically of interest
ks_interest = np.argsort(np.abs(k_interp - kph_central))
# Eliminate first few bandpowers, since many times they are contaminated by the window
# Also, get only the 15 closest bins to the central k_phys
ks_interest = ks_interest[ks_interest > 3][:15]
mono_box_mean = np.mean(mono_box_interp[:,ks_interest],axis=1)
quad_box_mean = np.mean(quad_box_interp[:,ks_interest],axis=1)

#crossmono_box_mean = np.mean(crossmono_box_interp[:,ks_interest],axis=1)
#crossquad_box_mean = np.mean(crossquad_box_interp[:,ks_interest],axis=1)


# There are several types of spectra corrections (biases,
# additional variances, etc.
#
# 1) Gaussian/Lognormal density field correspondence (ln_spec_corr).
#    This is a spectral distortion (an additional "bias"), but we will
#    also use to add an extra variance
#
# 2) Angle averages on a square box are NOT the same as 4\Pi angle integrals
#    This is more like an additional error bar.
#    These corrections are like A^\ell P(k) / P^\ell (k) ,
#    where A^\ell are the multipole moments computed on the square box,
#    involving only averages over <\mu^2>, <\mu^4> etc.,
#    and P^\ell (k) are the angle averages of the multipoles of the spectra.

# Corrections & Plots
colorsequence=['darkred','r','darkorange','goldenrod','y','yellowgreen','g','lightseagreen','c','deepskyblue','b','darkviolet','m']
jump=int(len(colorsequence)/ntracers)

# This will be S == P_camb / (P_ln / b^2)
spec_corrections = np.zeros((ntracers+1,len(k_interp)))
spec_corrections[0]=k_interp

# This will store the "real" monopole on the box, P^(r,0) == P_ln^(0) * S
mono_box_model = np.zeros((ntracers+1,len(k_interp)))
mono_box_model[0]=k_interp

mono_halos_box_model = np.zeros((ntracers+1,len(k_interp)))
mono_halos_box_model[0]=k_interp

# This will store the "real" quadrupole on the box, P^(r,2) == P_ln^(2) * S
quad_box_model = np.zeros((ntracers+1,len(k_interp)))
quad_box_model[0]=k_interp

# This will store the "theory" monopole on the box
mono_box_theory = np.zeros((ntracers+1,len(k_interp)))
mono_box_theory[0]=k_interp

# This will store the "theory" quadrupole on the box
quad_box_theory = np.zeros((ntracers+1,len(k_interp)))
quad_box_theory[0]=k_interp

# For the cross spectra, do not include vector of k's
crossmono_box_model = np.zeros((ntracers*(ntracers-1)//2,len(k_interp)))
crossquad_box_model = np.zeros((ntracers*(ntracers-1)//2,len(k_interp)))

crossmono_box_theory = np.zeros((ntracers*(ntracers-1)//2,len(k_interp)))
crossquad_box_theory = np.zeros((ntracers*(ntracers-1)//2,len(k_interp)))


cm_subsection = np.linspace(0, 1, ntracers)
mycolor = [ cm.jet(x) for x in cm_subsection ]

pl.loglog(k_interp,Pk_camb_interp,'k-')

for nt in range(ntracers):
    # Flatten, then order Pk_ln_unnorm
    Spec_ln_flat = ((Pk_ln_norm[nt].flatten())[k_order])
    Mono_ln_flat = ((Pk_ln_Mono_norm[nt].flatten())[k_order])
    Mono_ln_halos_flat = ((Pk_ln_halos_Mono_norm[nt].flatten())[k_order])
    Quad_ln_flat = ((Pk_ln_Quad_norm[nt].flatten())[k_order])

    # Combine in bins of k
    Spec_ln_interp = np.asarray( [ Spec_ln_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
    Mono_ln_interp = np.asarray( [ Mono_ln_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
    Quad_ln_interp = np.asarray( [ Quad_ln_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
    Mono_ln_halos_interp = np.asarray( [ Mono_ln_halos_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )

    # corrections
    spec_corrections[nt+1] = bias[nt]**2 * Pk_camb_interp/Spec_ln_interp
    # Where it deviates too much from 1, or is nan, substitute for 1.0
    spec_corrections[nt+1,np.isnan(spec_corrections[nt+1])] = 1.0
    spec_corrections[nt+1,np.abs(spec_corrections[nt+1])>5.0] = 1.0
    spec_corrections[nt+1,np.abs(spec_corrections[nt+1])<0.2] = 1.0
    # Smooth this function some more
    spec_corrections[nt+1] = ndimage.gaussian_filter1d(spec_corrections[nt+1],0.5,mode="reflect")
    # These are the "model" monopole and quadrupole
    mono_box_model[nt+1] = Mono_ln_interp * spec_corrections[nt+1]
    quad_box_model[nt+1] = Quad_ln_interp * spec_corrections[nt+1]
    mono_halos_box_model[nt+1] = Mono_ln_halos_interp * spec_corrections[nt+1]
    # Theory values for box monopole and quadrupole
    mono_box_theory[nt+1] = mono_box_interp[nt] * Pk_camb_interp
    quad_box_theory[nt+1] = quad_box_interp[nt] * Pk_camb_interp
    # Plot
    pl.loglog(k_interp, spec_corrections[nt+1]*Pk_camb_interp, color=mycolor[nt], linestyle=':')
    pl.loglog(k_interp,mono_box_model[nt+1],color=mycolor[nt],linestyle='-')
    pl.loglog(k_interp,mono_box_theory[nt+1],color=mycolor[nt],linestyle='--')
    pl.loglog(k_interp,mono_halos_box_model[nt+1],color=mycolor[nt],linestyle='-.')
    pl.loglog(k_interp,quad_box_model[nt+1],color=mycolor[nt],linestyle='-')
    pl.loglog(k_interp,quad_box_theory[nt+1],color=mycolor[nt],linestyle='--')

pl.title('Spectral corrections; theory monopoles and quadrupoles')
pl.xlim([2.0*k_min,0.75*k_max])
ylow=np.mean(np.abs(mono_box_model[1:,-10:-5]))*0.1*(0.1+matgrowcentral)
yhigh=np.mean(mono_box_model[1:,5:10])*10.0
pl.ylim([ylow,yhigh])
pl.savefig(dir_figs + '/spec_corrs_P0_P2_theory.png')
pl.close()


index=0
for nt in range(ntracers):
    Spec_ln_flat = ((Pk_ln_norm[nt].flatten())[k_order])
    Spec_ln_interp = np.asarray( [ Spec_ln_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
    spec_corrections[nt+1] = bias[nt]**2 * Pk_camb_interp/Spec_ln_interp
    # Where it deviates too much from 1, or is nan, substitute for 1.0
    spec_corrections[nt+1,np.isnan(spec_corrections[nt+1])] = 1.0
    spec_corrections[nt+1,np.abs(spec_corrections[nt+1])>5.0] = 1.0
    spec_corrections[nt+1,np.abs(spec_corrections[nt+1])<0.2] = 1.0
    # Smooth this function some more
    spec_corrections[nt+1] = ndimage.gaussian_filter1d(spec_corrections[nt+1],0.5,mode="reflect")
    # These are the "model" monopole and quadrupole
    for ntp in range(nt+1,ntracers):        
        crossmono_box_theory[index] = crossmono_box_interp[index] * Pk_camb_interp
        crossquad_box_theory[index] = crossquad_box_interp[index] * Pk_camb_interp
        Cross_Mono_ln_flat = ((Cross_Pk_ln_Mono_norm[index].flatten())[k_order])
        Cross_Quad_ln_flat = ((Cross_Pk_ln_Quad_norm[index].flatten())[k_order])
        Cross_Mono_ln_interp = np.asarray( [ Cross_Mono_ln_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
        Cross_Quad_ln_interp = np.asarray( [ Cross_Quad_ln_flat[k_sort_bins == i].mean() for i in range(1,k_max_index+1) ] )
        crossmono_box_model[index] = Cross_Mono_ln_interp * np.sqrt(spec_corrections[nt+1]*spec_corrections[ntp+1])
        crossquad_box_model[index] = Cross_Quad_ln_interp * np.sqrt(spec_corrections[nt+1]*spec_corrections[ntp+1])
        index += 1


np.savetxt(dir_specs + '/spec_corrections.dat',spec_corrections.T,fmt="%2.3f")
np.savetxt(dir_specs + '/monopole_model.dat',mono_box_model.T,fmt="%4.3f")
np.savetxt(dir_specs + '/quadrupole_model.dat',quad_box_model.T,fmt="%4.3f")
np.savetxt(dir_specs + '/monopole_theory.dat',mono_box_theory.T,fmt="%4.3f")
np.savetxt(dir_specs + '/quadrupole_theory.dat',quad_box_theory.T,fmt="%4.3f")

np.savetxt(dir_specs + '/cross_monopole_model.dat',crossmono_box_model.T,fmt="%4.3f")
np.savetxt(dir_specs + '/cross_quadrupole_model.dat',crossquad_box_model.T,fmt="%4.3f")

np.savetxt(dir_specs + '/cross_monopole_theory.dat',crossmono_box_theory.T,fmt="%4.3f")
np.savetxt(dir_specs + '/cross_quadrupole_theory.dat',crossquad_box_theory.T,fmt="%4.3f")

camb_save = np.array([k_interp,Pk_camb_interp]).T
np.savetxt(dir_specs + '/Pk_camb.dat',camb_save,fmt="%4.3f")



print('Done! Now writing maps of halos to files...')

# Now export maps
np.set_printoptions(precision=4)

maps_out = np.array(maps,dtype='int32')

for nm in range(n_maps):
    if len(str(nm))==1:
        map_num = '00' + str(nm)
    elif len(str(nm))==2:
        map_num = '0' + str(nm)
    else:
        map_num = str(nm)
    hdf5_map_file = dir_maps + '/Box_' + map_num + '_' + str(n_x) + '_' + str(n_y) + '_' + str(n_z) + '.hdf5'
    print('Writing file ', hdf5_map_file)
    h5f = h5py.File(hdf5_map_file,'w')
    #h5f.create_dataset(hdf5_map_file, data=maps_out[nt], dtype='int64')
    h5f.create_dataset('sims', data=maps_out[:,nm], dtype='int32',compression='gzip')
    h5f.close()

# Print diagnostics
b = np.asarray(bias)
f = matgrowcentral
theory_flatsky_mono = b**2 + 2./3.*b*f + 1./5*f**2
theory_flatsky_quad = 8./5.*(1./3.*b*f + 1./7.*f**2)

map_mono = np.average(mono_box_model[1:,ks_interest]/Pk_camb_interp[ks_interest],axis=1)
map_quad = np.average(quad_box_model[1:,ks_interest]/Pk_camb_interp[ks_interest],axis=1)

print( '################')
print('# Some diagnostics')
print('# Theory (flat sky) monopoles:', [ "{:2.4f}".format(x) for x in theory_flatsky_mono])
print('# Real monopoles of sim. maps:', [ "{:2.4f}".format(x) for x in map_mono])
print('# ')
print('# Theory (flat sky) quadrupoles:', [ "{:2.4f}".format(x) for x in theory_flatsky_quad])
print('# Real quadrupoles of sim. maps:', [ "{:2.4f}".format(x) for x in map_quad])
print('################')


# Save input parameters for bookkeeping
fileparams = dir_specs + '/sims_parameters.dat'

# Save regular file for records
params=np.array([ ['ntracers',str(ntracers)], ['tracerIDs',halos_ids], ['nbar',str(nbar)], ['bias',str(bias)] , ['zcentral',str(zcentral)], ['matgrow',str(matgrowcentral)] , ['adip',str(adip)] , ['cell_size',str(cell_size)] , ['nx,ny,nz',str(n_x),str(n_y),str(n_z)] ,['n_maps',str(n_maps)], ['Full sky, plane-parallel monopoles', str(theory_flatsky_mono)], ['Box volume monopoles', str(mono_box_mean)], ['Full sky, plane-parallel quadrupoles', str(theory_flatsky_quad)], ['Box volume quadrupoles', str(quad_box_mean)], ["Cosmological parameters"] , ["Omegak", str(Omegak)], ["w0", str(w0)] , ["w1", str(w1)] , ["Omegab", str(Omegab)] , ["Omegac", str(Omegac)] , ["H0", str(H0)] , ["n_SA", str(n_SA)] , ["ln10e10ASA", str(ln10e10ASA)] , ["z_re", str(z_re)] ] )
np.savetxt(fileparams,params,delimiter='  ',fmt='%s')


print('Done!')

