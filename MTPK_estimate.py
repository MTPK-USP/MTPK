#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
------------
Initial code started by Arthur E. da Mota Loureiro, 04/2015
Additonal changes by R. Abramo 07/2015, 02/2016
Added multi-tracer generalization and other changes, 03-12/2016
Added Halo Model generalization and other changes, 01/2018
Added cross-spectra -- Francisco Maion, 01/2020
------------
"""

#from __future__ import print_function
import numpy as np
import os, sys
import uuid


if sys.platform == "darwin":
    import pylab as pl
    from matplotlib import cm
else:
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib import pylab, mlab, pyplot
    from matplotlib import cm
    from IPython.display import display
    from IPython.core.pylabtools import figsize, getfigs
    pl=pyplot


# Add path to /inputs directory in order to load inputs
# Change as necessary according to your installation
this_dir = os.getcwd()
input_dir = this_dir + '/inputs'
sys.path.append(input_dir)

from time import time , strftime
from scipy import interpolate
from scipy import special
from scipy.optimize import leastsq
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

# My classes -- functions used by the MTPK suite
import fkp_multitracer as fkpmt
import fkp_class as fkp  # This is the new class, which computes auto- and cross-spectra
import gauss_pk_class as pkgauss
import pk_multipoles_gauss as pkmg
import pk_crossmultipoles_gauss as pkmg_cross
from camb_spec import camb_spectrum
from cosmo_funcs import matgrow, H

import h5py
import glob
import grid3D as gr

small=1.e-8



# Bias computed given the monopole and quadrupole.
# Returns a single value
def est_bias(m,q):
    denom = 60.*np.sqrt(3.)*q
    delta=np.sqrt(35.*m**2 + 10.*m*q - 7.*q**2)
    b0=10.*np.sqrt(7.)*m - 5.*np.sqrt(7.)*q + 2.*np.sqrt(5.)*delta
    b1=np.sqrt(35.*m + 5.*q - np.sqrt(35.)*delta)
    return b0*b1/denom

# Matter growth rate computed given the monopole and quadrupole.
# Returns a single value
def est_f(m,q):
    delta=np.sqrt(35.*m**2 + 10.*m*q - 7.*q**2)
    return 0.25*np.sqrt(7./3.)*np.sqrt(35*m+5*q-np.sqrt(35.)*delta)




#####################################################
# Load specs for this run
#####################################################
from MTPK import *


###################
print()
print()
print( 'This is the Multi-tracer power spectrum estimator')

print()
print('Handle of this run (fiducial spectra, biases, etc.): ', handle_estimates)
print()

# Load those properties
input_filename = glob.glob('inputs/*' + handle_estimates + '.py')
if (len(input_filename)==0) or (len(input_filename)>2) :
    print ('Input files not found -- or more than one with same handle in dir!')
    print ('Check on /inputs.  Aborting now...')
    sys.exit(-1)

exec ("from " + handle_estimates + " import *")

# Directory with data and simulated maps
dir_maps = 'maps/sims/' + handle_sims
# Directory with data
if (not sims_only) or (sel_fun_data):
    dir_data = 'maps/data/' + handle_data


# Will save results of the estimations to these directories:
dir_specs = 'spectra/' + handle_estimates
dir_figs = 'figures/' + handle_estimates
if not os.path.exists(dir_specs):
    os.makedirs(dir_specs)
if not os.path.exists(dir_figs):
    os.makedirs(dir_figs)


# Save estimations for each assumed k_phys in subdirectories named after k_phys
strkph = str(kph_central)

dir_specs += '/k=' + strkph
dir_figs  += '/k=' + strkph

# If directories do not exist, create them now...
if not os.path.exists(dir_specs):
    os.makedirs(dir_specs)
else:
    print ('Directory ', dir_specs, 'exists!')
    # print [name for name in os.listdir(dir_specs)]
    answer = input('Continue anyway? y/n  ')
    if answer!='y':
        print ('Aborting now...')
        sys.exit(-1)

if not os.path.exists(dir_figs):
    os.makedirs(dir_figs)
else:
    print ('Directory ', dir_figs , 'exists!')
    # print [name for name in os.listdir(dir_figs)]
    answer = input('Continue anyway? y/n  ')
    if answer!='y':
        print ('Aborting now...')
        sys.exit(-1)


########################## Some other cosmological quantities ######################################
Omegam = Omegac + Omegab
OmegaDE = 1. - Omegam - Omegak
h = H0/100.

cH = 299792.458*h/H(H0, Omegam, OmegaDE, w0, w1, zcentral)  # c/H(z) , in units of h^-1 Mpc

try:
    gamma
except:
    gamma = 0.55

try:
    matgrowcentral
except:
    matgrowcentral = matgrow(Omegam,OmegaDE,w0,w1,zcentral,gamma)
else:
    print('ATTENTION: pre-defined (on input) matter growth rate =' , matgrowcentral)

# Velocity dispersion. vdisp is defined on inputs with units of km/s
vdisp = np.asarray(vdisp) #km/s
sigma_v = vdisp/H(100,Omegam,OmegaDE,-1,0.0,zcentral) #Mpc/h
a_vdisp = vdisp/c #Adimensional vdisp

# Redshift errors. sigz_est is defined on inputs, and is adimensional
sigz_est = np.asarray(sigz_est)
sigma_z = sigz_est*c/H(100,Omegam,OmegaDE,-1,0.0,zcentral) # Mpc/h

# Joint factor considering v dispersion and z error
sig_tot = np.sqrt(sigma_z**2 + sigma_v**2) #Mpc/h
a_sig_tot = np.sqrt(sigz_est**2 + a_vdisp**2) #Adimensional sig_tot

###################################################################################################

#############Calling CAMB for calculations of the spectra#################
print('Beggining CAMB calculations\n')

# It is strongly encouraged to use k_min >= 1e-4, since it is a lot faster
try:
    k_min_camb
    k_max_camb
except:
    k_min_camb = 1.0e-4
    k_max_camb = 1.0e1

nklist = 1000
k_camb = np.logspace(np.log10(k_min_camb),np.log10(k_max_camb),nklist)

kc, pkc = camb_spectrum(H0, Omegab, Omegac, w0, w1, z_re, zcentral, n_SA, k_min_camb, k_max_camb, whichspec)
Pk_camb = np.asarray( np.interp(k_camb, kc, pkc) )
############# Ended CAMB calculation #####################################

try:
    power_low
except:
    pass
else:
    Pk_camb = power_low*np.power(Pk_camb,pk_power)


# Construct spectrum that decays sufficiently rapidly, and interpolate
k_interp = np.append(k_camb,np.array([2*k_camb[-1],4*k_camb[-1],8*k_camb[-1],16*k_camb[-1]]))
P_interp = np.append(Pk_camb,np.array([1./4.*Pk_camb[-1],1./16*Pk_camb[-1],1./64*Pk_camb[-1],1./256*Pk_camb[-1]]))
pow_interp=interpolate.PchipInterpolator(k_interp,P_interp)


#####################################################
#####################################################
#####################################################

# In the power spectrum estimation code we have two options:
# (a) work directly with the halos, both "data" and simulations --> do_galaxies = False
# (b) combine halos into galaxies assuming some HOD --> do_galaxies = True
#
# The main difference is that, when using halos directly, we do not re-sample the original halo maps
if do_galaxies:
    print ("Attention: forming galaxy mocks from a mass function, HOD & halo bias.")
    try:
        hod = np.loadtxt("inputs/" + hod_file)
        hod = np.reshape(hod,(ngals,nhalos))
        #if len(hod.shape) < 2:
        #    hod = np.eye(nhalos)
        mass_fun = mult_sel_fun*np.loadtxt("inputs/" + mass_fun_file)
        halo_bias = np.loadtxt("inputs/" + halo_bias_file)
        # N.B.: the mass function is given in units of h^3 Mpc^-3 !
        # nbar below is in cell units
        halo_densities_cell = np.asarray(mass_fun)*(cell_size)**3

        # N.B.: these number densities (in the mass function) are in units of h^3 Mpc^-3!
        # Notice that the HOD itself is adimensional.
        gal_densities = np.dot(hod,mass_fun)
        gal_densities_cell = gal_densities*(cell_size)**3
        # One-halo term in units of h^-3 Mpc^3. This takes into account galaxy Poisson sampling.
        # Can also be given in terms of the HOD, or defined in the inputs
        try:
            gal_p1h
            p1h = np.asarray(gal_p1h)
        except:
            p1h = np.dot(hod**2,mass_fun)/(np.dot(hod,mass_fun))**2

        # HODs for the density contrast -- notice that Tr(hod_delta) = 1
        hod_delta = ((hod*mass_fun).T/np.dot(hod,mass_fun)).T
        gal_bias = np.asarray(halo_bias)
        gal_bias = np.dot(hod,gal_bias*mass_fun)/np.dot(hod,mass_fun)
        gal_adip = np.asarray(adip)
        gal_adip = np.dot(hod,gal_adip*mass_fun)/np.dot(hod,mass_fun)
        gal_sigz_est = np.asarray(sigz_est)
        gal_sigz_est = np.sqrt( np.dot(hod,gal_sigz_est**2*mass_fun)/np.dot(hod,mass_fun) )
        gal_vdisp = np.asarray(vdisp)
        gal_vdisp = np.sqrt( np.dot(hod,gal_vdisp**2*mass_fun)/np.dot(hod,mass_fun) )

        a_gal_sig_tot = np.sqrt((gal_vdisp/c)**2 + gal_sigz_est**2)
    except:
        print ("Something's wrong... did not find HOD and/or mass function!")
        print ("Check in the /inputs directory. Aborting now...")
        sys.exit(-1)
else:
    print ("Using fiducial biases for the tracers directly from the input file:")
    try:
        mass_fun = mult_sel_fun * np.loadtxt("inputs/" + mass_fun_file)
        halo_bias = np.loadtxt("inputs/" + halo_bias_file)
        bias = np.asarray(halo_bias)
        gal_bias = np.asarray(halo_bias)
        gal_adip = np.asarray(adip)
        gal_sigz_est = np.asarray(sigz_est)
        gal_vdisp = np.asarray(vdisp)

        a_gal_sig_tot = np.sqrt((gal_vdisp/c)**2 + gal_sigz_est**2)
    except:
        print()
        print ("Could not find halo mass function or/and halo bias values from input files!")
        print ("Aborting now...")
        sys.exit(-1)



#####################################################
# Generate real- and Fourier-space grids for FFTs
#####################################################
# print 'Generating the k-space Grid...'
L_x = n_x*cell_size ; L_y = n_y*cell_size ; L_z = n_z*cell_size
grid = gr.grid3d(n_x,n_y,n_z,L_x,L_y,L_z)


# Galaxy selection function
# If given by data
if sel_fun_data:
    try:
        h5map = h5py.File(dir_data + '/' + sel_fun_file,'r')
        h5data = h5map.get(list(h5map.keys())[0])
        n_bar_matrix_fid = np.asarray(h5data,dtype='float32')
        n_bar_matrix_fid = np.asarray(mult_sel_fun*(n_bar_matrix_fid + shift_sel_fun),dtype='float32')
        h5map.close
    except:
        print ('Could not find file with data selection function!')
        print ('Check your directory ', dir_data)
        print ('Aborting now...')
        sys.exit(-1)
    if (np.shape(n_bar_matrix_fid)[1] != n_x) or (np.shape(n_bar_matrix_fid)[2] != n_y) or (np.shape(n_bar_matrix_fid)[3] != n_z):
        print ('WARNING!!! Dimensions of data selection function box =', n_bar_matrix_fid.shape, ' , differ from input file!')
        print ('Please correct/check input files and/or maps. Aborting now.')
        sys.exit(-1)
    # Normalize the galaxy densities from the HODs with the galaxy densities from the data selection function
    if do_galaxies:
    # Check if everything is OK
        if (len(gal_bias) != ngals) or (len(gal_adip) != ngals):
            print ('There should be values of bias & adip for each tracers; ngals= ' , ngals, ' galaxy types!')
            print ('Please correct/check input files, HOD file or maps. Aborting now.')
            sys.exit(-1)
        hod_3D = np.zeros((ngals,nhalos,n_x,n_y,n_z),dtype='float32')
        for ng in range(ngals):
            for nh in range(nhalos):
                hod_3D[ng,nh] = np.asarray(hod[ng,nh]/gal_densities_cell[ng]*n_bar_matrix_fid[ng],dtype='float32')
        bias = gal_bias
else:
    if do_galaxies:
        # Use analytical fit for selection function, from selection_function.py
        from selection_function_multitracer import *
        print ("Att.: using analytical selection function for galaxies (check parameters in input file).")
        nbar = mult_sel_fun*gal_densities_cell
        #if (len(bias) != ngals) or (len(nbar) != ngals) or (len(ncentral) != ngals) or (len(nsigma) != ngals) or (len(bias) != ngals) or (len(adip) != ngals):
        if (len(gal_bias) != ngals) or (len(nbar) != ngals) or (len(gal_bias) != ngals) or (len(gal_adip) != ngals):
            print ('Inputs (bias, sel. function parameters, etc.) must all have lengths equal to the number of tracers = ' , ngals, ' !')
            print ('Please correct/check input file and/or maps. Aborting now.')
            sys.exit(-1)
        hod_3D = np.zeros((ngals,nhalos,n_x,n_y,n_z),dtype='float32')
        n_bar_matrix_fid = np.zeros((ngals,n_x,n_y,n_z),dtype='float32')
        for ng in range(ngals):
            for nh in range(nhalos):
                hod_3D[ng,nh] = mult_sel_fun*np.asarray(hod[ng,nh]/nbar[ng] * selection_func(grid.grid_r, nbar[ng],ncentral[ng],nsigma[ng]),dtype='float32')
                n_bar_matrix_fid[ng] += hod_3D[ng,nh]*halo_densities_cell[nh]
    else:   # this case is for when we do halos directly, and don't use a selection function
        try:
            mass_fun = mult_sel_fun*np.asarray(np.loadtxt("inputs/" + mass_fun_file))
            halo_bias = np.asarray(np.loadtxt("inputs/" + halo_bias_file))
            bias = halo_bias
            print ("Using halo mass function and halo bias from input files")
        except:
            print ("Could not find either halo mass function or/and halo bias files on inputs/ !")
            print ("Please check. Aborting now...")
            sys.exit(-1)
        # N.B.: the mass function is given in units of h^3 Mpc^-3 !
        # nbar below is in cell units
        halo_densities_cell = np.asarray(mass_fun)*(cell_size)**3
        nbar = mult_sel_fun*np.asarray(mass_fun)*(cell_size)**3
        print ("Att.: using constant selection function for halos, with n_bar(halos) =", nbar)
        n_bar_matrix_fid = np.zeros((nhalos,n_x,n_y,n_z),dtype='float32')
        for nh in range(nhalos):
            n_bar_matrix_fid[nh] = nbar[nh]*np.ones((n_x,n_y,n_z))
 


if sims_only:
    mapnames_sims = sorted(glob.glob(dir_maps + '/*.hdf5'))
    if len(mapnames_sims)==0 :
        print()
        print ('Simulated map files not found! Check input simulation files.')
        print ('Exiting program...')
        print ()
        sys.exit(-1)
    if len(mapnames_sims) != n_maps :
        print ('You are using', n_maps, ' mocks out of the existing', len(mapnames_sims))
    answer = input('Continue anyway? y/n  ')
    if answer!='y':
        print ('Aborting now...')
        sys.exit(-1)
    print ('Will use the N =', n_maps, ' simulation-only maps contained in directory', dir_maps)
else:
    mapnames_data = glob.glob(dir_data + '/*DATA.hdf5')
    if len(mapnames_data) > 1 :
        print()
        print( 'Only one DATA map file supported. Check /inputs and /maps.')
        print( 'Exiting program...')
        print ()
        sys.exit(-1)
    mapnames_sims = sorted(glob.glob(dir_maps + '/*.hdf5'))
    if (len(mapnames_sims)==0) or (len(mapnames_data)==0) :
        print()
        print ('SIMS or DATA map files not found! Check MTPK.py, /inputs and /maps.')
        print ('Exiting program...')
        print()
        sys.exit(-1)

    mapnames_data = mapnames_data[0]
    print ('Will use the data maps contained in the file:')
    print ('     ' + mapnames_data)
    print ('and the N =', n_maps , ' simulated maps contained contained in directory', dir_maps)
    # read data box (all galaxy types)
    h5map = h5py.File(mapnames_data,'r')
    data_maps = np.asarray(h5map.get(list(h5map.keys())[0]))
    h5map.close
    if not data_maps.shape == (ngals,n_x,n_y,n_z):
        print ('Unexpected shape of data maps:', data_maps.shape)
        print ('Please check again. Aborting now...')
        sys.exit(-1)


## !! NEW !! Low-cell-count threshold. Will apply to data AND to mocks
## We will treat this as an additional MASK (thresh_mask) for data and mocks
try:
    cell_low_count_thresh
    thresh_mask = np.ones_like(n_bar_matrix_fid)
    thresh_mask[n_bar_matrix_fid < cell_low_count_thresh] = 0.0
    n_bar_matrix_fid = thresh_mask * n_bar_matrix_fid
except:
    pass


print ()
print ('Geometry: (nx,ny,nz) = (' +str(n_x)+','+str(n_y)+','+str(n_z)+'),  cell_size=' + str(cell_size) + ' h^-1 Mpc')


print()
if whichspec == 0:
    print ('Using LINEAR power spectrum from CLASS')
elif whichspec == 1:
    print ('Using power spectrum from CLASS + HaloFit')
else:
    print ('Using power spectrum from CLASS + HaloFit with PkEqual')

print()
print ('----------------------------------')
print()




#####################################################
# Start computing physical sizes of boxes
#####################################################
box_vol = L_x*L_y*L_z            # Box's volume
L_max = np.sqrt(L_x*L_x + L_y*L_y + L_z*L_z)    

##########################################
#  Generating the Bins Matrix M^a_{ijl}
#  The matrix MR maps the \vec{k}'s into bins (k,mu)
#  The matrix MRk maps the \vec{k}'s into bins (k)
##########################################

#  Fundamental frequencies (NOT ANGULAR FREQUENCIES) of the grid
#  Here nn, kk_bar, etc., are in units of the grid, for which cell size == 1

#R NOTE ON CONVERSION OF k's (which here are FREQUENCIES) to PHYSICAL k:
#R
#R   k_phys = 2*pi/cell_size * frequency
#R
nn = int(np.sqrt(n_x**2 + n_y**2 + n_z**2))
kk_bar = np.fft.fftfreq(nn)


### K_MAX_MIN
#  Maximum ***frequency*** allowed
#  Nyquist frequency is 0.5 (in units of 1/cell)
try:
    kmax_phys
    kmaxbar = min(0.5,kmax_phys*cell_size/2.0/np.pi)
    kmax_phys = kmaxbar*2*np.pi/cell_size
except:
    kmax_phys = 0.5 # in h/Mpc
    kmaxbar = min(0.4,kmax_phys*cell_size/2.0/np.pi)
    kmax_phys = kmaxbar*2*np.pi/cell_size

# The number of bins should be set by the maximum k to be computed,
# together with the minimum separation based on the physical volume.
#
# Typically, dk_phys =~ 1.4/L , where L is the typical physical size
# of the survey (Abramo 2012)
# For some particular applications (esp. large scales) we can use larger bins
#dk0=1.4/np.power(n_x*n_y*n_z,1/3.)/(2.0*np.pi)
dk0 = 3.0/np.power(n_x*n_y*n_z,1/3.)/(2.0*np.pi)
dk_phys = 2.0*np.pi*dk0/cell_size

# Ensure that the binning is at least a certain size
dk_phys = max(dk_phys,dkph_bin)
dk0 = dk_phys*cell_size/2.0/np.pi

#  Physically, the maximal useful k is perhaps k =~ 0.3 h/Mpc (non-linear scale)
np.set_printoptions(precision=3)

print ('Will estimate modes up to k[h/Mpc] = ', '%.4f'% kmax_phys,' in bins with Delta_k =', '%.4f' %dk_phys)

print()
print ('----------------------------------')
print()

#R This line makes some np variables be printed with less digits
np.set_printoptions(precision=6)


#R Here are the k's that will be estimated (in grid units):
kgrid = grid.grid_k
kminbar = 1./4.*(kgrid[1,0,0]+kgrid[0,1,0]+kgrid[0,0,1]) + dk0/4.0

### K_MAX_MIN
try:
    kmin_phys
    kminbar = kmin_phys*cell_size/2.0/np.pi
except:
    pass

### K_MAX_MIN
num_binsk=np.int((kmaxbar-kminbar)/dk0)
dk_bar = dk0*np.ones(num_binsk)
k_bar = kminbar + dk0*np.arange(num_binsk)
r_bar = 1/2.0 + ((1.0*n_x)/num_binsk)*np.arange(num_binsk)


# Some nomenclature about outputs:
# k in physical units
kph = k_bar*2*np.pi/cell_size



#####################################################
# Import the map (s), and if necessary reshape it
#####################################################


# Define the "effective bias" as the amplitude of the monopole
try:
    kdip_phys
except:
    kdip_phys = 1./(cell_size*(n_z_orig + n_z/2.))
else:
    print ('ATTENTION: pre-defined (on input) alpha-dipole k_dip [h/Mpc]=', '%1.4f'%kdip_phys)

dip = np.asarray(adip) * kdip_phys
pk_mg = pkmg.pkmg(halo_bias,dip,matgrowcentral,k_camb,len(bias)*[0],a_sig_tot,cH,zcentral)

monopoles = pk_mg.mono
quadrupoles = pk_mg.quad

try:
    pk_mg_cross = pkmg_cross.pkmg_cross(halo_bias,dip,matgrowcentral,k_camb,len(bias)*[0],a_sig_tot,cH,zcentral)
    cross_monopoles = pk_mg_cross.monos
    cross_quadrupoles = pk_mg_cross.quads
except:
    cross_monopoles = np.zeros((len(k_camb),1))
    cross_quadrupoles = np.zeros((len(k_camb),1))


# Compute effective dipole and bias of tracers
where_kph_central = np.argmin(np.abs(k_camb - kph_central))

effadip = dip*matgrowcentral/(0.00000000001 + kph_central)
effbias = np.sqrt(monopoles[:,where_kph_central])

try:
    data_bias
    pk_mg_data = pkmg.pkmg(data_bias,dip,matgrowcentral,k_camb,len(bias)*[0],a_sig_tot,cH,zcentral)
    monopoles_data = pk_mg_data.mono
    effbias_data = np.sqrt(monopoles_data[:,where_kph_central])
except:
    pass



# Import correction factors for the lognormal halo simulations
# Use these to correct the spectra of the lognormal sims
# halo_spec_corr are the corrections in REAL (not redshift) space
dir_spec_corr_sims = 'spectra/' + handle_sims
try:
    Pk_camb_sim = np.loadtxt(dir_spec_corr_sims + '/Pk_camb.dat')[:,1]
    halo_spec_corr = np.loadtxt(dir_spec_corr_sims + '/spec_corrections.dat')
    halo_mono_model = np.loadtxt(dir_spec_corr_sims + '/monopole_model.dat')
    halo_quad_model = np.loadtxt(dir_spec_corr_sims + '/quadrupole_model.dat')
    halo_mono_theory = np.loadtxt(dir_spec_corr_sims + '/monopole_theory.dat')
    halo_quad_theory = np.loadtxt(dir_spec_corr_sims + '/quadrupole_theory.dat')
    halo_crossmono_model = np.loadtxt(dir_spec_corr_sims + '/cross_monopole_model.dat')
    halo_crossquad_model = np.loadtxt(dir_spec_corr_sims + '/cross_quadrupole_model.dat')
    halo_crossmono_theory = np.loadtxt(dir_spec_corr_sims + '/cross_monopole_theory.dat')
    halo_crossquad_theory = np.loadtxt(dir_spec_corr_sims + '/cross_quadrupole_theory.dat')
    k_corr = halo_spec_corr[:,0]
    nks = len(k_corr)
    if len(halo_crossmono_model.shape) == 1:
        halo_crossmono_model = np.reshape(halo_crossmono_model, (nks, 1))
        halo_crossquad_model = np.reshape(halo_crossquad_model, (nks, 1))
        halo_crossmono_theory = np.reshape(halo_crossmono_theory, (nks, 1))
        halo_crossquad_theory = np.reshape(halo_crossquad_theory, (nks, 1))

except:
    print()
    print ("Did not find spectral corrections and theory spectra on directory:")
    print (dir_spec_corr_sims)
    print ("[Sometimes these files are created by the lognormal map-creating tool.]")
    print ("Will assume spectral corrections are all unity, in the interval k_phys: [0,1],")
    print ("and that monopoles and quadrupoles are from linear bias + RSD model, with CAMB spectrum.")
    print()
    k_corr = k_camb
    nks = len(k_camb)
    halo_spec_corr = np.ones((nks,nhalos+1))
    halo_mono_model = np.ones((nks,nhalos+1))
    halo_quad_model = np.ones((nks,nhalos+1))
    halo_mono_theory = np.ones((nks,nhalos+1))
    halo_quad_theory = np.ones((nks,nhalos+1))
    halo_mono_model[:,0]=k_camb
    halo_quad_model[:,0]=k_camb
    halo_mono_theory[:,0]=k_camb
    halo_mono_theory[:,0]=k_camb
    halo_crossmono_model = np.ones((nks,nhalos*(nhalos-1)//2))
    halo_crossquad_model = np.ones((nks,nhalos*(nhalos-1)//2))
    halo_crossmono_theory = np.ones((nks,nhalos*(nhalos-1)//2))
    halo_crossquad_theory = np.ones((nks,nhalos*(nhalos-1)//2))

    index=0
    for nt in range(nhalos):
        halo_mono_model[:,nt+1]= monopoles[nt]*Pk_camb
        halo_quad_model[:,nt+1]= quadrupoles[nt]*Pk_camb
        halo_mono_theory[:,nt+1]= monopoles[nt]*Pk_camb
        halo_quad_theory[:,nt+1]= quadrupoles[nt]*Pk_camb
        for ntp in range(nt+1,nhalos):
            halo_crossmono_model[:,index] = cross_monopoles[index]*Pk_camb
            halo_crossmono_theory[:,index] = cross_monopoles[index]*Pk_camb
            halo_crossquad_model[:,index] = cross_quadrupoles[index]*Pk_camb
            halo_crossquad_theory[:,index] = cross_quadrupoles[index]*Pk_camb
            index += 1


# Discard the first column of the auto-spectra ("model" and "theory"), since they are simply the values of k
halo_spec_corr = halo_spec_corr[:,1:]
halo_mono_model = halo_mono_model[:,1:]
halo_quad_model = halo_quad_model[:,1:]
halo_mono_theory = halo_mono_theory[:,1:]
halo_quad_theory = halo_quad_theory[:,1:]

# NOW INCLUDING CROSS-CORRELATIONS
all_halo_monopoles_model = np.zeros((nks,nhalos,nhalos))
all_halo_quadrupoles_model = np.zeros((nks,nhalos,nhalos))
all_halo_monopoles_theory = np.zeros((nks,nhalos,nhalos))
all_halo_quadrupoles_theory = np.zeros((nks,nhalos,nhalos))

index=0
for i in range(nhalos):
    all_halo_monopoles_model[:,i,i] = halo_mono_model[:,i]
    all_halo_monopoles_theory[:,i,i] = halo_mono_theory[:,i]
    all_halo_quadrupoles_model[:,i,i] = halo_quad_model[:,i]
    all_halo_quadrupoles_theory[:,i,i] = halo_quad_theory[:,i]
    for j in range(i+1,nhalos):
        all_halo_monopoles_model[:,i,j] = halo_crossmono_model[:,index]
        all_halo_monopoles_theory[:,i,j] = halo_crossmono_theory[:,index]
        all_halo_quadrupoles_model[:,i,j] = halo_crossquad_model[:,index]
        all_halo_quadrupoles_theory[:,i,j] = halo_crossquad_theory[:,index]
        all_halo_monopoles_model[:,j,i] = halo_crossmono_model[:,index] 
        all_halo_monopoles_theory[:,j,i] = halo_crossmono_theory[:,index] 
        all_halo_quadrupoles_model[:,j,i] = halo_crossquad_model[:,index]
        all_halo_quadrupoles_theory[:,j,i] = halo_crossquad_theory[:,index]
        index += 1


if do_galaxies:
    spec_corr = np.zeros((nks,ngals))

    mono_model = np.zeros((nks,ngals))
    quad_model = np.zeros((nks,ngals))
    mono_theory = np.zeros((nks,ngals))
    quad_theory = np.zeros((nks,ngals))

    cross_mono_model = np.zeros((nks,ngals*(ngals-1)//2))
    cross_quad_model = np.zeros((nks,ngals*(ngals-1)//2))
    cross_mono_theory = np.zeros((nks,ngals*(ngals-1)//2))
    cross_quad_theory = np.zeros((nks,ngals*(ngals-1)//2))

    mat_pspec = np.interp(k_corr,k_camb,Pk_camb)

    spec_corr = (np.dot(hod_delta,np.sqrt(halo_spec_corr).T).T)**2

    all_mono_model = np.zeros((nks,ngals,ngals))
    all_mono_theory = np.zeros((nks,ngals,ngals))
    all_quad_model = np.zeros((nks,ngals,ngals))
    all_quad_theory = np.zeros((nks,ngals,ngals))

    all_mono_model = np.einsum('ij,kjl',hod_delta,all_halo_monopoles_model)
    all_mono_model = np.einsum('ij,lkj',hod_delta,all_mono_model)
    all_mono_model = np.swapaxes(all_mono_model,0,1)

    all_mono_theory = np.einsum('ij,kjl',hod_delta,all_halo_monopoles_model)
    all_mono_theory = np.einsum('ij,lkj',hod_delta,all_mono_theory)
    all_mono_theory = np.swapaxes(all_mono_theory,0,1)

    all_quad_model = np.einsum('ij,kjl',hod_delta,all_halo_quadrupoles_model)
    all_quad_model = np.einsum('ij,lkj',hod_delta,all_quad_model)
    all_quad_model = np.swapaxes(all_quad_model,0,1)

    all_quad_theory = np.einsum('ij,kjl',hod_delta,all_halo_quadrupoles_model)
    all_quad_theory = np.einsum('ij,lkj',hod_delta,all_quad_theory)
    all_quad_theory = np.swapaxes(all_quad_theory,0,1)

    index = 0
    for i in range(ngals):
        for j in range(i+1,ngals):
            cross_mono_model[:,index] = all_mono_model[:,i,j]
            cross_mono_theory[:,index] = all_mono_theory[:,i,j]
            cross_quad_model[:,index] = all_quad_model[:,i,j]
            cross_quad_theory[:,index] = all_quad_theory[:,i,j]
            index += 1
else:

    mono_model = np.zeros((nks,nhalos))
    quad_model = np.zeros((nks,nhalos))
    mono_theory = np.zeros((nks,nhalos))
    quad_theory = np.zeros((nks,nhalos))

    cross_mono_model = halo_crossmono_model
    cross_quad_model = halo_crossquad_model
    cross_mono_theory = halo_crossmono_theory
    cross_quad_theory = halo_crossquad_theory

    spec_corr = halo_spec_corr
    all_mono_model = all_halo_monopoles_model
    all_mono_theory = all_halo_monopoles_theory
    all_quad_model = all_halo_quadrupoles_theory
    all_quad_theory = all_halo_quadrupoles_theory

# Now define which "tracers" we will use: galaxies, or halos
if do_galaxies:
    ntracers = ngals
else:
    ntracers = nhalos

for ng in range(ntracers):
    mono_model[:,ng] = all_mono_model[:,ng,ng]
    quad_model[:,ng] = all_quad_model[:,ng,ng]
    mono_theory[:,ng] = all_mono_theory[:,ng,ng]
    quad_theory[:,ng] = all_quad_theory[:,ng,ng]


# Get effective bias (sqrt of monopoles) for final tracers
pk_mg = pkmg.pkmg(gal_bias,gal_adip,matgrowcentral,k_camb,len(bias)*[0],a_gal_sig_tot,cH,zcentral)

monopoles = pk_mg.mono
quadrupoles = pk_mg.quad

# Compute effective dipole and bias of tracers
where_kph_central = np.argmin(np.abs(k_camb - kph_central))

effadip = gal_adip*matgrowcentral/(0.00000000001 + kph_central)
effbias = np.sqrt(monopoles[:,where_kph_central])

try:
    data_bias
    pk_mg_data = pkmg.pkmg(data_bias,gal_adip,matgrowcentral,k_camb,len(bias)*[0],a_gal_sig_tot,cH,zcentral)
    monopoles_data = pk_mg_data.mono
    effbias_data = np.sqrt(monopoles_data[:,where_kph_central])
except:
    pass



k_spec_corr = np.append(np.append(0.,k_corr),k_corr[-1] + dkph_bin )
pk_ln_spec_corr = np.vstack((np.vstack((np.asarray(spec_corr[0]),np.asarray(spec_corr))),spec_corr[-1]))

pk_ln_mono_model = np.vstack((np.vstack((np.asarray(mono_model[0]),np.asarray(mono_model))),mono_model[-1]))
pk_ln_quad_model = np.vstack((np.vstack((np.asarray(quad_model[0]),np.asarray(quad_model))),quad_model[-1]))

pk_ln_mono_theory = np.vstack((np.vstack((np.asarray(mono_theory[0]),np.asarray(mono_theory))),mono_theory[-1]))
pk_ln_quad_theory = np.vstack((np.vstack((np.asarray(quad_theory[0]),np.asarray(quad_theory))),quad_theory[-1]))

cross_mono_model = np.vstack((np.vstack((np.asarray(cross_mono_model[0]),np.asarray(cross_mono_model))),cross_mono_model[-1]))
cross_mono_theory = np.vstack((np.vstack((np.asarray(cross_mono_theory[0]),np.asarray(cross_mono_theory))),cross_mono_theory[-1]))

cross_quad_model = np.vstack((np.vstack((np.asarray(cross_quad_model[0]),np.asarray(cross_quad_model))),cross_quad_model[-1]))
cross_quad_theory = np.vstack((np.vstack((np.asarray(cross_quad_theory[0]),np.asarray(cross_quad_theory))),cross_quad_theory[-1]))






###############################
#R
#R  Now let's start the construction of the bin matrices, MR and MRk.
#R
#R  * MR accounts for the modes that should be averaged to obtain a bin in (k,\mu)
#R  * MRk accounts for the modes that should be averaged to obtain a bin in (k)
#R
#R  Hence, MR gives the RSD bin matrix; MRk gives the bin matrix for the monopole P_0(k).
#R
#R  In order to minimize memory usage and computation time,
#R  we employ sparse matrix methods.

#R  First, construct the flattened array of |k|, kflat .
#R  NOTICE THAT kflat is in GRID units -- must divide by cell_size eventually
#R  (Remembering that 1/2 of the grid is redundant, since the maps are real-valued)
#RR kflat=np.ndarray.flatten(grid.grid_k[:,:,:n_z/2-1])

kflat=(np.ndarray.flatten(kgrid[:,:,:n_z//2+1]))
lenkf=len(kflat)


print ('Central physical k values where spectra will be estimated:', kph_central)

# Get G(z)^2*P(k_central) for the central value of k and central value of z
kcmin = kph_central - 2.0*np.pi*( 4.0*dk0 )/cell_size
kcmax = kph_central + 2.0*np.pi*( 4.0*dk0 )/cell_size
# This will be the value used in the FKP and MT weights
powercentral = np.mean( Pk_camb[ (k_camb > kcmin) & (k_camb < kcmax) ] )

# Theory power spectra, interpolated on the k_physical used for estimations
powtrue = np.interp(kph,k_camb,Pk_camb)
pow_bins = len(kph)




################################################################
#R   Initialize the sparse matrices MR and MRk
################################################################
#R
#R Entries of these matrices are 0 or 1.
#R Use dtype=int8 to keep size as small as possible.
#R
#R Each row of those matrices corresponds to a (k,mu) bin, or to a (k) bin
#R The columns are the values of the flattened array kflat=|k|=|(kx,ky,kz)|
#R
#R The entries of the MR/MRk matrices are:
#R   1 when the value of (k,mu)/(k) belongs to the bin
#R   0 if it does not

print ('Initializing the k-binning matrix...')
tempor = time()

#R Initialize first row of the M-matrices (vector of zeros)
MRline = np.zeros(lenkf,dtype=np.int8)

#R These are the sparse matrices, built in the fastest way:
MRkl = coo_matrix([MRline] , dtype=np.int8)

#R And these are the matrices in the final format (easiest to make computations):
MRk = csc_matrix((num_binsk,lenkf),dtype=np.int8)

#R Now build the M-matrix by stacking the rows for each bin in (mu,k)

for ak in range(0,num_binsk):
    #R Now for the M-matrix that applies only for k bins (not mu), and for r bins:
    MRline[ np.where(  (kflat >= k_bar[ak] - dk0/2.00) & \
                     (kflat < k_bar[ak] + dk0/2.00) ) ] = 1
    MRline[ np.isnan(MRline) ] = 0
    # stack the lines to construct the new M matrix
    MRkl = vstack([MRkl,coo_matrix(MRline)], dtype=np.int8)
    MRline = np.zeros(lenkf,dtype=np.int8)

#R The way the matrix MRk was organized is such that
#R each rows corresponds to the kflats that belong to the bins:
#R (null), (k[0]), (k[1]), ... , (k[-1])
#R
######################################################################
#R ATTENTION! The first lines of the matrix MRk is ALL ZEROS,
#R and should be DISCARDED!
######################################################################

#R Convert MR and MRk matrices to csr format
Mknew=MRkl.tocsr()
MRklr=vstack(Mknew[1:,:])

#R Now convert the matrix to csc format, which is fastest for computations
MRk = MRklr.tocsc()

# Delete the initial sparse matrices to save memory
MRkl = None
Mknew = None
MRklr = None
del MRkl
del Mknew

# Counts in each bin
kkbar_counts = MRk.dot(np.ones(lenkf))

print ('Done with k-binning matrices. Time cost: ', np.int((time()-tempor)*1000)/1000., 's')
print ('Memory occupied by the binning matrix: ', MRk.nnz)

#print('Bin counts in k, using M(k):')
#print(kkbar_counts[10:15])

#R We defined "target" k's , but <k> on bins may be slightly different
kkav=(((MRk*kflat))/(kkbar_counts+0.00001))
print ('Originally k_bar was defined as:', [ "{:1.4f}".format(x) for x in k_bar[10:16:2] ])
print ('The true mean of k for each bin is:', [ "{:1.4f}".format(x) for x in kkav[10:16:2] ])
print()
print ('----------------------------------')
print()
print ('Now estimating the power spectra...')

######################################################################
#R    FKP of the data to get the P_data(k) -- using MR and MRk
#R
#R  N.B.: In this code, we do this n_maps times -- once for each map, independently.
#R  Also, we make 4 estimates, for 4 "bandpower ranges" between k_min and k_max .
######################################################################




# Theory monopoles of spectra (as realized on the rectangular box)
pk_ln_spec_corr_kbar=np.zeros((ntracers,pow_bins))
P0_theory=np.zeros((ntracers,pow_bins))
P2_theory=np.zeros((ntracers,pow_bins))
P0_model=np.zeros((ntracers,pow_bins))
P2_model=np.zeros((ntracers,pow_bins))

Cross_P0_model=np.zeros((ntracers*(ntracers-1)//2,pow_bins))
Cross_P2_model=np.zeros((ntracers*(ntracers-1)//2,pow_bins))
Cross_P0_theory=np.zeros((ntracers*(ntracers-1)//2,pow_bins))
Cross_P2_theory=np.zeros((ntracers*(ntracers-1)//2,pow_bins))

index=0
for i in range(ntracers):
    pk_ln_spec_corr_kbar[i] = np.interp(kph,k_spec_corr,pk_ln_spec_corr[:,i])
    P0_model[i] = np.interp(kph,k_spec_corr,pk_ln_mono_model[:,i])
    P2_model[i] = np.interp(kph,k_spec_corr,pk_ln_quad_model[:,i])
    P0_theory[i] = np.interp(kph,k_spec_corr,pk_ln_mono_theory[:,i])
    P2_theory[i] = np.interp(kph,k_spec_corr,pk_ln_quad_theory[:,i])
    for j in range(i+1,ntracers):
        Cross_P0_model[index] = np.interp(kph,k_spec_corr,cross_mono_model[:,index])
        Cross_P2_model[index] = np.interp(kph,k_spec_corr,cross_quad_model[:,index])
        Cross_P0_theory[index] = np.interp(kph,k_spec_corr,cross_mono_theory[:,index])
        Cross_P2_theory[index] = np.interp(kph,k_spec_corr,cross_quad_theory[:,index])
        index += 1

# Corrections for cross-spectra
cross_pk_ln_spec_corr_kbar=np.zeros((ntracers*(ntracers-1)//2,pow_bins))
index = 0
for i in range(ntracers):
    for j in range(i+1,ntracers):
        cross_pk_ln_spec_corr_kbar[index] = np.sqrt(pk_ln_spec_corr_kbar[i]*pk_ln_spec_corr_kbar[j])
        index +=1

# These are the theory values for the total effective power spectrum
nbarbar = np.mean(n_bar_matrix_fid,axis=(1,2,3))/(cell_size)**3
ntot = np.sum(nbarbar*effbias**2)
P0tot_theory = np.sum(nbarbar*P0_theory.T,axis=1)/ntot
P0tot_model = np.sum(nbarbar*P0_model.T,axis=1)/ntot



########################################
# ATTENTION: these are more like biases, not errors.
# We include them in the computation of the covariance, with an
# arbitrary ("fudge") factor that SHOULD BE UPDATED!
# biaserr
biaserr = 0.05
#
# Relative error due to angle averaging on square box
dd_P0_rel_kbar = biaserr*np.abs(P0_model - P0_model)/(small + np.abs(P0_model))
dd_P2_rel_kbar = biaserr*np.abs(P2_model - P2_model)/(small + np.abs(P2_model))
# Relative error due to Gaussian/Lognormal correspondence
dd_P_spec_kbar = biaserr*np.sqrt(np.var(pk_ln_spec_corr_kbar))*pk_ln_spec_corr_kbar
########################################




#############################################################################
# Here we prepare to apply the Jing (2005) deconvolution of the mass assignement function
# For the situations when this is necessary, see the input file
winmass_sims=np.ones(pow_bins)
winmass_data=np.ones(pow_bins)
if (jing_dec_sims) or (not sims_only):
    print ('Preparing to apply Jing deconvolution of mass assignement window function...')

    kN = np.pi/cell_size  # Nyquist frequency
    nxyz = np.arange(-4,5)
    idxyz= np.ones_like(nxyz)
    nx_xyz = np.einsum('i,j,k', nxyz,idxyz,idxyz)
    ny_xyz = np.einsum('i,j,k', idxyz,nxyz,idxyz)
    nz_xyz = np.einsum('i,j,k', idxyz,idxyz,nxyz)
    nxyz2 = nx_xyz**2 + ny_xyz**2 + nz_xyz**2

    nvec_xyz = np.meshgrid(nxyz,nxyz,nxyz)
    dmu_phi=0.1
    dmu_th=0.1
    # With these options for nxyz, dmu_phi and dmu_th, the WF is accurate to ~1% up to k~0.3 h/Mpc
    phi_xyz=np.arange(0.+dmu_phi/2.,1.,dmu_phi)*2*np.pi
    cosphi_xyz=np.cos(phi_xyz)
    sinphi_xyz=np.sin(phi_xyz)
    costheta_xyz=np.arange(-1.+dmu_th/2.,1.,dmu_th)
    sintheta_xyz=np.sqrt(1-costheta_xyz**2)

    # More or less randomly placed unit vectors
    unitxyz=np.zeros((len(phi_xyz),len(costheta_xyz),3))
    for iphi in range(len(phi_xyz)):
        for jth in range(len(costheta_xyz)):
            unitxyz[iphi,jth] = np.array([sintheta_xyz[jth]*cosphi_xyz[iphi],sintheta_xyz[jth]*sinphi_xyz[iphi],costheta_xyz[jth]])

    Nangles=len(phi_xyz)*len(costheta_xyz)
    unitxyz_flat = np.reshape(unitxyz,(Nangles,3))

    def wj02(ki,ni,power_jing):
        return np.abs(np.power(np.abs(special.j0(np.pi*(ki/kN/2. + ni))),power_jing))

    for i_k in range(pow_bins):
        kxyz = kph[i_k]*unitxyz_flat
        sum_sims=0.0
        sum_data=0.0
        for iang in range(Nangles):
            kdotk = kN*(kxyz[iang,0]*nx_xyz + kxyz[iang,1]*ny_xyz + kxyz[iang,2]*nz_xyz)
            kprime = np.sqrt( kph[i_k]**2 + 4*kN**2*nxyz2 + 2*kdotk )
            sum_sims += np.sum(wj02(kxyz[iang,0],nx_xyz,power_jing_sims)*wj02(kxyz[iang,1],ny_xyz,power_jing_sims)*wj02(kxyz[iang,2],nz_xyz,power_jing_sims)*pow_interp(kprime))
            sum_data += np.sum(wj02(kxyz[iang,0],nx_xyz,power_jing_data)*wj02(kxyz[iang,1],ny_xyz,power_jing_data)*wj02(kxyz[iang,2],nz_xyz,power_jing_data)*pow_interp(kprime))

        winmass_sims[i_k] = sum_sims/Nangles/powtrue[i_k]
        winmass_data[i_k] = sum_data/Nangles/powtrue[i_k]

    print ('... OK, computed Jing deconvolution functions')

#############################################################################
# BEGIN ESTIMATION
print ('Starting power spectra estimation')


# Initialize outputs

# Multitracer method: monopole, quadrupole, th. covariance(FKP-like)
# Original (convolved) spectra:
P0_data = np.zeros((n_maps,ntracers,num_binsk))
P2_data = np.zeros((n_maps,ntracers,num_binsk))
# Deconvolved spectra (divided by the window function):
P0_data_dec = np.zeros((n_maps,ntracers,num_binsk))
P2_data_dec = np.zeros((n_maps,ntracers,num_binsk))


# Traditional (FKP) method
P0_fkp = np.zeros((n_maps,ntracers,num_binsk))
P2_fkp = np.zeros((n_maps,ntracers,num_binsk))
Cross0 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
Cross2 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))

P0_fkp_dec = np.zeros((n_maps,ntracers,num_binsk))
P2_fkp_dec = np.zeros((n_maps,ntracers,num_binsk))
Cross0_dec = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
Cross2_dec = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))

# Covariance
ThCov_fkp = np.zeros((n_maps,ntracers,num_binsk))


# Range where we estimate some parameters
myk_min = int(len(k_bar)/4.)
myk_max = int(len(k_bar)*3./4.)
myran = np.arange(myk_min,myk_max)



#################################
# Initialize the multi-tracer estimation class
# We can assume many biases for the estimation of the power spectra:
#
# 1. Use the original bias
#fkp_mult = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,bias,cell_size,n_x,n_y,n_z,MRk,powercentral)
# 2. Use SQRT(monopole) for the bias. Now, we have two choices:
#
# 2a. Use the fiducial monopole as the MT bias
#
# 2b. Use the monopole estimated using the FKP technique
#fkp_mult = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,fkpeffbias,cell_size,n_x,n_y,n_z,MRk,powercentral)

# If shot noise is huge, then damp the effective bias of that species 
# so that it doesn't end up biasing the multi-tracer estimator: 
# nbar*b^2*P_0 > 0.01 , with P_0 = 2.10^4
effbias_mt = np.copy(effbias)
effbias_mt[nbarbar*effbias**2 < 0.5e-6] = 0.01


print( "Initializing multi-tracer estimation toolbox...")
fkp_mult = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,effbias_mt,cell_size,n_x,n_y,n_z,n_x_orig,n_y_orig,n_z_orig,MRk,powercentral)

# If data bias is different from mocks
try:
    data_bias
    effbias_mt_data = np.copy(effbias_mt)
    effbias_mt_data[nbarbar*effbias_data**2 < 0.5e-6] = 0.01
    fkp_mult_data = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,effbias_mt_data,cell_size,n_x,n_y,n_z,n_x_orig,n_y_orig,n_z_orig,MRk,powercentral)
except:
    pass

##
# UPDATED THIS TO NEW FKP CLASS WITH AUTO- AND CROSS-SPECTRA
print( "Initializing traditional (FKP) estimation toolbox...")
fkp_many = fkp.fkp_init(num_binsk, n_bar_matrix_fid, effbias, cell_size, n_x, n_y, n_z, n_x_orig, n_y_orig, n_z_orig, MRk, powercentral)


# If data bias is different from mocks
try:
    data_bias
    fkp_many_data = fkp.fkp_init(num_binsk, n_bar_matrix_fid, effbias_data, cell_size, n_x, n_y, n_z, n_x_orig, n_y_orig, n_z_orig, MRk, powercentral)
except:
    pass

print ("... done. Starting computations for each map (box) now.")
print()

#################################

est_bias_fkp = np.zeros(ntracers)
est_bias_mt = np.zeros(ntracers)

for nm in range(n_maps):
    time_start=time()
    if sims_only:
        print ('Loading simulated box #', nm)
        h5map = h5py.File(mapnames_sims[nm],'r')
        maps_halos = np.asarray(h5map.get(list(h5map.keys())[0]))
        if not maps_halos.shape == (nhalos,n_x,n_y,n_z):
            print()
            print ('Unexpected shape of simulated halo maps! Found', maps_halos.shape)
            print ('Check inputs and sims.  Aborting now...')
            print()
            sys.exit(-1)
        h5map.close
    elif nm == 0:
        print ('Loading data box...')
        h5map_data = h5py.File(mapnames_data,'r')
        h5data_data = h5map_data.get(list(h5map_data.keys())[0])
        maps = np.asarray(h5data_data)
        if not maps.shape == (ngals,n_x,n_y,n_z):
            print()
            print ('Incorrect shape of data (galaxy) maps! Found', maps.shape)
            print ('Aborting now...')
            print()
            sys.exit(-1)
        h5map_data.close
        # read and store sims maps
    elif nm > 0:
        print ('Loading simulated box #', nm)
        h5map = h5py.File(mapnames_sims[nm-1],'r')
        maps_halos = np.asarray(h5map.get(list(h5map.keys())[0]))
        if not maps_halos.shape == (nhalos,n_x,n_y,n_z):
            print()
            print( 'Unexpected shape of simulated halo maps! Found', maps_halos.shape)
            print ('Check inputs and sims.  Aborting now...')
            print()
            sys.exit(-1)
        h5map.close

    if do_galaxies:
        # If simulations only, all maps will be recast in the new form
        if (sims_only) or (nm>0):
            print ('Creating galaxy maps from halo maps now...')
            maps = np.zeros((ngals,n_x,n_y,n_z))
            for ng in range(ngals):
                for nh in range(nhalos):
                    maps[ng] += hod_3D[ng,nh]*maps_halos[nh]
                # Poisson sampling
                if do_poisson:
                    maps[ng] = np.random.poisson(maps[ng])
    else:
        if (sims_only) or (nm>0):
            maps = maps_halos
    #sys.exit(-1)

    if 'maps_halos' in globals():
        del maps_halos

    print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))

    ## !! NEW !! Additional mask from low-cell-count threshold
    try:
        cell_low_count_thresh
        maps = thresh_mask*maps
        print ("Total number of objects AFTER additional threshold mask:", np.sum(maps,axis=(1,2,3)))
    except:
        pass

    ##################################################
    # ATTENTION: The FKP estimator computes P_m(k), effectively dividing by the input bias factor.
    # Here the effective bias factor is bias*(growth function)*(amplitude of the monopole)
    # Notice that this means that the output of the FKP quadrupole
    # is P^2_FKP == (amplitude quadrupole)/(amplitude of the monopole)*P_m(k)
    ##################################################

    # Notice that we use "effective bias" (i.e., some estimate of the monopole) here;
    print ('  Estimating FKP power spectra...')
    normsel = np.mean(n_bar_matrix_fid,axis=(1,2,3))/np.mean(maps,axis=(1,2,3))
    normsel[normsel > 1.5] = 1.5
    normsel[normsel < 0.5] = 0.5
    if nm==0:
        # If data bias is different from mocks
        try:
            data_bias
            FKPmany = fkp_many_data.fkp((normsel*maps.T).T)
            P0_fkp[nm] = np.abs(fkp_many.P_ret)
            P2_fkp[nm] = fkp_many.P2a_ret
            Cross0[nm] = fkp_many.cross_spec
            Cross2[nm] = fkp_many.cross_spec2
            ThCov_fkp[nm] = (fkp_many.sigma)**2
        except:
            FKPmany = fkp_many.fkp((normsel*maps.T).T)
            P0_fkp[nm] = np.abs(fkp_many.P_ret)
            P2_fkp[nm] = fkp_many.P2a_ret
            Cross0[nm] = fkp_many.cross_spec
            Cross2[nm] = fkp_many.cross_spec2
            ThCov_fkp[nm] = (fkp_many.sigma)**2
    else:
        FKPmany = fkp_many.fkp((normsel*maps.T).T)
        P0_fkp[nm] = np.abs(fkp_many.P_ret)
        P2_fkp[nm] = fkp_many.P2a_ret
        Cross0[nm] = fkp_many.cross_spec
        Cross2[nm] = fkp_many.cross_spec2
        ThCov_fkp[nm] = (fkp_many.sigma)**2

    #################################
    # Now, the multi-tracer method
    print ('  Now estimating multi-tracer spectra...')
    normsel = np.mean(n_bar_matrix_fid,axis=(1,2,3))/np.mean(maps,axis=(1,2,3))
    normsel[normsel > 1.5] = 1.5
    normsel[normsel < 0.5] = 0.5
    if nm==0:
    # If data bias is different from mocks
        try:
            data_bias
            FKPmult = fkp_mult_data.fkp((normsel*maps.T).T)
            P0_data[nm] = np.abs(fkp_mult_data.P0_mu_ret)
            P2_data[nm] = fkp_mult_data.P2_mu_ret
        except:
            FKPmult = fkp_mult.fkp((normsel*maps.T).T)
            P0_data[nm] = np.abs(fkp_mult.P0_mu_ret)
            P2_data[nm] = fkp_mult.P2_mu_ret
    else:
        FKPmult = fkp_mult.fkp((normsel*maps.T).T)
        P0_data[nm] = np.abs(fkp_mult.P0_mu_ret)
        P2_data[nm] = fkp_mult.P2_mu_ret


    if nm==0:
    # If data bias is different from mocks
        try:
            data_bias
            est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
            est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
            print ("Biases of these maps:")
            print ("   Fiducial=", ["%.3f"%b for b in effbias])
            print ("        FKP=", ["%.3f"%b for b in est_bias_fkp])
            print ("         MT=", ["%.3f"%b for b in est_bias_mt])
            dt = time() - time_start
            print()
            print ("Elapsed time for computation of spectra for this map:", dt)
            print ("TIME NOW:", strftime("%a, %d %b %Y %H:%M:%S"))
            print()
        except:
            est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
            est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
            print ("Biases of these maps:")
            print ("   Fiducial=", ["%.3f"%b for b in effbias])
            print ("        FKP=", ["%.3f"%b for b in est_bias_fkp])
            print ("         MT=", ["%.3f"%b for b in est_bias_mt])
            dt = time() - time_start
            print()
            print ("Elapsed time for computation of spectra for this map:", dt)
            print ("TIME NOW:", strftime("%a, %d %b %Y %H:%M:%S"))
            print()
    else:
        est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
        est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
        print( "Biases of these maps:")
        print( "   Fiducial=", ["%.3f"%b for b in effbias])
        print( "        FKP=", ["%.3f"%b for b in est_bias_fkp])
        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
        dt = time() - time_start
        print()
        print ("Elapsed time for computation of spectra for this map:", dt)
        print ("TIME NOW:", strftime("%a, %d %b %Y %H:%M:%S"))
        print()



# Correct missing factor of 2 in definition
Theor_Cov_FKP = 2.0*np.mean(ThCov_fkp,axis=0)

del maps
maps = None



################################################################################
################################################################################


time_end=time()
print ('Total time cost for estimation of spectra: ', time_end - time_start)
print()
print ('Reducing extra shot noise...')

################################################################################
################################################################################

tempor=time()

################################################################################
################################################################################

print ('Applying window function corrections...')

################################################################################
# 1) Jing (cells) corrections
# Now apply ln correction and/or Jing deconvolution
if sims_only:
    P0_fkp = (P0_fkp)/winmass_sims
    P2_fkp = (P2_fkp)/winmass_sims
    Cross0 = Cross0/winmass_sims
    Cross2 = Cross2/winmass_sims
else:
    P0_fkp[0] = (P0_fkp[0])/winmass_data
    P2_fkp[0] = (P2_fkp[0])/winmass_data
    Cross0[0] = Cross0[0]/winmass_data
    Cross2[0] = Cross2[0]/winmass_data

    P0_fkp[1:] = (P0_fkp[1:])/winmass_sims
    P2_fkp[1:] = (P2_fkp[1:])/winmass_sims
    Cross0[1:] = Cross0[1:]/winmass_sims
    Cross2[1:] = Cross2[1:]/winmass_sims

if sims_only:
    P0_data = (P0_data)/winmass_sims
    P2_data = (P2_data)/winmass_sims
else:
    P0_data[0] = (P0_data[0])/winmass_data
    P2_data[0] = (P2_data[0])/winmass_data
    P0_data[1:] = (P0_data[1:])/winmass_sims
    P2_data[1:] = (P2_data[1:])/winmass_sims

# Means
P0_fkp_mean = np.mean(P0_fkp[1:],axis=0)
P2_fkp_mean = np.mean(P2_fkp[1:],axis=0)
Cross0_mean = np.mean(Cross0[1:],axis=0)
Cross2_mean = np.mean(Cross2[1:],axis=0)

P0_mean = np.mean(P0_data[1:],axis=0)
P2_mean = np.mean(P2_data[1:],axis=0)


################################################################################
# 2) Extra shot noise/1-halo-term subtraction
ksn1 = (3*pow_bins)//4
ksn2 = -1
spec_index = np.mean(np.diff(np.log(powtrue[ksn1:ksn2]))/np.diff(np.log(kph[ksn1:ksn2])))

P1h_fkp_data = np.zeros(ntracers)
P1h_fkp_sims = np.zeros(ntracers)
P1h_MT_data = np.zeros(ntracers)
P1h_MT_sims = np.zeros(ntracers)

for nt in range(ntracers):
    #print
    #print "Data"
    #print "Tracer", nt
    spec_index = np.mean(np.diff(np.log(P0_model[nt,ksn1:ksn2]))/np.diff(np.log(kph[ksn1:ksn2])))

    frac_p1h = 1.0 - 1./spec_index*np.mean(np.diff(np.log(np.abs(P0_fkp[0,nt,ksn1:ksn2])))/np.diff(np.log(kph[ksn1:ksn2])))
    #print 'FKP:', frac_p1h
    frac_p1h = np.min([0.9, np.max([frac_p1h,-0.9])])
    P1h_fkp_data[nt] = frac_p1h * np.mean(P0_fkp[0,nt,ksn1:ksn2])
    if np.isnan(P1h_fkp_data[nt]):
        P1h_fkp_data[nt] = 0.0

    frac_p1h = 1.0 - 1./spec_index*np.mean(np.diff(np.log(np.abs(P0_data[0,nt,ksn1:ksn2])))/np.diff(np.log(kph[ksn1:ksn2])))
    #print 'MT:', frac_p1h
    frac_p1h = np.min([0.9, np.max([frac_p1h,-0.9])])
    P1h_MT_data[nt] = frac_p1h * np.mean(P0_data[0,nt,ksn1:ksn2])
    if np.isnan(P1h_MT_data[nt]):
        P1h_MT_data[nt] = 0.0

    frac_p1h = 1.0 - 1./spec_index*np.mean(np.diff(np.log(np.abs(P0_fkp_mean[nt,ksn1:ksn2])))/np.diff(np.log(kph[ksn1:ksn2])))
    #print "Sims"
    #print 'FKP:', frac_p1h
    frac_p1h = np.min([0.9, np.max([frac_p1h,-0.9])])
    P1h_fkp_sims[nt] = frac_p1h * np.mean(P0_fkp_mean[nt,ksn1:ksn2])
    if np.isnan(P1h_fkp_sims[nt]):
        P1h_fkp_sims[nt] = 0.0

    frac_p1h = 1.0 - 1./spec_index*np.mean(np.diff(np.log(np.abs(P0_mean[nt,ksn1:ksn2])))/np.diff(np.log(kph[ksn1:ksn2])))
    #print ' MT:', frac_p1h
    frac_p1h = np.min([0.9, np.max([frac_p1h,-0.9])])
    P1h_MT_sims[nt] = frac_p1h * np.mean(P0_mean[nt,ksn1:ksn2])
    if np.isnan(P1h_MT_sims[nt]):
        P1h_MT_sims[nt] = 0.0


# Here subtract the shot noise, using the shot_fudge defined in the input file
P0_fkp_dec[0] = P0_fkp[0] - shot_fudge*np.outer(P1h_fkp_data,np.ones_like(kph))
P0_data_dec[0] = P0_data[0] - shot_fudge*np.outer(P1h_MT_data,np.ones_like(kph))
for nt in range(ntracers):
    P0_fkp_dec[1:,nt] = P0_fkp[1:,nt] - shot_fudge*np.outer(P1h_fkp_sims[nt],np.ones_like(kph))
    P0_data_dec[1:,nt] = P0_data[1:,nt] - shot_fudge*np.outer(P1h_MT_sims[nt],np.ones_like(kph))


P0_fkp_mean_dec = np.mean(P0_fkp_dec[1:],axis=0)
P0_mean_dec = np.mean(P0_data_dec[1:],axis=0)

P2_fkp_dec = np.copy(P2_fkp)
P2_data_dec = np.copy(P2_data)

P2_fkp_mean_dec = np.mean(P2_fkp_dec[1:],axis=0)
P2_mean_dec = np.mean(P2_data_dec[1:],axis=0)


#
# Plot MT estimates along with theory -- convolved spectra

pl.rcParams["axes.titlesize"] = 8
cm_subsection = np.linspace(0, 1, ntracers)
mycolor = [ cm.jet(x) for x in cm_subsection ]

pl.xscale('log')
pl.yscale('log')
xlow=0.99*kph[2]
xhigh=1.01*kph[-1]

for nt in range(ntracers):
    ddk = dkph_bin*0.1*(nt-ntracers/2.0+0.25)
    # Monopole
    color1=mycolor[nt]
    p = P0_data_dec[0,nt]
    errp = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt].T)))
    pl.errorbar(kph+ddk,p,errp,color=color1,linestyle='None',linewidth=0.5, marker='s', capsize=3, markersize=3)
    p = effbias[nt]**2*P0_fkp_dec[0,nt]
    errp = np.sqrt(np.diag(np.cov(effbias[nt]**2*P0_fkp_dec[1:,nt].T)))
    pl.errorbar(kph+2*ddk,p,errp,color=color1,linestyle='None',linewidth=0.1, marker='^', capsize=3, markersize=3)
    #pl.plot(kph,gp*P0_model[nt],color=color1,linewidth=1.2)
    pl.plot(kph,P0_mean[nt],color=color1,linewidth=1.0)

pl.legend()
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(\ell)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Convolved spectra - $P_i^{(0)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_errbar_Data_Conv_corr_kbar=' + str(kph_central) + '.pdf')
pl.close('all')


# Update bias estimate
kb_cut_min = np.argsort(np.abs(kph-kmin_bias))[0]
kb_cut_max = np.argsort(np.abs(kph-kmax_bias))[0]

def residuals(norm,data):
    return norm*data[1] - data[0]

# The measured monopoles and the flat-sky/theory monopoles are different; find relative normalization
normmonos = np.zeros(ntracers)
chi2_red = np.zeros(ntracers)

for nt in range(ntracers):
    err = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt,kb_cut_min:kb_cut_max].T)))
    data = [ P0_data_dec[0,nt,kb_cut_min:kb_cut_max]/err , P0_mean_dec[nt,kb_cut_min:kb_cut_max]/err ]
    this_norm, success = leastsq(residuals,1.0,args=data)
#    print weights_mono
    normmonos[nt] = np.sqrt(this_norm)
#    norm_monos[nt] = np.mean(np.sqrt(P0_data_dec[0,nt,kb_cut_min:kb_cut_max]/P0_mean_dec[nt,kb_cut_min:kb_cut_max]))
    chi2_red[nt] = np.sum(residuals(this_norm,data)**2)/(len(err)-1.)

print()
print ("Fiducial bias of the mocks, compared with data, before deconvolution:")
for nt in range(ntracers):
    print ('Fiducial (mocks):', np.around(gal_bias[nt],3), '; based on these mocks, data bias should be:', np.around(normmonos[nt]*gal_bias[nt],3), ' (chi^2 = ', np.around(chi2_red[nt],3), ')')

print ()



################################################################################
# 3) Corrections for lognormal maps / sim bias
if sims_only:
    P0_fkp_dec = (P0_fkp_dec*pk_ln_spec_corr_kbar)
    P2_fkp_dec = (P2_fkp*pk_ln_spec_corr_kbar)
    Cross0_dec = (Cross0*cross_pk_ln_spec_corr_kbar)
    Cross2_dec = (Cross2*cross_pk_ln_spec_corr_kbar)

    P0_data_dec = (P0_data_dec*pk_ln_spec_corr_kbar)
    P2_data_dec = (P2_data*pk_ln_spec_corr_kbar)
else:
    P0_fkp_dec[1:] = (P0_fkp_dec[1:]*pk_ln_spec_corr_kbar)
    P2_fkp_dec[1:] = (P2_fkp[1:]*pk_ln_spec_corr_kbar)
    Cross0_dec[1:] = (Cross0[1:]*cross_pk_ln_spec_corr_kbar)
    Cross2_dec[1:] = (Cross2[1:]*cross_pk_ln_spec_corr_kbar)

    P0_data_dec[1:] = (P0_data_dec[1:]*pk_ln_spec_corr_kbar)
    P2_data_dec[1:] = (P2_data[1:]*pk_ln_spec_corr_kbar)


P0_fkp_mean_dec = np.mean(P0_fkp_dec[1:],axis=0)
P0_mean_dec = np.mean(P0_data_dec[1:],axis=0)
Cross0_mean_dec = np.mean(Cross0_dec[1:],axis=0)
Cross2_mean_dec = np.mean(Cross2_dec[1:],axis=0)

P2_fkp_mean_dec = np.mean(P2_fkp_dec[1:],axis=0)
P2_mean_dec = np.mean(P2_data_dec[1:],axis=0)




################################################################################
# 4) Window functions corrections (computed from simulations or from theory)

winfun0 = np.ones((ntracers,pow_bins))
winfun0_cross = np.ones((ntracers*(ntracers-1)//2,pow_bins))

winfun2 = np.ones((ntracers,pow_bins))
winfun2_cross = np.ones((ntracers*(ntracers-1)//2,pow_bins))


# In order to be consistent with older versions:
try:
	use_window_function
	is_n_body_sims = use_window_function
except:
	use_window_function = is_n_body_sims


if use_window_function:
    win_fun_file=glob.glob("spectra/"+win_fun_dir+"*WinFun0*")
    win_fun_file_cross0=glob.glob("spectra/"+win_fun_dir+"*WinFun_Cross0*")
    win_fun_file_cross2=glob.glob("spectra/"+win_fun_dir+"*WinFun_Cross2*")
    if (len(win_fun_file)!=1) | (len(win_fun_file_cross0)!=1) | (len(win_fun_file_cross2)!=1):
        print ("Could not find (or found more than one) specified window functions at", "spectra/", win_fun_dir)
        print ("Using no window function")
        wf = np.ones((pow_bins,2*ntracers))
        wf_c0 = np.ones((pow_bins,ntracers*(ntracers-1)//2))
        wf_c2 = np.ones((pow_bins,ntracers*(ntracers-1)//2))
    else:
        win_fun_file = win_fun_file[0]
        win_fun_file_cross0 = win_fun_file_cross0[0]
        win_fun_file_cross2 = win_fun_file_cross2[0]

        wf = np.loadtxt(win_fun_file)

        if (len(wf) != pow_bins) | (len(wf.T) != 2*ntracers):
            print ("Dimensions of window functions (P0 auto & P2 auto) do not match those of this estimation code!")
            print ("Please check that window function, or create a new one. Aborting now...")
            sys.exit(-1)

        if(ntracers>1):
            wf_c0 = np.loadtxt(win_fun_file_cross0)
            wf_c2 = np.loadtxt(win_fun_file_cross2)

            if ( (ntracers>2) and (len(wf_c0) != pow_bins) | (len(wf_c0.T) != ntracers*(ntracers-1)//2) ):
                print ("Dimensions of window function of cross spectra for P0 do not match those of this estimation code!")
                print ("Please check that window function, or create a new one. Aborting now...")
                sys.exit(-1)
            if (len(wf_c0.T)!=pow_bins):
                print ("Dimensions of window function of cross spectra for P0 do not match those of this estimation code!")
                print ("Please check that window function, or create a new one. Aborting now...")
                sys.exit(-1)
            wf_c0 = np.reshape(wf_c0, (pow_bins,1))

            if ((ntracers>2) and (len(wf_c2) != pow_bins) | (len(wf_c2.T) != ntracers*(ntracers-1)//2)):
                print ("Dimensions of window function of cross spectra for P2 do not match those of this estimation code!")
                print ("Please check that window function, or create a new one. Aborting now...")
                sys.exit(-1)

            if (len(wf_c2.T)!=pow_bins):
                print ("Dimensions of window function of cross spectra for P0 do not match those of this estimation code!")
                print ("Please check that window function, or create a new one. Aborting now...")
                sys.exit(-1)
            wf_c2 = np.reshape(wf_c2, (pow_bins,1))


    mean_win0 = wf[:,:ntracers]
    mean_win2 = wf[:,ntracers:]
    # Deconvolve FKP
    P0_fkp_dec = P0_fkp_dec / (small + mean_win0.T)
    P2_fkp_dec = P2_fkp_dec / (small + mean_win2.T)
    P0_fkp_mean_dec = np.mean(P0_fkp_dec,axis=0)
    P2_fkp_mean_dec = np.mean(P2_fkp_dec,axis=0)
    # Deconvolve MT
    P0_data_dec = P0_data_dec / (small + mean_win0.T)
    P2_data_dec = P2_data_dec / (small + mean_win2.T)
    P0_mean_dec = np.mean(P0_data_dec,axis=0)
    P2_mean_dec = np.mean(P2_data_dec,axis=0)

    index = 0
    for i in range(ntracers):
        for j in range(i+1,ntracers):
            Cross0_dec[:,index] = Cross0_dec[:,index] / (small + wf_c0[:,index])
            Cross2_dec[:,index] = Cross2_dec[:,index] / (small + wf_c2[:,index])
            index += 1
else:
    wf_c0 = np.ones((pow_bins,ntracers*(ntracers-1)//2))
    wf_c2 = np.ones((pow_bins,ntracers*(ntracers-1)//2))
    for nt in range(ntracers):
        # FKP has its window function...
        winfun0[nt] = effbias[nt]**2*P0_fkp_mean_dec[nt]/(small + P0_model[nt])
        winfun2[nt] = effbias[nt]**2*P2_fkp_mean_dec[nt]/(small + P2_model[nt])
        P0_fkp_dec[:,nt] = P0_fkp_dec[:,nt] / (small + winfun0[nt])
        P2_fkp_dec[:,nt] = P2_fkp_dec[:,nt] / (small + winfun2[nt])
        P0_fkp_mean_dec[nt] = np.mean(P0_fkp_dec[:,nt],axis=0)
        P2_fkp_mean_dec[nt] = np.mean(P2_fkp_dec[:,nt],axis=0)
        # MT has its window function... which is the only one we store... so, same name
        winfun0[nt] = P0_mean_dec[nt]/(small + P0_model[nt])
        winfun2[nt] = P2_mean[nt]/(small + P2_model[nt])
        P0_data_dec[:,nt] = P0_data_dec[:,nt] / (small + winfun0[nt])
        P2_data_dec[:,nt] = P2_data[:,nt] / (small + winfun2[nt])
        P0_mean_dec[nt] = np.mean(P0_data_dec[:,nt],axis=0)
        P2_mean_dec[nt] = np.mean(P2_data_dec[:,nt],axis=0)
    # Cross spectra        
    index = 0
    for i in range(ntracers):
        for j in range(i+1,ntracers):
            model0 = np.sqrt( P0_model[i]*P0_model[j] )
            model2 = np.sqrt( P2_model[i]*P0_model[j] )
            wf_c0[:,index] = effbias[i]*effbias[j]*Cross0_mean_dec[index] / (small + model0)
            wf_c2[:,index] = effbias[i]*effbias[j]*Cross2_mean_dec[index] / (small + model2)
            index += 1



################################################################################
################################################################################

# Compute the theoretical covariance:
# first, compute the Fisher matrix, then invert it.
# Do this for for each k bin


# free up memory
n_bar_matrix_fid=None
del n_bar_matrix_fid



################################################################################
################################################################################
################################################################################
################################################################################




print ('Now computing data covariances of the simulated spectra...')
tempor=time()

# First, compute total effective spectrum of all species combined
P0tot_MT = np.zeros((n_maps,num_binsk),dtype="float16")
P0tot_FKP = np.zeros((n_maps,num_binsk),dtype="float16")
for i in range(n_maps):
    P0tot_MT[i] = np.sum(nbarbar*P0_data_dec[i].T,axis=1)/ntot
    P0tot_FKP[i] = np.sum(nbarbar*effbias**2*P0_fkp_dec[i].T,axis=1)/ntot

P0tot_mean_MT = np.mean(P0tot_MT,axis=0)
P0tot_mean_FKP = np.mean(P0tot_FKP,axis=0)

# These are the RELATIVE covariances between all tracers .
# We have Dim P0,2_data : [nmaps,ntracers,num_binsk]
relcov0_MT  = np.zeros((ntracers,ntracers,num_binsk,num_binsk),dtype="float16")
relcov0_FKP = np.zeros((ntracers,ntracers,num_binsk,num_binsk),dtype="float16")
relcov2_MT  = np.zeros((ntracers,ntracers,num_binsk,num_binsk),dtype="float16")
relcov2_FKP = np.zeros((ntracers,ntracers,num_binsk,num_binsk),dtype="float16")

# Covariance of the RATIOS between tracers -- there are n*(n-1)/2 of them -- like the cross-covariances
fraccov0_MT  = np.zeros((ntracers*(ntracers-1)//2,num_binsk,num_binsk),dtype="float16")
fraccov0_FKP = np.zeros((ntracers*(ntracers-1)//2,num_binsk,num_binsk),dtype="float16")
fraccov2_MT  = np.zeros((ntracers*(ntracers-1)//2,num_binsk,num_binsk),dtype="float16")
fraccov2_FKP = np.zeros((ntracers*(ntracers-1)//2,num_binsk,num_binsk),dtype="float16")


# Covariance calculations
# Use the convolved estimators for the amplitudes of the spectra;
# Use the DEconvolved estimators for the ratios of spectra

# Build "super array" containing Ptot and ratios of spectra
P0_tot_ratios_FKP = P0tot_FKP[1:]
P0_tot_ratios_MT = P0tot_MT[1:]
ntcount=0
for nt in range(ntracers):
    rc0_MT = np.cov(P0_data_dec[1:,nt].T)/(small+np.abs(np.outer(P0_mean_dec[nt],P0_mean_dec[nt])))
    rc0_FKP = np.cov(P0_fkp_dec[1:,nt].T)/(small+np.abs(np.outer(P0_fkp_mean_dec[nt],P0_fkp_mean_dec[nt])))
    dd_rc = np.diag(dd_P_spec_kbar[nt]**2)
    norm_rc0_nt = np.var(dd_P0_rel_kbar[nt])
    dd_rc0 = norm_rc0_nt*np.diag(dd_P0_rel_kbar[nt]**2)
    relcov0_MT[nt,nt] = rc0_MT + dd_rc + dd_rc0
    relcov0_FKP[nt,nt] = rc0_FKP + dd_rc + dd_rc0
    dd_rc2 = np.diag(dd_P2_rel_kbar[nt])
    rc2_MT = np.cov(P2_data_dec[1:,nt].T)/(small+np.abs(np.outer(P2_mean_dec[nt],P2_mean_dec[nt])))
    rc2_FKP = np.cov(P2_fkp_dec[1:,nt].T)/(small+np.abs(np.outer(P2_fkp_mean_dec[nt],P2_fkp_mean_dec[nt])))
    norm_rc2_nt = np.var(dd_P2_rel_kbar[nt])
    dd_rc2 = norm_rc2_nt*np.diag(dd_P2_rel_kbar[nt]**2)
    relcov2_MT[nt,nt] = rc2_MT + dd_rc + dd_rc2
    relcov2_FKP[nt,nt] = rc2_FKP + dd_rc + dd_rc2
    for ntp in range(nt+1,ntracers):
        dd_rc = np.diag(dd_P_spec_kbar[nt]*dd_P_spec_kbar[ntp])
        norm_rc0_ntp = np.var(dd_P0_rel_kbar[ntp])
        dd_rc0 = np.sqrt(norm_rc0_nt*norm_rc0_ntp)*np.diag(dd_P0_rel_kbar[nt]*dd_P0_rel_kbar[ntp])
        norm_rc2_ntp = np.var(dd_P2_rel_kbar[ntp])
        dd_rc2 = np.sqrt(norm_rc2_nt*norm_rc2_ntp)*np.diag(dd_P2_rel_kbar[nt]*dd_P2_rel_kbar[ntp])
        ppmt = small + np.abs(np.outer(P0_mean_dec[nt],P0_mean_dec[ntp]))
        ppmt2 = small + np.abs(np.outer(P2_mean_dec[nt],P2_mean_dec[ntp]))
        ppfkp = small + np.abs(np.outer(P0_fkp_mean_dec[nt],P0_fkp_mean_dec[ntp]))
        ppfkp2 = small + np.abs(np.outer(P2_fkp_mean_dec[nt],P2_fkp_mean_dec[ntp]))
        relcov0_MT[nt,ntp] = ((np.cov(P0_data_dec[1:,nt].T,P0_data_dec[1:,ntp].T))[num_binsk:,:num_binsk])/(small+ppmt) + dd_rc + dd_rc0
        relcov2_MT[nt,ntp] = ((np.cov(P2_data_dec[1:,nt].T,P2_data_dec[1:,ntp].T))[num_binsk:,:num_binsk])/(small+ppmt2) + dd_rc + dd_rc2
        relcov0_FKP[nt,ntp] = ((np.cov(P0_fkp_dec[1:,nt].T,P0_fkp_dec[1:,ntp].T))[num_binsk:,:num_binsk])/(small+ppfkp) + dd_rc + dd_rc0
        relcov2_FKP[nt,ntp] = ((np.cov(P2_fkp_dec[1:,nt].T,P2_fkp_dec[1:,ntp].T))[num_binsk:,:num_binsk])/(small+ppfkp2) + dd_rc + dd_rc2
        relcov0_MT[ntp,nt] = relcov0_MT[nt,ntp]
        relcov2_MT[ntp,nt] = relcov2_MT[nt,ntp]
        relcov0_FKP[ntp,nt] = relcov0_FKP[nt,ntp]
        relcov2_FKP[ntp,nt] = relcov2_FKP[nt,ntp]
        rat0_MT = P0_data_dec[1:,nt].T/(small+P0_data_dec[1:,ntp].T)
        rat2_MT = P2_data_dec[1:,nt].T/(small+P2_data_dec[1:,ntp].T)
        fraccov0_MT[ntcount] = np.cov(rat0_MT)
        fraccov2_MT[ntcount] = np.cov(rat2_MT)
        rat0_FKP = effbias[nt]**2*P0_fkp_dec[1:,nt].T/(small+effbias[ntp]**2*P0_fkp_dec[1:,ntp].T)
        rat2_FKP = effbias[nt]**2*P2_fkp_dec[1:,nt].T/(small+effbias[ntp]**2*P2_fkp_dec[1:,ntp].T)
        fraccov0_FKP[ntcount] = np.cov(rat0_FKP)
        fraccov2_FKP[ntcount] = np.cov(rat2_FKP)
        P0_tot_ratios_MT = np.hstack((P0_tot_ratios_MT,rat0_MT.T))
        P0_tot_ratios_FKP = np.hstack((P0_tot_ratios_FKP,rat0_FKP.T))
        ntcount = ntcount + 1


# Correlation matrix of total effective power spectrum and ratios of spectra
cov_Pt_ratios_MT = np.cov(P0_tot_ratios_MT.T)
cov_Pt_ratios_FKP = np.cov(P0_tot_ratios_MT.T)
cov_Pt_MT = np.cov(P0tot_MT.T)
cov_Pt_FKP = np.cov(P0tot_FKP.T)

print ('Done computing data covariances. Time spent: ', np.int((time()-tempor)*1000)/1000., 's')
print ()
print ('----------------------------------')
print ()


#print('Mean D0D0/true:',np.mean(frac00))

print ('Results:')

Vfft_to_Vk = 1.0/((n_x*n_y)*(n_z/2.))

## Compare theory with sims and with data
eff_mono_fkp = np.median(effbias**2*(P0_fkp_mean_dec/powtrue).T [myran],axis=0)
eff_mono_mt = np.median((P0_mean_dec/powtrue).T [myran],axis=0)
eff_quad_fkp = np.median(effbias**2*(P2_fkp_mean_dec/powtrue).T [myran],axis=0)
eff_quad_mt = np.median((P2_mean_dec/powtrue).T [myran],axis=0)

eff_mono_fkp_data = np.median(effbias**2*(P0_fkp_dec[0]/powtrue).T [myran],axis=0)
eff_mono_mt_data = np.median((P0_data_dec[0]/powtrue).T [myran],axis=0)
eff_quad_fkp_data = np.median(effbias**2*(P2_fkp_dec[0]/powtrue).T [myran],axis=0)
eff_quad_mt_data = np.median((P2_data_dec[0]/powtrue).T [myran],axis=0)

mono_theory = np.median((P0_model/powtrue).T [myran],axis=0)
quad_theory = np.median((P2_model/powtrue).T [myran],axis=0)


print('At k=', kph[myk_min], '...', kph[myk_max])
for nt in range(ntracers):
    print('----------------------------------')
    print('Tracer:', nt )
    print('    Theory averaged monopole = ', 0.001*np.int( 1000.0*mono_theory[nt]))
    print('FKP (sims) averaged monopole = ', 0.001*np.int( 1000.0*eff_mono_fkp[nt]))
    print(' MT (sims) averaged monopole = ', 0.001*np.int( 1000.0*eff_mono_mt[nt]))
    print('FKP (data) averaged monopole = ', 0.001*np.int( 1000.0*eff_mono_fkp_data[nt]))
    print(' MT (data) averaged monopole = ', 0.001*np.int( 1000.0*eff_mono_mt_data[nt]))
    print('    Theory averaged quadrupole = ', 0.001*np.int( 1000.0*quad_theory[nt]))
    print('FKP (sims) averaged quadrupole = ', 0.001*np.int( 1000.0*eff_quad_fkp[nt]))
    print(' MT (sims) averaged quadrupole = ', 0.001*np.int( 1000.0*eff_quad_mt[nt]))
    print('FKP (data) averaged quadrupole = ', 0.001*np.int( 1000.0*eff_quad_fkp_data[nt]))
    print(' MT (data) averaged quadrupole = ', 0.001*np.int( 1000.0*eff_quad_mt_data[nt]))

print('----------------------------------')
print ()






################################################################################
################################################################################
################################################################################
################################################################################
#
# Computations are basically completed.
# Now create plots, outputs, etc.
#
################################################################################
################################################################################
################################################################################
################################################################################



pl.rcParams["axes.titlesize"] = 8
cm_subsection = np.linspace(0, 1, ntracers)
mycolor = [ cm.jet(x) for x in cm_subsection ]

#colorsequence=['darkred','r','darkorange','goldenrod','y','yellowgreen','g','lightseagreen','c','deepskyblue','b','darkviolet','m']
#jumpcolor=np.int(len(colorsequence)/ntracers)


print('Plotting results to /figures...')

if plot_all_cov:
    # Plot 2D correlation of Ptot and ratios
    nblocks=1+ntracers*(ntracers-1)//2
    indexcov = np.arange(0,nblocks*num_binsk,np.int(num_binsk//4))
    nameindex = nblocks*[str(0.001*np.round(1000*kin)) for kin in kph[0:-1:np.int(num_binsk//4)]]
    onesk=np.diag(np.ones((nblocks*num_binsk)))
    dF=np.sqrt(np.abs(np.diag(cov_Pt_ratios_FKP)))
    dM=np.sqrt(np.abs(np.diag(cov_Pt_ratios_MT)))
    dF2 = small + np.outer(dF,dF)
    dM2 = small + np.outer(dM,dM)
    fullcov = np.tril(np.abs(cov_Pt_ratios_FKP)/dF2) + np.triu(np.abs(cov_Pt_ratios_MT.T)/dM2) - onesk
    pl.imshow(fullcov,origin='lower',interpolation='none')
    pl.title("Covariance of total effective power spectrum and ratios of spectra (monopoles only)")
    pl.xticks(indexcov,nameindex,size=6,name='monospace')
    pl.yticks(indexcov,nameindex,size=8,name='monospace')
    pl.annotate('FKP',(np.int(pow_bins/5.),2*pow_bins),fontsize=20)
    pl.annotate('Multi-tracer',(2*pow_bins,np.int(pow_bins/5.)),fontsize=20)
    pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
    pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
    pl.colorbar()
    pl.savefig(dir_figs + '/' + handle_estimates + '_2D_tot_ratios_corr.pdf')
    pl.close('all')

    # Plot 2D correlation coefficients
    indexcov = np.arange(0,num_binsk,np.int(num_binsk//5))
    nameindex = [str(0.001*np.round(1000*kin)) for kin in kph[0:-1:np.int(num_binsk//5)]]
    onesk=np.diag(np.ones(num_binsk))
    for nt in range(ntracers):
        for ntp in range(nt,ntracers):
            kk = np.outer(kph,kph)
            FKPcov=relcov0_FKP[nt,ntp]
            MTcov=relcov0_MT[nt,ntp]
            dF=np.sqrt(np.abs(np.diag(FKPcov)))
            dM=np.sqrt(np.abs(np.diag(MTcov)))
            FKPcorr=FKPcov/(small+np.outer(dF,dF))
            MTcorr=MTcov/(small+np.outer(dM,dM))
            fullcov = np.tril(np.abs(FKPcorr)) + np.triu(np.abs(MTcorr.T)) - onesk
            thistitle = 'Corr(P_' + str(nt) + ',P_' + str(ntp) + ') '
            pl.imshow(fullcov,origin='lower',interpolation='none')
            pl.title(thistitle)
            pl.xticks(indexcov,nameindex,size=20,name='monospace')
            pl.yticks(indexcov,nameindex,size=20,name='monospace')
            pl.annotate('FKP',(np.int(pow_bins//10),np.int(pow_bins/2.5)),fontsize=20)
            pl.annotate('Multi-tracer',(np.int(pow_bins//2),np.int(pow_bins//10)),fontsize=20)
            pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
            pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
            pl.colorbar()
            pl.savefig(dir_figs + '/' + handle_estimates + '_2Dcorr_' + str(nt) + '_' + str(ntp) + '.pdf')
            pl.close('all')


# Marker: open circle
mymark=[r'$\circ$']
smoothfilt = np.ones(5)/5.0
kphsmooth = np.convolve(kph,smoothfilt,mode='valid')

# Plot relative errors (diagonal of covariance)
# of the monopoles and quadrupoles for SINGLE tracers
# Use the convolved estimator for the variances/covariances
for nt in range(ntracers):
    tcfkp = np.convolve(Theor_Cov_FKP[nt]/(small+P0_fkp_mean[nt]**2),smoothfilt,mode='valid')
    pl.plot(kphsmooth,np.sqrt(np.abs(tcfkp)),'r-',linewidth=0.5)
    fkpcov= np.sqrt(np.abs(np.diagonal(relcov0_FKP[nt,nt])))
    fkpcov2= np.sqrt(np.abs(np.diagonal(relcov2_FKP[nt,nt])))
    label = str(nt)
    pl.semilogy(kph,fkpcov,marker='x',color='r',linestyle='', label = 'FKP Mono Tracer ' + label)
    pl.semilogy(kph,fkpcov2,marker='+',color='g',linestyle='', label = 'FKP Quad Tracer' + label)
    #tcmt = np.convolve(Theor_Cov_MT[nt,nt]/(small+P0_mean[nt]**2),smoothfilt,mode='valid')
    #pl.semilogy(kphsmooth,np.abs(tcmt),'k-',linewidth=2.5)
    mtcov = np.sqrt(np.abs(np.diagonal(relcov0_MT[nt,nt])))
    mtcov2 = np.sqrt(np.abs(np.diagonal(relcov2_MT[nt,nt])))
    pl.semilogy(kph,mtcov,marker='o',color='k',linestyle='', label = 'MT Mono Tracer ' + label)
    pl.semilogy(kph,mtcov2,marker='.',color='b',linestyle='', label = 'MT Quad Tracer ' + label)
    pl.legend()
    pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
    pl.ylabel(r'$\sigma [P^{(0)}]/[P^{(0)}]$ , $\sigma [P^{(2)}]/[P^{(2)}] $',fontsize=14)
    thistitle = 'Variance of tracer ' + str(nt)
    pl.title(thistitle,fontsize=16)
    pl.savefig(dir_figs + '/' + handle_estimates + '_sigmas_' + str(nt+1) + '.pdf')
    pl.close('all')


# Plot relative errors (diagonal of covariance)
# of the monopoles and quadrupoles for the CROSS-COVARIANCE between tracers
if plot_all_cov:
    if ntracers > 1:
        for nt in range(ntracers):
            for ntp in range(nt+1,ntracers):
                fkpcov= np.diagonal(relcov0_FKP[nt,ntp])
                fkpcov2= np.diagonal(relcov2_FKP[nt,ntp])
                pl.semilogy(kph,np.abs(fkpcov),marker='x',color='r',linestyle='')
                pl.semilogy(kph,np.abs(fkpcov2),marker='+',color='g',linestyle='')
                mtcov = np.diagonal(relcov0_MT[nt,ntp])
                mtcov2 = np.diagonal(relcov2_MT[nt,ntp])
                pl.semilogy(kph,np.abs(mtcov),marker='o',color='k',linestyle='')
                pl.semilogy(kph,np.abs(mtcov2),marker='.',color='b',linestyle='')
                ymin, ymax = 0.00000001, 10.0
                pl.xlim([kph[2],kph[-2]])
                pl.ylim([ymin,ymax])
                pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
                pl.ylabel('Cross-covariances (relative)',fontsize=12)
                thistitle = 'Cross-covariance of tracers ' + str(nt) + ' , ' + str(ntp)
                pl.title(thistitle,fontsize=16)
                pl.savefig(dir_figs + '/'  + handle_estimates + '_cross_cov_' + str(nt+1) + str(ntp+1) +'.pdf')
                pl.close('all')


# Plot relative errors (diagonal of covariance) of the RATIOS
# between monopoles and quadrupoles between tracers.
# N.B.: we are using the DEconcolved spectra for these error estimations!
if plot_all_cov:
    ntcount=0
    if ntracers > 1:
        for nt in range(ntracers):
            for ntp in range(nt+1,ntracers):
                fkpcov= np.sqrt(np.diagonal(fraccov0_FKP[ntcount]))
                fkpcov2= np.sqrt(np.diagonal(fraccov2_FKP[ntcount]))
                pl.semilogy(kph,fkpcov,marker='x',color='r',linestyle='')
                pl.semilogy(kph,fkpcov2,marker='+',color='g',linestyle='')
                mtcov = np.sqrt(np.diagonal(fraccov0_MT[ntcount]))
                mtcov2 = np.sqrt(np.diagonal(fraccov2_MT[ntcount]))
                pl.semilogy(kph,mtcov,marker='o',color='k',linestyle='')
                pl.semilogy(kph,mtcov2,marker='.',color='b',linestyle='')
                ymin, ymax = np.min(mtcov)*0.5, 10.0
                pl.xlim([kph[2],kph[-5]])
                pl.ylim([ymin,ymax])
                pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
                pl.ylabel(r'$\sigma(P_\mu/P_\nu)$',fontsize=12)
                thistitle = 'Relative variance of the ratios between tracers ' + str(nt+1) + ' , ' + str(ntp+1)
                pl.title(thistitle,fontsize=16)
                pl.savefig(dir_figs + '/' + handle_estimates + '_frac_cov_' + str(nt) + str(ntp) +'.pdf')
                pl.close('all')
                ntcount = ntcount + 1


# Plot DECONVOLVED estimates along with theory and data
# Monopole only
pl.xscale('log')
pl.yscale('log')
ylow=np.median((kph*P0_mean_dec)[0,-10:])*0.2
yhigh=np.mean(np.abs((kph*P0_mean_dec)[-1,2:10]))*5.0
xlow=0.99*kph[2]
xhigh=1.01*kph[-1]
pl.ylim([ylow,yhigh])
pl.xlim([xlow,xhigh])
# Plot total effective power spectrum: theory, model (box) and data
s0=np.sqrt(np.diag(cov_Pt_MT))
p0plus = np.minimum(yhigh,np.abs(kph*(P0tot_model + s0)))
p0minus = np.maximum(ylow,np.abs(kph*(P0tot_model - s0)))
pl.fill_between( kph, p0minus, p0plus, color = 'k', alpha=0.15)
pl.plot(kph, kph*P0tot_model, color='k', linestyle='-', linewidth=0.4)
# plot means of sims
pl.plot(kph, kph*P0tot_mean_MT, color='k', linestyle='--', linewidth=0.5)
pl.plot(kph, kph*P0tot_mean_FKP, color='k', linestyle='--', linewidth=0.2)
# plot data for total effective power
pl.plot(kph, kph*P0tot_MT[0], color='k', marker='.', linestyle='none')
pl.plot(kph, kph*P0tot_FKP[0], color='k', marker='x', linestyle='none')
for nt in range(ntracers):
    color1=mycolor[nt]
    # plot error bars as filled regions
    s0=np.sqrt(np.diag(relcov0_MT[nt,nt]))
    p0plus = np.minimum(yhigh,np.abs(np.abs(kph*P0_model[nt])*(1.0 + s0)))
    p0minus = np.maximum(ylow,np.abs(np.abs(kph*P0_model[nt])*(1.0 - s0)))
    pl.fill_between( kph, p0minus, p0plus, color = color1, alpha=0.15)
    # plot means of sims
    pl.plot(kph, kph*effbias[nt]**2*P0_fkp_mean_dec[nt], color=color1, linestyle='-', linewidth=0.2)
    pl.plot(kph,-kph*effbias[nt]**2*P0_fkp_mean_dec[nt], color=color1, linestyle='-.', linewidth=0.2)
    pl.plot(kph, kph*P0_mean_dec[nt], color=color1, linestyle='-', linewidth=0.5)
    pl.plot(kph,-kph*P0_mean_dec[nt], color=color1, linestyle='-.', linewidth=0.5)
    # Plot theory
    pl.plot(kph, kph*P0_model[nt], color=color1, linestyle='-', linewidth=1)
    # plot data -- markers
    pl.plot(kph, kph*effbias[nt]**2*P0_fkp_dec[0,nt], color=color1, marker='x', linestyle='none')
    pl.plot( kph, kph*P0_data_dec[0,nt], color=color1, marker='.', linestyle='none')

pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$k \, P^{(0)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Deconvolved spectra - $P_i^{(0)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_P0_deconv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')



# Plot DECONVOLVED estimates along with theory and data
# Monopole and quadrupole
pl.xscale('log')
pl.yscale('log')
ylow=np.median(P0_mean_dec[0,-10:])*0.2
yhigh=np.mean(np.abs(P0_mean_dec[:,2:10]))*5.0
xlow=0.99*kph[2]
xhigh=1.01*kph[-1]
pl.ylim([ylow,yhigh])
pl.xlim([xlow,xhigh])
for nt in range(ntracers):
    color1=mycolor[nt]
    # plot error bars as filled regions
    s0=np.sqrt(np.diag(relcov0_MT[nt,nt]))
    s2=np.sqrt(np.diag(relcov2_MT[nt,nt]))
    p0plus = np.minimum(yhigh,np.abs(P0_data_dec[0,nt]*(1.0 + s0)))
    p0minus = np.maximum(ylow,np.abs(P0_data_dec[0,nt]*(1.0 - s0)))
    p2plus = np.minimum(yhigh,np.abs(P2_data_dec[0,nt]*(1.0 + s2)))
    p2minus = np.maximum(ylow,np.abs(P2_data_dec[0,nt]*(1.0 - s2)))
    pl.fill_between( kph, p0minus, p0plus, color = color1, alpha=0.15)
    pl.fill_between( kph, p2minus, p2plus, color = color1, alpha=0.15)
    # plot means of sims
    pl.plot(kph, effbias[nt]**2*P0_fkp_mean_dec[nt], color=color1, linestyle='-', linewidth=0.2)
    pl.plot(kph,-effbias[nt]**2*P0_fkp_mean_dec[nt], color=color1, linestyle='-.', linewidth=0.2)
    pl.plot(kph,-effbias[nt]**2*P2_fkp_mean_dec[nt], color=color1, linestyle='-.',linewidth=0.2)
    pl.plot(kph, effbias[nt]**2*P2_fkp_mean_dec[nt], color=color1, linestyle='-', linewidth=0.2)
    pl.plot(kph, P0_mean_dec[nt], color=color1, linestyle='-', linewidth=0.5)
    pl.plot(kph,-P0_mean_dec[nt], color=color1, linestyle='-.', linewidth=0.5)
    pl.plot(kph,-P2_mean_dec[nt], color=color1, linestyle='-.',linewidth=0.5)
    pl.plot(kph, P2_mean_dec[nt], color=color1, linestyle='-', linewidth=0.5)
    # Plot theory
    pl.plot(kph, P0_model[nt], color=color1, linestyle='-', linewidth=1)
    pl.plot(kph, P2_model[nt], color=color1, linestyle='-', linewidth=1)
    # plot data -- markers
    pl.plot(kph, effbias[nt]**2*P0_fkp_dec[0,nt], color=color1, marker='x', linestyle='none')
    pl.plot(kph, effbias[nt]**2*P2_fkp_dec[0,nt], color=color1, marker='x', linestyle='none')
    pl.plot(kph,-effbias[nt]**2*P2_fkp_dec[0,nt], color=color1, marker='1', linestyle='none')
    pl.plot( kph, P0_data_dec[0,nt], color=color1, marker='.', linestyle='none')
    pl.plot( kph, P2_data_dec[0,nt], color=color1, marker='.', linestyle='none')
    pl.plot( kph, -P2_data_dec[0,nt], color=color1, marker='v', linestyle='none')

pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(\ell)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Deconvolved spectra - $P_i^{(0)} (k)$ , $P_i^{(2)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_deconv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')




# Plot CONVOLVED spectra; same limits as above
# Monopole and quadrupole
pl.xscale('log')
pl.yscale('log')
for nt in range(ntracers):
    color1=mycolor[nt]
    # plot error bars as filled regions
    s0=np.sqrt(np.diag(relcov0_MT[nt,nt]))
    s2=np.sqrt(np.diag(relcov2_MT[nt,nt]))
    p0plus = np.minimum(yhigh,np.abs(P0_data[0,nt]*(1.0 + s0)))
    p0minus = np.maximum(ylow,np.abs(P0_data[0,nt]*(1.0 - s0)))
    p2plus = np.minimum(yhigh,np.abs(P2_data[0,nt]*(1.0 + s2)))
    p2minus = np.maximum(ylow,np.abs(P2_data[0,nt]*(1.0 - s2)))
    pl.fill_between( kph, p0minus, p0plus, color = color1, alpha=0.15)
    pl.fill_between( kph, p2minus, p2plus, color = color1, alpha=0.15)
    # plot means of sims
    pl.plot(kph, effbias[nt]**2*P0_fkp_mean[nt], color=color1, linestyle='-', linewidth=0.2)
    pl.plot(kph,-effbias[nt]**2*P0_fkp_mean[nt], color=color1, linestyle='-.', linewidth=0.2)
    pl.plot(kph,-effbias[nt]**2*P2_fkp_mean[nt], color=color1, linestyle='-.',linewidth=0.2)
    pl.plot(kph, effbias[nt]**2*P2_fkp_mean[nt], color=color1, linestyle='-', linewidth=0.2)
    pl.plot(kph, P0_mean[nt], color=color1, linestyle='-', linewidth=0.5)
    pl.plot(kph,-P0_mean[nt], color=color1, linestyle='-.', linewidth=0.5)
    pl.plot(kph,-P2_mean[nt], color=color1, linestyle='-.',linewidth=0.5)
    pl.plot(kph, P2_mean[nt], color=color1, linestyle='-', linewidth=0.5)
    # Plot theory
    pl.plot(kph, P0_model[nt], color=color1, linestyle='-', linewidth=1)
    pl.plot(kph, P2_model[nt], color=color1, linestyle='-', linewidth=1)
    # plot FKP data
    pl.plot(kph, effbias[nt]**2*P0_fkp[0,nt], color=color1, marker='x', linestyle='none')
    pl.plot(kph, effbias[nt]**2*P2_fkp[0,nt], color=color1, marker='x', linestyle='none')
    pl.plot(kph, -effbias[nt]**2*P2_fkp[0,nt], color=color1, marker='1', linestyle='none')
    # plot MT data
    pl.plot( kph, P0_data[0,nt], color=color1, marker='.', linestyle='none')
    pl.plot( kph, P2_data[0,nt], color=color1, marker='.', linestyle='none')
    pl.plot( kph, -P2_data[0,nt], color=color1, marker='v', linestyle='none')

pl.ylim([ylow,yhigh])
pl.xlim([xlow,xhigh])
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(\ell)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Convolved spectra - $\hat{P}_i^{(0)} (k)$ , $ \hat{P}_i^{(2)} (k)$ ',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_conv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')


# Error bar plots
# Plot MT estimates along with theory -- convolved spectra
pl.xscale('log')
pl.yscale('log')
ylow=np.median(P0_mean_dec[0,-10:])*0.1
yhigh=np.mean(np.abs(P0_mean_dec[:,2:10]))*3.0

for nt in range(ntracers):
    gk = 1.+0.01*(nt-1.5)
    # Monopole
    color1=mycolor[nt]
    p = P0_data[0,nt]
    errp = np.sqrt(np.diag(np.cov(P0_data[1:,nt].T)))
    label = str(nt)
    pl.errorbar(gk*kph,p,errp,color=color1,linestyle='None', marker='s', markersize=3,capsize=3, label = 'Monopole Tracer ' + label)
    #pl.plot(kph,gp*P0_model[nt],color=color1,linewidth=1.2)
    pl.plot(kph,P0_mean[nt],color=color1,linewidth=0.6)
    # Quadrupole
    p = P2_data[0,nt]
    errp = np.sqrt(np.diag(np.cov(P2_data[1:,nt].T)))
    pl.errorbar(gk*kph,p,errp,color=color1,linestyle='None',marker = '^',markersize=3,capsize=3, label = 'Quadrupole Tracer ' + label)
    #pl.plot(kph,gp*P2_model[nt],color=color1,linewidth=1.2)
    pl.plot(kph,P2_mean[nt],color=color1,linewidth=0.6)

pl.legend()
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(\ell)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Convolved spectra - $P_i^{(0)} (k)$ , $P_i^{(2)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_errbar_conv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')


# Plot estimates along with theory -- deconvolved spectra
pl.xscale('log')
pl.yscale('log')
ylow=np.median(P0_mean_dec[0,-10:])*0.1
yhigh=np.mean(np.abs(P0_mean_dec[:,2:10]))*3.0
for nt in range(ntracers):
    gp = 1.+0.02*(nt-1.5)
    gk = 1.+0.01*(nt-1.5)
    # Monopole
    color1=mycolor[nt]
    p = gp*P0_data_dec[0,nt]
    errp = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt].T)))
    pl.errorbar(gk*kph,p,errp,color=color1,linestyle='None',linewidth=0.6, marker = 's', markersize = 3, capsize=3,label='MT - Monopole')
    pl.plot(kph,gp*P0_model[nt],color=color1,linewidth=0.4)
    # Quadrupole
    p = gp*P2_data_dec[0,nt]
    errp = np.sqrt(np.diag(np.cov(P2_data[1:,nt].T)))
    pl.errorbar(gk*kph,p,errp,color=color1,linestyle='None',linewidth=0.6, marker = '^', markersize=3,capsize=3,label='MT - Quadrupole')
    pl.plot(kph,gp*P2_model[nt],color=color1,linewidth=0.4)

pl.legend()
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(\ell)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Deconvolved spectra - $P_i^{(0)} (k)$ , $P_i^{(2)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_errbar_deconv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')


# Plot estimates along with theory -- deconvolved spectra
# Monopole only
pl.xscale('log')
pl.yscale('log')
for nt in range(ntracers):
    gk = 1.+0.01*(nt-1.5)
    # Monopole
    color1=mycolor[nt]
    p = np.abs(P0_data_dec[0,nt])
    errp = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt].T)))
    pl.errorbar(gk*kph,p,errp,color=color1,linestyle='None',linewidth=0.6, marker = 's', markersize = 3, capsize=3, label = 'MT Tracer ' + str(nt))
    pl.plot(kph,P0_model[nt],color=color1,linewidth=0.4, label = 'Model Tracer ' + str(nt))

pl.legend()
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(0)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Deconvolved spectra - $P_i^{(0)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_errbar_P0_deconv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')


# Plot the ratios of the spectra with respect to the theoretical expectation
#pl.xscale('log', nonposy='clip')
pl.xscale('log')
#pl.yscale('log', nonposy='clip')
ylow = 0
yhigh= 2*ntracers + 1.

for nt in range(ntracers):
    color1=mycolor[nt]
    s0=np.sqrt(np.diag(relcov0_MT[nt,nt]))
    s2=np.sqrt(np.diag(relcov2_MT[nt,nt]))
    pp=2*nt+P0_mean_dec[nt]*(1.0 + s0)/(P0_model[nt])
    pm=2*nt+P0_mean_dec[nt]*(1.0 - s0)/(P0_model[nt])
    p0plus = np.minimum(yhigh,pp)
    p0minus = np.maximum(ylow,pm)
    pp=2*nt+1+P2_mean_dec[nt]*(1.0 + s2)/(P2_model[nt])
    pm=2*nt+1+P2_mean_dec[nt]*(1.0 - s2)/(P2_model[nt])
    p2plus = np.minimum(yhigh,pp)
    p2minus = np.maximum(ylow,pm)
    pl.fill_between( kph, p0minus, p0plus, color = color1, alpha=0.15)
    pl.fill_between( kph, p2minus, p2plus, color = color1, alpha=0.15)
    pl.plot(kph,2*nt+P0_mean_dec[nt]/(P0_model[nt]),color=color1,linestyle='-')
    pl.plot(kph,2*nt+effbias[nt]**2*P0_fkp_mean_dec[nt]/(P0_model[nt]),color=color1,linestyle='--')
    pl.plot(kph,2*nt+P0_data_dec[0,nt]/(P0_model[nt]),color=color1,marker='.', linestyle='none')
    pl.plot(kph,2*nt+1+P2_mean_dec[nt]/(P2_model[nt]),color=color1,linestyle='-')
    pl.plot(kph,2*nt+1+effbias[nt]**2*P2_fkp_mean_dec[nt]/(P2_model[nt]),color=color1,linestyle='--')
    pl.plot(kph,2*nt+1+P2_data_dec[0,nt]/(P2_model[nt]),color=color1,marker='x', linestyle='none')

pl.ylim([ylow,yhigh])
pl.xlim([xlow,xhigh])
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{\ell, D} (k)/P^{\ell, T}_i $',fontsize=14)
pl.title('Data/Theory',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_relative_errors_kbar=' + str(kph_central) + '.pdf')
pl.close('all')



#NEW FIGURES, stored in /selected directory
if not os.path.exists(dir_figs + '/selected'):
    os.makedirs(dir_figs + '/selected')

        
if plot_all_cov:
    # Plot 2D correlation of Ptot and ratios
    nblocks=1+ntracers*(ntracers-1)//2
    indexcov = np.arange(0,nblocks*num_binsk,np.int(num_binsk//4))
    nameindex = nblocks*[str(0.001*np.round(1000*kin)) for kin in kph[0:-1:np.int(num_binsk//4)]]
    onesk=np.diag(np.ones((nblocks*num_binsk)))
    dF=np.sqrt(np.abs(np.diag(cov_Pt_ratios_FKP)))
    dM=np.sqrt(np.abs(np.diag(cov_Pt_ratios_MT)))
    dF2 = small + np.outer(dF,dF)
    dM2 = small + np.outer(dM,dM)
    fullcov = np.tril(np.abs(cov_Pt_ratios_FKP)/dF2) + np.triu(np.abs(cov_Pt_ratios_MT.T)/dM2) - onesk
    pl.imshow(fullcov,origin='lower',interpolation='none')
    pl.title("Covariance of total effective power spectrum and ratios of spectra (monopoles only)")
    pl.xticks(indexcov,nameindex,size=6,name='monospace')
    pl.yticks(indexcov,nameindex,size=8,name='monospace')
    pl.annotate('FKP',(np.int(pow_bins/5.),2*pow_bins),fontsize=20)
    pl.annotate('Multi-tracer',(2*pow_bins,np.int(pow_bins/5.)),fontsize=20)
    pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
    pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
    pl.colorbar()
    pl.savefig(dir_figs + '/' +'selected/' + handle_estimates + '_2D_tot_ratios_corr_newformat.pdf')
    pl.close('all')

    # Plot 2D correlation coefficients
    num_binsk0 = 26
    kph0=kph[:num_binsk0]
    indexcov = np.arange(0,num_binsk0+1,np.int(num_binsk0//5))
    #nameindex = [str(0.001*np.round(1000*kin)) for kin in kph0[0:-1:np.int(num_binsk0/6)]]
    nameindex = [str(0.001*np.round(1000*kin))[0:5] for kin in kph0[0:num_binsk0:np.int(num_binsk0/5)]]
    onesk=np.diag(np.ones(num_binsk))
    for nt in range(ntracers):
        for ntp in range(nt,ntracers):
            kk = np.outer(kph,kph)
            FKPcov=relcov0_FKP[nt,ntp]
            MTcov=relcov0_MT[nt,ntp]
            dF=np.sqrt(np.abs(np.diag(FKPcov)))
            dM=np.sqrt(np.abs(np.diag(MTcov)))
            FKPcorr=FKPcov/(small+np.outer(dF,dF))
            MTcorr=MTcov/(small+np.outer(dM,dM))
            #fullcov = np.tril(np.abs(FKPcorr)) + np.triu(np.abs(MTcorr.T)) - onesk
            fullcov = np.tril(FKPcorr) + np.triu(MTcorr.T) - onesk  
            thistitle = 'Corr(P_' + str(nt) + ',P_' + str(ntp) + ') '
            pl.imshow(fullcov,extent=[0,26,0,26],origin='lower',interpolation='nearest',cmap='jet')
            #pl.title(thistitle)
            #print(indexcov,nameindex)   
            pl.xticks(indexcov,nameindex,size=12,name='monospace')
            pl.yticks(indexcov,nameindex,size=12,name='monospace')
            pl.annotate(r'$\bf{FKP}$',(np.int(len(kph0)//10),np.int(len(kph0)/2.5)),fontsize=28, color='white')
            pl.annotate(r'$\bf{MTOE}$',(np.int(len(kph0)//2),np.int(len(kph0)//10)),fontsize=28,color='white')
            pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=12)
            pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=12)
            #pl.xlim([0.,0.4])
            #pl.ylim([0.,0.4])
            pl.colorbar()
            pl.savefig(dir_figs + '/' + 'selected/' + handle_estimates + '_2Dcorr_' + str(nt) + '_' + str(ntp) + 'newformat.pdf')
            pl.close('all')


        
            
# Marker: open circle
mymark=[r'$\circ$']
smoothfilt = np.ones(5)/5.0
kphsmooth = np.convolve(kph,smoothfilt,mode='valid')
    
    
# new plot!!! same as above but for MT/FKP gain
for nt in range(ntracers):
    #tcfkp = np.convolve(Theor_Cov_FKP[nt]/(small+P0_fkp_mean[nt]**2),smoothfilt,mode='valid')
    #pl.plot(kphsmooth,np.sqrt(np.abs(tcfkp)),'r-',linewidth=0.5)
    color1=mycolor[nt]
    fkpcov= np.sqrt(np.abs(np.diagonal(relcov0_FKP[nt,nt])))
    mtcov = np.sqrt(np.abs(np.diagonal(relcov0_MT[nt,nt])))
    pl.plot(kph,(fkpcov/fkpcov)*1.0,color='k',linestyle='--')
    pl.plot(kph,mtcov/fkpcov,marker='o',color=color1,markersize=5,linestyle='-',label='Tracer '+str(nt+1))
ymin, ymax = 0.0,1.3#1.2
#pl.xlim([kph[2],kph[-2]])
pl.xlim([kph[0],0.3])
pl.ylim([ymin,ymax])
#pl.text(0.3,1.07,"LC-selection",fontsize=14)
#pl.text(0.3,1.17,"Mass selection",fontsize=14)
pl.legend(loc="upper right",fontsize=7,frameon=False)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$\sigma [P^{MT}]/[P^{MT}] / \sigma [P^{FKP}]/[P^{FKP}] $',fontsize=14)
#thistitle = 'Gain on variance of tracer ' + str(nt)
#pl.title(thistitle,fontsize=16)
pl.savefig(dir_figs + '/' + 'selected/' + handle_estimates  +'_sigmas_gain.pdf')
pl.close('all')    



# new plot!!! same as above but for MT/FKP gain

for nt in range(ntracers):
    #tcfkp = np.convolve(Theor_Cov_FKP[nt]/(small+P0_fkp_mean[nt]**2),smoothfilt,mode='valid')
    #pl.plot(kphsmooth,np.sqrt(np.abs(tcfkp)),'r-',linewidth=0.5)
    color1=mycolor[nt]
    fkpcov= np.sqrt(np.abs(np.diagonal(relcov0_FKP[nt,nt])))
    mtcov = np.sqrt(np.abs(np.diagonal(relcov0_MT[nt,nt])))
    pl.plot(kph,(fkpcov/fkpcov)*1.0,color='k',linestyle='--')
    pl.plot(kph,mtcov/fkpcov,marker='o',color=color1,markersize=5,linestyle='-',label='Tracer '+str(nt+1))
ymin, ymax = 0.0,1.3#1.2
#pl.xlim([kph[2],kph[-2]])
pl.xlim([kph[0],0.3])
pl.ylim([ymin,ymax])
#pl.text(0.3,1.07,"LC-selection",fontsize=14)
#pl.text(0.3,1.17,"LC-selection",fontsize=14)
pl.legend(loc="upper right",fontsize=8,frameon=False)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$\sigma [P^{MT}]/[P^{MT}] / \sigma [P^{FKP}]/[P^{FKP}] $',fontsize=14)
#thistitle = 'Gain on variance of tracer ' + str(nt)
#pl.title(thistitle,fontsize=16)
pl.savefig(dir_figs + '/' + 'selected/' + handle_estimates + '_sigmas_gain_2.pdf')
pl.close('all')    
    


##color_code = ['blue','cyan','lime','orange','red','maroon','pink','green','black','grey','navy','salmon','purple']

if plot_all_cov:
    ntcount=0
    if ntracers > 1:
        for nt in range(ntracers):
            for ntp in range(nt+1,ntracers):
                #color1=color_code[ntcount]
                fkpcov= np.sqrt(np.diagonal(fraccov0_FKP[ntcount]))
                mtcov = np.sqrt(np.diagonal(fraccov0_MT[ntcount]))
                pl.plot(kph,(fkpcov/fkpcov)*1.0,color='k',linestyle='--')
                pl.plot(kph,mtcov/fkpcov,marker='o',markersize = 5.0,linestyle='-', label='ratio '+str(nt+1)+'-'+str(ntp+1))
                ntcount = ntcount + 1

#pl.legend(loc="upper right",fontsize=6,frameon=False)
ymin, ymax = 0.0,1.5
#pl.xlim([kph[2],kph[-5]])
pl.xlim([kph[0],0.3])
pl.ylim([ymin,ymax])
#pl.text(0.3,1.07,"LC-selection",fontsize=14)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$\sigma(P_\mu/P_\nu)_{MT}/\sigma(P_\mu/P_\nu)_{FKP}$',fontsize=12)
#thistitle = 'MT/FKP gain on ratios of spectra for ' + str(nt+1) + ' , ' + str(ntp+1)
#pl.title(thistitle,fontsize=16)
pl.savefig(dir_figs + '/' + 'selected/' +handle_estimates + '_frac_cov_gain.pdf')
pl.close('all')
               

  
# Plot estimates along with theory -- deconvolved spectra
# Monopole only
pl.xscale('log')
pl.yscale('log')
#ylow=np.median(P0_mean_dec[0,-10:])*0.1
#yhigh=np.mean(np.abs(P0_mean_dec[:,2:10]))*3.0
ylow=400.
yhigh=120000.


for nt in range(ntracers):
    gp = 1.#+0.02*(nt-1.5)
    gk = 1.#+0.01*(nt-1.5)
    gp2 = 1.#+0.04*(nt-1.5)
    gk2 = 1.#+0.03*(nt-1.5)
    # Monopole
    color1=mycolor[nt]
    p = np.abs(gp*P0_data_dec[0,nt])
    p_mean = np.abs(gp*np.mean(P0_data_dec[:,nt,:],axis=0))
    p2 = gp2*effbias[nt]**2*P0_fkp_dec[0,nt]
    # print(P0_fkp[:,nt].shape)
    p2_mean = gp2*effbias[nt]**2*(np.mean(P0_fkp_dec[:,nt,:],axis=0))
    #p2 = gp*effbias[nt]**2*P0_fkp[0,nt]
    errp = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt].T)))
    errp2 = np.sqrt(np.diag(np.cov(effbias[nt]**2*P0_fkp_dec[1:,nt].T)))
    if nt==0:
        pl.errorbar(gk*kph,p_mean,errp,color=color1,linestyle ='None', marker='^', capsize = 3, markersize = 3,elinewidth=1.2, label='MT (mean of mocks)')
        #pl.errorbar(gk2*kph,p2,errp2,color=color1,linestyle ='None', marker='x',ms=3.,elinewidth=1.2, label = 'FKP')
        pl.errorbar(gk2*kph,p2_mean,errp2,color=color1,linestyle ='None', marker='x',capsize=3, markersize = 4,elinewidth=1.2, label = 'FKP (mean of mocks)')
        pl.plot(kph,gp*P0_model[nt],color=color1,linewidth=1.,label='Theoretical model')
        ##pl.plot(kph,p_mean,color=color1,linewidth=2.,linestyle='--',label='Mean of mocks')
    else:
        pl.errorbar(gk*kph,p_mean,errp,color=color1,linestyle ='None', marker='^', capsize=3, markersize=3,elinewidth=1.2)
        #pl.errorbar(gk2*kph,p2,errp2,color=color1,linestyle ='None', marker='x',ms=3.,elinewidth=1.2)
        pl.errorbar(gk2*kph,p2_mean,errp2,color=color1,linestyle ='None', marker='x',capsize=3,markersize=3,elinewidth=1.2)
        pl.plot(kph,gp*P0_model[nt],color=color1,linewidth=1.)
        ##pl.plot(kph,p_mean,color=color1,linewidth=2.,linestyle='--')
        
#pl.text(0.3,20000,r"$0.6<z<0.75$", fontsize = 12)
#pl.text(0.18,20000,"W4 field", fontsize = 12)
pl.legend(loc="lower left",fontsize=12.,frameon=False)
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(0)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
#pl.title(r'Deconvolved spectra - $P_i^{(0)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + 'selected/' + handle_estimates + '_errbar_P0_deconv_kbar=' + str(kph_central) + '_paper_new.pdf')
pl.close('all')







print('Figures created and saved in /fig .')

print ()
print('----------------------------------')
print ()



######################
# Now writing results to file
print('Writing results to /spectra/' + handle_estimates)

# Full dataset. Stack the spectra, then take covariances
p_stack = np.zeros((n_maps,2*ntracers*num_binsk),dtype="float32")
p_stack_FKP = np.zeros((n_maps,2*ntracers*num_binsk),dtype="float32")
p_theory = np.zeros((num_binsk,2*ntracers),dtype="float32")
p_data = np.zeros((num_binsk,2*ntracers),dtype="float32")

for nt in range(ntracers):
    p_stack[:,2*nt*num_binsk:(2*nt+1)*num_binsk] = P0_data_dec[:,nt]
    p_stack[:,(2*nt+1)*num_binsk:(2*nt+2)*num_binsk] = P2_data_dec[:,nt]
    p_stack_FKP[:,2*nt*num_binsk:(2*nt+1)*num_binsk] = effbias[nt]**2*P0_fkp_dec[:,nt]
    p_stack_FKP[:,(2*nt+1)*num_binsk:(2*nt+2)*num_binsk] = effbias[nt]**2*P2_fkp_dec[:,nt]
    p_theory[:,2*nt] = P0_model[nt]
    p_theory[:,2*nt+1] = P2_model[nt]
    p_data[:,2*nt] = P0_data_dec[0,nt]
    p_data[:,2*nt+1] = P2_data_dec[0,nt]


# Return bias to the computed cross-spectra (done in FKP) as well:
index=0
for i in range(ntracers):
    for j in range(i+1,ntracers):
        Cross0[:,index] = effbias[i]*effbias[j]*Cross0[:,index]
        Cross0_dec[:,index] = effbias[i]*effbias[j]*Cross0_dec[:,index]
        Cross2[:,index] = effbias[i]*effbias[j]*Cross2[:,index]
        Cross2_dec[:,index] = effbias[i]*effbias[j]*Cross2_dec[:,index]
        index += 1


# Export data
np.savetxt(dir_specs + '/' + handle_estimates + '_vec_k.dat',kph,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_P0_P2_theory.dat',p_theory,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_P0_P2_data.dat',p_data,fmt="%6.4f")

np.savetxt(dir_specs + '/' + handle_estimates + '_decP_k_data_MT.dat',p_stack.T,fmt="%6.2f")
np.savetxt(dir_specs + '/' + handle_estimates + '_decP_k_data_FKP.dat',p_stack_FKP.T,fmt="%6.2f")

np.savetxt(dir_specs + '/' + handle_estimates + '_nbar_mean.dat',nbarbar,fmt="%2.6f")
np.savetxt(dir_specs + '/' + handle_estimates + '_bias.dat',gal_bias,fmt="%2.3f")
np.savetxt(dir_specs + '/' + handle_estimates + '_effbias.dat',effbias,fmt="%2.3f")


# Saving cross spectra

np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P0_theory.dat',Cross_P0_theory.T,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P0_model.dat',Cross_P0_model.T,fmt="%6.4f")

np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P2_theory.dat',Cross_P2_theory.T,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P2_model.dat',Cross_P2_model.T,fmt="%6.4f")

Cross0=np.reshape(Cross0,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
Cross2=np.reshape(Cross2,(n_maps,ntracers*(ntracers-1)//2*pow_bins))

Cross0_dec=np.reshape(Cross0_dec,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
Cross2_dec=np.reshape(Cross2_dec,(n_maps,ntracers*(ntracers-1)//2*pow_bins))

np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P0_data.dat',Cross0.T,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_dec_P0_data.dat',Cross0_dec.T,fmt="%6.4f")

np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P2_data.dat',Cross2.T,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_dec_P2_data.dat',Cross2_dec.T,fmt="%6.4f")


########################################
# Window function
wf = np.zeros((2*ntracers,num_binsk),dtype="float32")
for k in range(pow_bins):
    wf[:,k]  = np.hstack((winfun0[:,k],winfun2[:,k]))

# make sure that window function is not zero
wf[wf==0]=1.0
wf[np.isnan(wf)]=1.0
np.savetxt(dir_specs + '/' + handle_estimates + '_WinFun02_k_data.dat',wf.T,fmt="%3.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_WinFun_Cross0_k_data.dat',wf_c0,fmt="%3.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_WinFun_Cross2_k_data.dat',wf_c2,fmt="%3.4f")


####################################
# Checking cross-correlations
print()

index=0
for nt in range(ntracers):
    for ntp in range(nt+1,ntracers):
        print("Cross-corr of P^0 of tracers", nt, ntp, " -- dec/model, dec/theory")
        var = np.array([ effbias[nt]*effbias[ntp]*Cross0_mean_dec[index]/Cross_P0_model[index] , effbias[nt]*effbias[ntp]*Cross0_mean_dec[index]/Cross_P0_model[index] ])
        print(var.T)
        index += 1
        print()

print()
index=0
for nt in range(ntracers):
    for ntp in range(nt+1,ntracers):
        print("Cross-corr of P^0 x Auto-corr of P^0 of tracers", nt, ntp, " -- P_ij^2[fkp]/P_i[MTOE] P_j[MTOE]")
        print(effbias[nt]**2*effbias[ntp]**2*Cross0_mean_dec[index]**2/P0_mean_dec[nt]/P0_mean_dec[ntp])
        print()
        index += 1

print()
index=0
for nt in range(ntracers):
    for ntp in range(nt+1,ntracers):
        print("Cross-corr of P^0 x Auto-corr of P^0 of tracers", nt, ntp, " -- P_ij^2[fkp]/P_i[FKP] P_j[FKP]")
        print(Cross0_mean_dec[index]**2/P0_fkp_mean_dec[nt]/P0_fkp_mean_dec[ntp])
        print()
        index += 1


####################################
# The measured monopoles and the flat-sky/theory monopoles are different; find relative normalization

normmonos_data = np.zeros(ntracers)
normmonos_mocks = np.zeros(ntracers)
chi2_red_data = np.zeros(ntracers)
chi2_red_mocks = np.zeros(ntracers)

for nt in range(ntracers):
    err = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt,kb_cut_min:kb_cut_max].T)))
    data = [ P0_data_dec[0,nt,kb_cut_min:kb_cut_max]/err , P0_theory[nt,kb_cut_min:kb_cut_max]/err ]
    this_norm, success = leastsq(residuals,1.0,args=data)
#    print weights_mono
    normmonos_data[nt] = np.sqrt(this_norm)
#    norm_monos[nt] = np.mean(np.sqrt(P0_data_dec[0,nt,kb_cut_min:kb_cut_max]/P0_mean_dec[nt,kb_cut_min:kb_cut_max]))
    chi2_red_data[nt] = np.sum(residuals(this_norm,data)**2)/(len(err)-1.)

    mock_data = [ P0_mean_dec[nt,kb_cut_min:kb_cut_max]/err , P0_theory[nt,kb_cut_min:kb_cut_max]/err ]
    this_norm, success = leastsq(residuals,1.0,args=mock_data)
#    print weights_mono
    normmonos_mocks[nt] = np.sqrt(this_norm)
#    norm_monos[nt] = np.mean(np.sqrt(P0_data_dec[0,nt,kb_cut_min:kb_cut_max]/P0_mean_dec[nt,kb_cut_min:kb_cut_max]))
    chi2_red_mocks[nt] = np.sum(residuals(this_norm,mock_data)**2)/(len(err)-1.)

print()
print ("Fiducial biases of the mocks, and best-fit values from theory (for updating values in the input file, or HOD)")
for nt in range(ntracers):
    print ('Tracer ', nt, ': fiducial bias = ', np.around(gal_bias[nt],3), ' ; best fit:', np.around(normmonos_mocks[nt]*gal_bias[nt],3), ' (chi^2 = ', np.around(chi2_red_mocks[nt],3), ')')
print()

try:
    data_bias
    print ("Fiducial biases of the data, and best-fit values from theory (for updating values in the input file, or HOD)")
    for nt in range(ntracers):
        print ('Tracer ', nt, ' -- data: fiducial bias = ', np.around(data_bias[nt],3), ' ; update this to:', np.around(normmonos_data[nt]*data_bias[nt],3), ' (chi^2 = ', np.around(chi2_red_data[nt],3), ')')
except:
    print ("Fiducial biases of the data, and best-fit values from theory (for updating values in the input file, or HOD)")
    for nt in range(ntracers):
        print ('Tracer ', nt, ' -- data: fiducial bias = ', np.around(gal_bias[nt],3), ' ; update this to:', np.around(normmonos_data[nt]*gal_bias[nt],3), ' (chi^2 = ', np.around(chi2_red_data[nt],3), ')')

print()
print ("Quick update bias (from MT estimator):")
for nt in range(ntracers):
    print (np.around(normmonos_mocks[nt]*gal_bias[nt],3))

sys.exit(-1)


