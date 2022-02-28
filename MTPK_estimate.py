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
import h5py
import glob
from time import time , strftime
from scipy import interpolate
from scipy import special
from scipy.optimize import leastsq
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import vstack
from scipy.ndimage import gaussian_filter

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


# My classes -- functions used by the MTPK suite
import fkp_multitracer as fkpmt
import fkp_class as fkp  # This is the new class, which computes auto- and cross-spectra
import pk_multipoles_gauss as pkmg
import pk_crossmultipoles_gauss as pkmg_cross
from camb_spec import camb_spectrum
from cosmo_funcs import matgrow, H
from analytical_selection_function import *
import grid3D as gr
#The following two classes have the inttention to substitute the old program of inputs
from cosmo import cosmo #class to cosmological parameter
from code_options import code_parameters #class to code options



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
# Parameters for this run
#####################################################
my_cosmology = cosmo() #This stantiates the class with default parameters
'''
To change the cosmological parameters at all, change in this way:
my_cosmology = cosmo(x = y)
where x is the parameter and y its new value
'''
physical_options = my_cosmology.default_params #This returns a dictionary with all the default parameters

print("##########################")
print('The cosmology is')
print("##########################")
# 1) Using the method cosmo_print
print(my_cosmology.cosmo_print())
# # 2) Using dictionary directly:
# print(physical_options)
# # 3) Printing the cosmology without directly call for the method cosmo_print
# print(my_cosmology.cosmo_print)

print()


# ##########################
# '''
# Local changes
# '''
# #Example of changing cosmology
# physical_options['h'] = 0.72
# # 1) Using the method cosmo_print
# print(my_cosmology.cosmo_print())
# # # 2) Using dictionary directly
# # print(physical_options)
# # # 3) Printing the cosmology without directly call for the method cosmo_print
# # print(my_cosmology.cosmo_print)
# # print()
# # print()
# ##########################

# #Testing methods of cosmos class
# print("Testing f_evolving(z = 1)", my_cosmology.f_evolving(1.0) )
# print("Testing f_phenomenological", my_cosmology.f_phenomenological() )
# print("Testing H(z = 0)", my_cosmology.H(0, True) )
# print("Testing H(z = 0)", my_cosmology.H(0, False) )
# print("Testing cosmological distance: z = 0:", my_cosmology.comoving(0, True) )
# print("Testing cosmological distance: z = 0:", my_cosmology.comoving(0, False) )
# print("Testing cosmological distance: z = 1:", my_cosmology.comoving(1., True) )
# print("Testing cosmological distance: z = 1:", my_cosmology.comoving(1., False) )
# print('chi_h: z = 0:', my_cosmology.chi_h(0.0))
# print('chi_h: z = 1:', my_cosmology.chi_h(1.0))

my_code_options = code_parameters() #This stantiates the class
parameters_code = my_code_options.default_params #This returns a dictionary with all the default parameters
print("##########################")
print('The code options are')
print("##########################")
'''
To change the code parameters at all, change in this way:
my_code_options = code_parameter(x = y)
where x is the parameter and y its new value
'''
# 1) Using the method parameters_print
print(my_code_options.parameters_print())
# # 2) Using dictionary directly
# print(parameters_code)
# # 3) Printing the cosmology without directly call for the method parameters_print
# print(my_code_options.parameters_print)
print()

# ##########################
# '''
# Local changes
# '''
# #Example of code parameters
# parameters_code['cell_size'] = 0.9
# # 1) Using the method parameters_print
# print(my_code_options.parameters_print())
# # # 2) Using dictionary directly
# # print(parameters_code)
# # # 3) Printing the cosmology without directly call for the method cosmo_print
# # print(my_code_options.parameters_print)
# print()
##########################

#Some other cosmological quantities
h = physical_options['h']
H0 = h*100.
clight = physical_options['c_light']
cH = clight*h/my_cosmology.H(physical_options['zcentral'], False) # c/H(z) , in units of h^-1 Mpc


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

mas_method = parameters_code['mas_method']
if mas_method == 'NGP':
    mas_power = 1.0
elif mas_method == 'CIC':
    mas_power = 2.0
elif mas_method == 'TSC':
    mas_power = 3.0
elif mas_method == 'PCS':
    mas_power = 4.0
else:
    print("Please specify an acceptable Mass Assignement Scheme")
    print("Acceptable values: NGP, CIC, TSC and PCS")
    print("Aborting now...")
    sys.exit(-1)

# Directory with data and simulated maps
dir_maps = this_dir + '/maps/sims/' + handle_sims

# Directory with data
use_mask = parameters_code['use_mask']
sel_fun_data = parameters_code['sel_fun_data']
dir_data = this_dir + '/maps/data/' + handle_data


# Will save results of the estimations to these directories:
dir_specs = this_dir + '/spectra/' + handle_estimates
dir_figs = this_dir + '/figures/' + handle_estimates
if not os.path.exists(dir_specs):
    os.makedirs(dir_specs)
if not os.path.exists(dir_figs):
    os.makedirs(dir_figs)

# Save estimations for each assumed k_phys in subdirectories named after k_phys
strkph = str(parameters_code['kph_central'])

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

Omegam = physical_options['Omega0_m']
OmegaDE = physical_options['Omega0_DE']
Omegab = physical_options['Omega0_b']
Omegac = physical_options['Omega0_cdm']
A_s = physical_options['A_s']
gamma = physical_options['gamma']
matgrowcentral = physical_options['matgrowcentral']
w0 = physical_options['w0']
w1 = physical_options['w1']
z_re = physical_options['z_re']
zcentral = physical_options['zcentral']
n_SA = physical_options['n_s']

# Velocity dispersion. vdisp is defined on inputs with units of km/s
vdisp = np.asarray(my_code_options.vdisp) #km/s
sigma_v = my_code_options.vdisp/my_cosmology.H(physical_options['zcentral'], False) #Mpc/h
a_vdisp = vdisp/physical_options['c_light'] #Adimensional vdisp

# Redshift errors. sigz_est is defined on inputs, and is adimensional
sigz_est = np.asarray(my_code_options.sigz_est)
sigma_z = sigz_est*physical_options['c_light']/my_cosmology.H(physical_options['zcentral'], False) # Mpc/h

whichspec = parameters_code['whichspec']

# Joint factor considering v dispersion and z error
sig_tot = np.sqrt(sigma_z**2 + sigma_v**2) #Mpc/h
a_sig_tot = np.sqrt(sigz_est**2 + a_vdisp**2) #Adimensional sig_tot

###################################################################################################


#############Calling CAMB for calculations of the spectra#################
print('Beggining CAMB calculations\n')

use_theory_spectrum = parameters_code['use_theory_spectrum']
theory_spectrum_file = parameters_code['theory_spectrum_file']
if use_theory_spectrum:
    print('Using pre-existing power spectrum in file:',theory_spectrum_file)
    kcpkc = np.loadtxt(theory_spectrum_file)
    if kcpkc.shape[1] > kcpkc.shape[0]: 
        k_camb=kcpkc[0]
        Pk_camb=kcpkc[1]
    else:
        k_camb=kcpkc[:,0]
        Pk_camb=kcpkc[:,1]
else:
    print('Computing matter power spectrum for given cosmology...\n')

    # It is strongly encouraged to use k_min >= 1e-4, since it is a lot faster
    k_min_camb = parameters_code['k_min_CAMB']
    k_max_camb = parameters_code['k_max_CAMB']

    nklist = 1000
    k_camb = np.logspace(np.log10(k_min_camb),np.log10(k_max_camb),nklist)

    kc, pkc = camb_spectrum(H0, Omegab, Omegac, w0, w1, z_re, zcentral, A_s, n_SA, k_min_camb, k_max_camb, whichspec)[:2]
    Pk_camb = np.asarray( np.interp(k_camb, kc, pkc) )

# Ended CAMB calculation #####################################

# Construct spectrum that decays sufficiently rapidly, and interpolate, using an initial ansatz for power-law of P ~ k^(-1) [good for HaloFit]
k_interp = np.append(k_camb,np.array([2*k_camb[-1],4*k_camb[-1],8*k_camb[-1],16*k_camb[-1],32*k_camb[-1],64*k_camb[-1],128*k_camb[-1]]))
P_interp = np.append(Pk_camb,np.array([1./2.*Pk_camb[-1],1./4*Pk_camb[-1],1./8*Pk_camb[-1],1./16*Pk_camb[-1],1./32*Pk_camb[-1],1./64*Pk_camb[-1],1./128*Pk_camb[-1]]))
pow_interp=interpolate.PchipInterpolator(k_interp,P_interp)

#####################################################
#####################################################
#####################################################

gal_bias = my_code_options.bias_file
adip = my_code_options.adip

gal_adip = np.asarray(adip)
gal_sigz_est = np.asarray(sigz_est)
gal_vdisp = np.asarray(vdisp)
a_gal_sig_tot = np.sqrt((gal_vdisp/clight)**2 + gal_sigz_est**2)

#####################################################
# Generate real- and Fourier-space grids for FFTs
#####################################################

print('.')
print('Generating the k-space Grid...')
print('.')

n_x = parameters_code['n_x']
n_y = parameters_code['n_y']
n_z = parameters_code['n_z']
n_x_orig = parameters_code['n_x_orig']
n_y_orig = parameters_code['n_y_orig']
n_z_orig = parameters_code['n_z_orig']

use_padding = parameters_code['use_padding']
padding_length = parameters_code['padding_length']
if use_padding:
    n_x_box = n_x + 2*padding_length[0]
    n_y_box = n_y + 2*padding_length[1]
    n_z_box = n_z + 2*padding_length[2]
    n_x_orig = n_x_orig - padding_length[0]
    n_y_orig = n_y_orig - padding_length[1]
    n_z_orig = n_z_orig - padding_length[2]
else:
    n_x_box = n_x
    n_y_box = n_y
    n_z_box = n_z

cell_size = parameters_code['cell_size']
ntracers = my_code_options.ntracers
nbar = my_code_options.nbar
ncentral = my_code_options.ncentral
nsigma = my_code_options.nsigma

L_x = n_x_box*cell_size ; L_y = n_y_box*cell_size ; L_z = n_z_box*cell_size
grid = gr.grid3d(n_x_box,n_y_box,n_z_box,L_x,L_y,L_z)

grid_orig = gr.grid3d(n_x,n_y,n_z,n_x*cell_size,n_y*cell_size,n_z*cell_size)

sel_fun_file = parameters_code['sel_fun_file']

# Selection function
# If given by data
if sel_fun_data:
    try:
        h5map = h5py.File(dir_data + '/' + sel_fun_file,'r')
        h5data = h5map.get(list(h5map.keys())[0])
        nbm = np.asarray(h5data,dtype='float32')
        #updated to shift selection function by small=1.e-9
        mult_sel_fun = parameters_code['mult_sel_fun']
        shift_sel_fun = parameters_code['shift_sel_fun']
        nbm = np.asarray(small + mult_sel_fun*(nbm + shift_sel_fun),dtype='float32')
        h5map.close
    except:
        print ('Could not find file with data selection function!')
        print ('Check your directory ', dir_data)
        print ('Aborting now...')
        sys.exit(-1)
    if len(nbm.shape)==3:
        n_bar_matrix_fid = np.zeros((1,n_x,n_y,n_z))
        n_bar_matrix_fid[0] = nbm
    elif len(nbm.shape)==4:
        n_bar_matrix_fid = nbm
        if (np.shape(n_bar_matrix_fid)[1] != n_x) or (np.shape(n_bar_matrix_fid)[2] != n_y) or (np.shape(n_bar_matrix_fid)[3] != n_z):
            print ('WARNING!!! Dimensions of data selection function box =', n_bar_matrix_fid.shape, ' , differ from input file!')
            print ('Please correct/check input files and/or maps. Aborting now.')
            sys.exit(-1)
    else:
        print ('WARNING!!! Data selection function has funny dimensions:', nbm.shape)
        print ('Please check, something is not right here. Aborting now...')
        sys.exit(-1)
else:
    try:
        mult_sel_fun = parameters_code['mult_sel_fun']
        mass_fun = parameters_code['mass_fun']
        mass_fun = mult_sel_fun*mass_fun
        nbar = mass_fun*cell_size**3
        ncentral = 10.0*nbar**0
        nsigma = 10000.0*nbar**0 
    except:
        print("Att.: using analytical selection function for galaxies (check parameters in input file).")
        print("Using n_bar, n_central, n_sigma  from input file")

    n_bar_matrix_fid = np.zeros((ntracers,n_x,n_y,n_z),dtype='float32')
    for nt in range(ntracers):
        # Here you can choose how to call the selection function, using the different dependencies
        # Exponential/Gaussian form:
        try:
            n_bar_matrix_fid[nt] = selection_func_Gaussian(grid_orig.grid_r, nbar[nt],ncentral[nt],nsigma[nt])
        except:  # If there is only one species of tracer, the nbar, ncentral etc. are not arrays
            n_bar_matrix_fid[nt] = selection_func_Gaussian(grid_orig.grid_r, nbar,ncentral,nsigma)
        # Linear form:
        #nbar_sel_fun = selection_func_Linear(grid.RX, grid.RY, grid.RZ, nbar[ng],ax[ng],ay[ng],az[ng])

        
if use_mask:
    try:
        h5map = h5py.File(dir_data + '/' + mask_filename,'r')
        h5data = h5map.get(list(h5map.keys())[0])
        mask = np.asarray(h5data,dtype='int32')
        h5map.close
    except:
        print ('Could not find file with mask!')
        print ('Check your directory ', dir_data)
        print ('Aborting now...')
        sys.exit(-1)
    if (np.shape(mask)[0] != n_x) or (np.shape(mask)[1] != n_y) or (np.shape(mask)[2] != n_z):
        print ('WARNING!!! Dimensions of mask, =', mask.shape, ' , differ from input file!')
        print ('Please correct/check input files and/or maps. Aborting now.')
        sys.exit(-1)
    n_bar_matrix_fid = n_bar_matrix_fid * mask

# Apply padding, if it exists
try:
    n_box = np.zeros((ntracers,n_x_box,n_y_box,n_z_box))
    n_box[:,padding_length[0]:-padding_length[0],padding_length[1]:-padding_length[1],padding_length[2]:-padding_length[2]] = n_bar_matrix_fid
    n_bar_matrix_fid = np.copy(n_box)
    n_box = None
    del n_box
except:
    pass
    
n_maps = parameters_code['n_maps']
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

## !! NEW !! Low-cell-count threshold. Will apply to data AND to mocks
## We will treat this as an additional MASK (thresh_mask) for data and mocks
use_cell_low_count_thresh = parameters_code['use_cell_low_count_thresh']
cell_low_count_thresh = parameters_code['cell_low_count_thresh']
if use_cell_low_count_thresh:
    cell_low_count_thresh = parameters_code['cell_low_count_thresh']
    thresh_mask = np.ones_like(n_bar_matrix_fid)
    thresh_mask[n_bar_matrix_fid < cell_low_count_thresh] = 0.0
    n_bar_matrix_fid = thresh_mask * n_bar_matrix_fid
else:
    pass


print(".")

print ('Geometry: (nx,ny,nz) = (' +str(n_x)+','+str(n_y)+','+str(n_z)+'),  cell_size=' + str(cell_size) + ' h^-1 Mpc')
# Apply padding, if it exists
try:
    print ('Geometry including bounding box: (nx,ny,nz) = (' +str(n_x_box)+','+str(n_y_box)+','+str(n_z_box) + ')')
except:
    pass

print(".")
if whichspec == 0:
    print ('Using LINEAR power spectrum from CAMB')
elif whichspec == 1:
    print ('Using power spectrum from CAMB + HaloFit')
else:
    print ('Using power spectrum from CAMB + HaloFit with PkEqual')

print(".")
print ('----------------------------------')
print(".")



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
nn = int(np.sqrt(n_x_box**2 + n_y_box**2 + n_z_box**2))
kk_bar = np.fft.fftfreq(nn)


### K_MAX_MIN
#  Maximum ***frequency*** allowed
#  Nyquist frequency is 0.5 (in units of 1/cell)
use_kmax_phys = parameters_code['use_kmax_phys']
if use_kmax_phys:
    kmax_phys = parameters_code['kmax_phys']
    kmaxbar = min(0.5,kmax_phys*cell_size/2.0/np.pi)
    kmax_phys = kmaxbar*2*np.pi/cell_size
else:
    kmaxbar = 0.5  # Use Nyquist frequency in cell units
    kmax_phys = np.pi/cell_size
    
# The number of bins should be set by the maximum k to be computed,
# together with the minimum separation based on the physical volume.
#
# Typically, dk_phys =~ 1.4/L , where L is the typical physical size
# of the survey (Abramo 2012)
# For some particular applications (esp. large scales) we can use larger bins
#dk0=1.4/np.power(n_x*n_y*n_z,1/3.)/(2.0*np.pi)
dk0 = 3.0/np.power(n_x_box*n_y_box*n_z_box,1/3.)/(2.0*np.pi)
dk_phys = 2.0*np.pi*dk0/cell_size

# Ensure that the binning is at least a certain size
dkph_bin = parameters_code['dkph_bin']
dk_phys = max(dk_phys,dkph_bin)
# Fourier bins in units of frequency
dk0 = dk_phys*cell_size/2.0/np.pi

#  Physically, the maximal useful k is perhaps k =~ 0.3 h/Mpc (non-linear scale)
np.set_printoptions(precision=3)

print ('Will estimate modes up to k[h/Mpc] = ', '%.4f'% kmax_phys,' in bins with Delta_k =', '%.4f' %dk_phys)

print(".")
print ('----------------------------------')
print(".")

#R This line makes some np variables be printed with less digits
np.set_printoptions(precision=6)


#R Here are the k's that will be estimated (in grid units):
kgrid = grid.grid_k
kminbar = 1./4.*(kgrid[1,0,0]+kgrid[0,1,0]+kgrid[0,0,1]) + dk0/4.0

### K_MAX_MIN
use_kmin_phys = parameters_code['use_kmin_phys']
if use_kmin_phys:
    kmin_phys = parameters_code['kmin_phys']
    kminbar = kmin_phys*cell_size/2.0/np.pi
else:
    pass

### K_MAX_MIN
num_binsk = int((kmaxbar-kminbar)/dk0)
dk_bar = dk0*np.ones(num_binsk)
k_bar = kminbar + dk0*np.arange(num_binsk)
r_bar = 1/2.0 + ((1.0*n_x_box)/num_binsk)*np.arange(num_binsk)

#
# k in physical units
#
kph = k_bar*2*np.pi/cell_size



##############################################
# Define the "effective bias" as the amplitude of the monopole
use_kdip_phys = parameters_code['use_kdip_phys']
if use_kdip_phys:
    kdip_phys = parameters_code['kdip_phys']
    print ('ATTENTION: pre-defined (on input) alpha-dipole k_dip [h/Mpc]=', '%1.4f'%kdip_phys)
    pass
else:
    kdip_phys = 1./(cell_size*(n_z_orig + n_z/2.))    
    
try:
    dip = np.asarray(gal_adip) * kdip_phys
except:
    dip = 0.0

pk_mg = pkmg.pkmg(gal_bias,dip,matgrowcentral,k_camb,a_gal_sig_tot,cH,zcentral)

monopoles = pk_mg.mono
quadrupoles = pk_mg.quad

# Hexadecapoles only in the Kaiser approximation
hexa = 8./35*matgrowcentral**2

hexapoles = hexa*np.power(monopoles,0)

try:
    pk_mg_cross = pkmg_cross.pkmg_cross(gal_bias,dip,matgrowcentral,k_camb,a_gal_sig_tot,cH,zcentral)
    cross_monopoles = pk_mg_cross.monos
    cross_quadrupoles = pk_mg_cross.quads
except:
    cross_monopoles = np.zeros((len(k_camb),1))
    cross_quadrupoles = np.zeros((len(k_camb),1))

cross_hexapoles = hexa*np.power(cross_monopoles,0)

# Compute effective dipole and bias of tracers
kph_central = my_code_options.kph_central
where_kph_central = np.argmin(np.abs(k_camb - kph_central))

effadip = dip*matgrowcentral/(0.00000000001 + kph_central)
effbias = np.sqrt(monopoles[:,where_kph_central])

# Get effective bias (sqrt of monopoles) for final tracers
pk_mg = pkmg.pkmg(gal_bias,gal_adip,matgrowcentral,k_camb,a_gal_sig_tot,cH,zcentral)

monopoles = pk_mg.mono
quadrupoles = pk_mg.quad

# Compute effective dipole and bias of tracers
where_kph_central = np.argmin(np.abs(k_camb - kph_central))

effadip = gal_adip*matgrowcentral/(0.00000000001 + kph_central)
effbias = np.sqrt(monopoles[:,where_kph_central])

print()
print ('----------------------------------')
print()



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

kflat=(np.ndarray.flatten(kgrid[:,:,:n_z_box//2+1]))
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

print ('Done with k-binning matrices. Time cost: ', np.int32((time()-tempor)*1000)/1000., 's')
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

###
#Corrections due to bin averaging
###
Pk_flat_camb = np.asarray( np.interp(kflat*2*np.pi/cell_size, k_camb, Pk_camb) )
Pk_av=(((MRk*Pk_flat_camb))/(kkbar_counts+0.00001))
k_av_corr= Pk_av/(powtrue + 0.000001*np.min(powtrue))

if (np.any(k_av_corr > 2.0) | np.any(k_av_corr < 0.5)):
    print("    !!  Warning: Fourier bin averaging may severely under/overestimating your spectrum at some bins !!")
    k_av_corr[k_av_corr > 2] = 10.0
    k_av_corr[k_av_corr < 0.1] = 0.1

Pk_flat_camb = None
del Pk_flat_camb
###

######################################################################
#R    FKP of the data to get the P_data(k) -- using MR and MRk
#R
#R  N.B.: In this code, we do this n_maps times -- once for each map, independently.
#R  Also, we make 4 estimates, for 4 "bandpower ranges" between k_min and k_max .
######################################################################



# These are the theory values for the total effective power spectrum
# updated nbar_bar to compute mean on cells with n > small=1.e-9
nbarbar = np.zeros(ntracers)
for nt in range(ntracers):
    nbarbar[nt] = np.mean(n_bar_matrix_fid[nt][n_bar_matrix_fid[nt] > small])/(cell_size)**3

ntot = np.sum(nbarbar*effbias**2)


#############################################################################
# BEGIN ESTIMATION
print ('Starting power spectra estimation')


# Initialize outputs

method = parameters_code['method']

if method == 'both':

    # Multitracer method: monopole, quadrupole, th. covariance(FKP-like)
    # Original (convolved) spectra:
    P0_data = np.zeros((n_maps,ntracers,num_binsk))
    P2_data = np.zeros((n_maps,ntracers,num_binsk))
    P4_data = np.zeros((n_maps,ntracers,num_binsk))

    # Traditional (FKP) method
    P0_fkp = np.zeros((n_maps,ntracers,num_binsk))
    P2_fkp = np.zeros((n_maps,ntracers,num_binsk))
    P4_fkp = np.zeros((n_maps,ntracers,num_binsk))

    # Cross spectra
    Cross0 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
    Cross2 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
    Cross4 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))

    # Covariance
    ThCov_fkp = np.zeros((n_maps,ntracers,num_binsk))


    # Range where we estimate some parameters
    myk_min = np.argsort(np.abs(kph-0.1))[0]
    myk_max = np.argsort(np.abs(kph-0.2))[0]
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

    fkp_mult = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,effbias_mt,cell_size,n_x_box,n_y_box,n_z_box,n_x_orig,n_y_orig,n_z_orig,MRk,powercentral,mas_power)

    ##
    # UPDATED THIS TO NEW FKP CLASS WITH AUTO- AND CROSS-SPECTRA
    print( "Initializing traditional (FKP) estimation toolbox...")
    fkp_many = fkp.fkp_init(num_binsk, n_bar_matrix_fid, effbias, cell_size, n_x_box, n_y_box, n_z_box, n_x_orig, n_y_orig, n_z_orig, MRk, powercentral,mas_power)

    '''
    Because of the very sensitive nature of shot noise subtraction 
    in the Jing de-aliasing, it may be better to normalize the counts of the 
    selection function to each map, and not to the mean of the maps.
    ''' 
    normsel = np.zeros((n_maps,ntracers))

    print ("... done. Starting computations for each map (box) now.")
    print()

    #################################


    est_bias_fkp = np.zeros(ntracers)
    est_bias_mt = np.zeros(ntracers)

    for nm in range(n_maps):
        time_start=time()
        print ('Loading simulated box #', nm)
        h5map = h5py.File(mapnames_sims[nm],'r')
        maps = np.asarray(h5map.get(list(h5map.keys())[0]))
        if not maps.shape == (ntracers,n_x,n_y,n_z):
            print()
            print ('Unexpected shape of simulated maps! Found:', maps.shape)
            print ('Check inputs and sims.  Aborting now...')
            print()
            sys.exit(-1)
        h5map.close()

        print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))

        ## !! NEW !! Additional mask from low-cell-count threshold
        if use_cell_low_count_thresh:
            maps = thresh_mask*maps
            #print ("Total number of objects AFTER additional threshold mask:", np.sum(maps,axis=(1,2,3)))
        else:
            pass

        if use_mask:
            maps = maps * mask


        # Apply padding, if it exists
        if use_padding:
            n_box = np.zeros((ntracers,n_x_box,n_y_box,n_z_box))
            n_box[:,padding_length[0]:-padding_length[0],padding_length[1]:-padding_length[1],padding_length[2]:-padding_length[2]] = maps
            maps = np.copy(n_box)
            n_box = None
            del n_box
        else:
            pass
        
        ##################################################
        # ATTENTION: The FKP estimator computes P_m(k), effectively dividing by the input bias factor.
        # Here the effective bias factor is bias*(growth function)*(amplitude of the monopole)
        # Notice that this means that the output of the FKP quadrupole
        # is P^2_FKP == (amplitude quadrupole)/(amplitude of the monopole)*P_m(k)
        ##################################################

        # Notice that we use "effective bias" (i.e., some estimate of the monopole) here;
        print ('  Estimating FKP power spectra...')
        # Use sum instead of mean to take care of empty cells
        normsel[nm] = np.sum(n_bar_matrix_fid,axis=(1,2,3))/np.sum(maps,axis=(1,2,3))
        # Updated definition of normsel to avoid raising a NaN when there are zero tracers
        normsel[nm,np.isinf(normsel[nm])]=1.0
        normsel[nm,np.isnan(normsel[nm])]=1.0
    
        if ( (normsel[nm].any() > 2.0) | (normsel[nm].any() < 0.5) ):
            print("Attention! Your selection function and simulation have very different numbers of objects:")
            print("Selecion function/map for all tracers:",np.around(normsel[nm],3))
            normsel[normsel > 2.0] = 2.0
            normsel[normsel < 0.5] = 0.5
            print(" Normalized selection function/map at:",np.around(normsel[nm],3))
        if nm==0:
            FKPmany = fkp_many.fkp((normsel[nm]*maps.T).T)
            P0_fkp[nm] = fkp_many.P_ret
            P2_fkp[nm] = fkp_many.P2_ret
            P4_fkp[nm] = fkp_many.P4_ret
            Cross0[nm] = fkp_many.cross_spec
            Cross2[nm] = fkp_many.cross_spec2
            Cross4[nm] = fkp_many.cross_spec4
            ThCov_fkp[nm] = (fkp_many.sigma)**2
        else:
            FKPmany = fkp_many.fkp((normsel[nm]*maps.T).T)
            P0_fkp[nm] = fkp_many.P_ret
            P2_fkp[nm] = fkp_many.P2_ret
            P4_fkp[nm] = fkp_many.P4_ret
            Cross0[nm] = fkp_many.cross_spec
            Cross2[nm] = fkp_many.cross_spec2
            Cross4[nm] = fkp_many.cross_spec4
            ThCov_fkp[nm] = (fkp_many.sigma)**2
        
        #################################
        # Now, the multi-tracer method
        print ('  Now estimating multi-tracer spectra...')
        if nm==0:
            FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
            P0_data[nm] = fkp_mult.P0_mu_ret
            P2_data[nm] = fkp_mult.P2_mu_ret
            P4_data[nm] = fkp_mult.P4_mu_ret
        else:
            FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
            P0_data[nm] = fkp_mult.P0_mu_ret
            P2_data[nm] = fkp_mult.P2_mu_ret
            P4_data[nm] = fkp_mult.P4_mu_ret

        #CORRECT FOR BIN AVERAGING
        P0_data[nm] = P0_data[nm]/k_av_corr
        P2_data[nm] = P2_data[nm]/k_av_corr
        P4_data[nm] = P4_data[nm]/k_av_corr
        P0_fkp[nm] = P0_fkp[nm]/k_av_corr
        P2_fkp[nm] = P2_fkp[nm]/k_av_corr
        P4_fkp[nm] = P4_fkp[nm]/k_av_corr
        Cross0[nm] = Cross0[nm]/k_av_corr
        Cross2[nm] = Cross2[nm]/k_av_corr
        Cross4[nm] = Cross4[nm]/k_av_corr

        if nm==0:
            # If data bias is different from mocks
            est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
            est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
            print ("  Effective biases of the simulated maps:")
            print ("   Fiducial=", ["%.3f"%b for b in effbias])
            print ("        FKP=", ["%.3f"%b for b in est_bias_fkp])
            print ("         MT=", ["%.3f"%b for b in est_bias_mt])
            dt = time() - time_start
            print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
            print(".")
        else:
            est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
            est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
            print( "  Effective biases of these maps:")
            print( "   Fiducial=", ["%.3f"%b for b in effbias])
            print( "        FKP=", ["%.3f"%b for b in est_bias_fkp])
            print ("         MT=", ["%.3f"%b for b in est_bias_mt])
            dt = time() - time_start
            print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
            print(".")
    
    #Update nbarbar to reflect actual
    nbarbar = nbarbar/np.mean(normsel,axis=0)

    # Correct missing factor of 2 in definition
    Theor_Cov_FKP = 2.0*np.mean(ThCov_fkp,axis=0)

    #del maps
    #maps = None

    ################################################################################
    ################################################################################


    time_end=time()
    print ('Total time cost for estimation of spectra: ', time_end - time_start)

    ################################################################################
    ################################################################################

    tempor=time()

    ################################################################################
    ################################################################################




    print ('Applying mass assignement window function corrections...')

    ################################################################################
    #############################################################################
    # Mass assignement correction

    # For k smaller than the smallest k_phys, we use the Planck power spectrum.
    # For k larger than the highest k_phys value, we extrapolate using the power law at k_N/2 -- or the highest value of k_phys in the range, whichever is smaller
    kph_min = kph[0]
    kN = np.pi/cell_size  # Nyquist frequency

    # Compute the mean power spectra -- I will use the MTOE for the iteration

    P0_mean = np.mean(P0_data,axis=0)
    P0_fkp_mean = np.mean(P0_fkp,axis=0)
    Cross0_mean = np.mean(Cross0,axis=0)

    # Cross0 and Cross2 are outputs of the FKP code, so they come out without the bias.
    # We can easily put back the bias by multiplying:
    # CrossX = cross_effbias**2 * CrossX
    cross_effbias = np.zeros(ntracers*(ntracers-1)//2)
    index=0
    for nt in range(ntracers):
        for ntp in range(nt+1,ntracers):
            cross_effbias[index] = np.sqrt(effbias[nt]*effbias[ntp])
            index += 1

    # FKP and Cross measurements need to have the bias returned in their definitions
    P0_fkp = np.transpose((effbias**2*np.transpose(P0_fkp,axes=(0,2,1))),axes=(0,2,1))
    P2_fkp = np.transpose((effbias**2*np.transpose(P2_fkp,axes=(0,2,1))),axes=(0,2,1))
    P4_fkp = np.transpose((effbias**2*np.transpose(P4_fkp,axes=(0,2,1))),axes=(0,2,1))

    C0_fkp = np.transpose((cross_effbias**2*np.transpose(Cross0,axes=(0,2,1))),axes=(0,2,1))
    C2_fkp = np.transpose((cross_effbias**2*np.transpose(Cross2,axes=(0,2,1))),axes=(0,2,1))
    C4_fkp = np.transpose((cross_effbias**2*np.transpose(Cross4,axes=(0,2,1))),axes=(0,2,1))

    # Means
    P0_mean = np.mean(P0_data,axis=0)
    P2_mean = np.mean(P2_data,axis=0)
    P4_mean = np.mean(P4_data,axis=0)
    P0_fkp_mean = np.mean(P0_fkp,axis=0)
    P2_fkp_mean = np.mean(P2_fkp,axis=0)
    P4_fkp_mean = np.mean(P4_fkp,axis=0)
    Cross0_mean = np.mean(C0_fkp,axis=0)
    Cross2_mean = np.mean(C2_fkp,axis=0)
    Cross4_mean = np.mean(C4_fkp,axis=0)

elif method == 'FKP':

    # Traditional (FKP) method
    P0_fkp = np.zeros((n_maps,ntracers,num_binsk))
    P2_fkp = np.zeros((n_maps,ntracers,num_binsk))
    P4_fkp = np.zeros((n_maps,ntracers,num_binsk))

    # Cross spectra
    Cross0 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
    Cross2 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
    Cross4 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))

    # Covariance
    ThCov_fkp = np.zeros((n_maps,ntracers,num_binsk))


    # Range where we estimate some parameters
    myk_min = np.argsort(np.abs(kph-0.1))[0]
    myk_max = np.argsort(np.abs(kph-0.2))[0]
    myran = np.arange(myk_min,myk_max)

    ##
    # UPDATED THIS TO NEW FKP CLASS WITH AUTO- AND CROSS-SPECTRA
    print( "Initializing traditional (FKP) estimation toolbox...")
    fkp_many = fkp.fkp_init(num_binsk, n_bar_matrix_fid, effbias, cell_size, n_x_box, n_y_box, n_z_box, n_x_orig, n_y_orig, n_z_orig, MRk, powercentral,mas_power)

    '''
    Because of the very sensitive nature of shot noise subtraction 
    in the Jing de-aliasing, it may be better to normalize the counts of the 
    selection function to each map, and not to the mean of the maps.
    ''' 
    normsel = np.zeros((n_maps,ntracers))

    print ("... done. Starting computations for each map (box) now.")
    print()

    #################################

    est_bias_fkp = np.zeros(ntracers)

    for nm in range(n_maps):
        time_start=time()
        print ('Loading simulated box #', nm)
        h5map = h5py.File(mapnames_sims[nm],'r')
        maps = np.asarray(h5map.get(list(h5map.keys())[0]))
        if not maps.shape == (ntracers,n_x,n_y,n_z):
            print()
            print ('Unexpected shape of simulated maps! Found:', maps.shape)
            print ('Check inputs and sims.  Aborting now...')
            print()
            sys.exit(-1)
        h5map.close()

        print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))

        ## !! NEW !! Additional mask from low-cell-count threshold
        if use_cell_low_count_thresh:
            maps = thresh_mask*maps
            #print ("Total number of objects AFTER additional threshold mask:", np.sum(maps,axis=(1,2,3)))
        else:
            pass

        if use_mask:
            maps = maps * mask


        # Apply padding, if it exists
        if use_padding:
            n_box = np.zeros((ntracers,n_x_box,n_y_box,n_z_box))
            n_box[:,padding_length[0]:-padding_length[0],padding_length[1]:-padding_length[1],padding_length[2]:-padding_length[2]] = maps
            maps = np.copy(n_box)
            n_box = None
            del n_box
        else:
            pass

        ##################################################
        # ATTENTION: The FKP estimator computes P_m(k), effectively dividing by the input bias factor.
        # Here the effective bias factor is bias*(growth function)*(amplitude of the monopole)
        # Notice that this means that the output of the FKP quadrupole
        # is P^2_FKP == (amplitude quadrupole)/(amplitude of the monopole)*P_m(k)
        ##################################################

        # Notice that we use "effective bias" (i.e., some estimate of the monopole) here;
        print ('  Estimating FKP power spectra...')
        # Use sum instead of mean to take care of empty cells
        normsel[nm] = np.sum(n_bar_matrix_fid,axis=(1,2,3))/np.sum(maps,axis=(1,2,3))
        # Updated definition of normsel to avoid raising a NaN when there are zero tracers
        normsel[nm,np.isinf(normsel[nm])]=1.0
        normsel[nm,np.isnan(normsel[nm])]=1.0
    
        if ( (normsel[nm].any() > 2.0) | (normsel[nm].any() < 0.5) ):
            print("Attention! Your selection function and simulation have very different numbers of objects:")
            print("Selecion function/map for all tracers:",np.around(normsel[nm],3))
            normsel[normsel > 2.0] = 2.0
            normsel[normsel < 0.5] = 0.5
            print(" Normalized selection function/map at:",np.around(normsel[nm],3))
        if nm==0:
            FKPmany = fkp_many.fkp((normsel[nm]*maps.T).T)
            P0_fkp[nm] = fkp_many.P_ret
            P2_fkp[nm] = fkp_many.P2_ret
            P4_fkp[nm] = fkp_many.P4_ret
            Cross0[nm] = fkp_many.cross_spec
            Cross2[nm] = fkp_many.cross_spec2
            Cross4[nm] = fkp_many.cross_spec4
            ThCov_fkp[nm] = (fkp_many.sigma)**2
        else:
            FKPmany = fkp_many.fkp((normsel[nm]*maps.T).T)
            P0_fkp[nm] = fkp_many.P_ret
            P2_fkp[nm] = fkp_many.P2_ret
            P4_fkp[nm] = fkp_many.P4_ret
            Cross0[nm] = fkp_many.cross_spec
            Cross2[nm] = fkp_many.cross_spec2
            Cross4[nm] = fkp_many.cross_spec4
            ThCov_fkp[nm] = (fkp_many.sigma)**2

        #CORRECT FOR BIN AVERAGING
        P0_fkp[nm] = P0_fkp[nm]/k_av_corr
        P2_fkp[nm] = P2_fkp[nm]/k_av_corr
        P4_fkp[nm] = P4_fkp[nm]/k_av_corr
        Cross0[nm] = Cross0[nm]/k_av_corr
        Cross2[nm] = Cross2[nm]/k_av_corr
        Cross4[nm] = Cross4[nm]/k_av_corr

        if nm==0:
            # If data bias is different from mocks
            est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
            print ("  Effective biases of the simulated maps:")
            print ("   Fiducial=", ["%.3f"%b for b in effbias])
            print ("        FKP=", ["%.3f"%b for b in est_bias_fkp])
            dt = time() - time_start
            print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
            print(".")
        else:
            est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
            print( "  Effective biases of these maps:")
            print( "   Fiducial=", ["%.3f"%b for b in effbias])
            print( "        FKP=", ["%.3f"%b for b in est_bias_fkp])
            dt = time() - time_start
            print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
            print(".")
    
    #Update nbarbar to reflect actual
    nbarbar = nbarbar/np.mean(normsel,axis=0)

    # Correct missing factor of 2 in definition
    Theor_Cov_FKP = 2.0*np.mean(ThCov_fkp,axis=0)

    #del maps
    #maps = None

    ################################################################################
    ################################################################################


    time_end=time()
    print ('Total time cost for estimation of spectra: ', time_end - time_start)

    ################################################################################
    ################################################################################

    tempor=time()

    ################################################################################
    ################################################################################




    print ('Applying mass assignement window function corrections...')

    ################################################################################
    #############################################################################
    # Mass assignement correction

    # For k smaller than the smallest k_phys, we use the Planck power spectrum.
    # For k larger than the highest k_phys value, we extrapolate using the power law at k_N/2 -- or the highest value of k_phys in the range, whichever is smaller
    kph_min = kph[0]
    kN = np.pi/cell_size  # Nyquist frequency

    # Compute the mean power spectra -- I will use the MTOE for the iteration

    P0_fkp_mean = np.mean(P0_fkp,axis=0)
    Cross0_mean = np.mean(Cross0,axis=0)

    # Cross0 and Cross2 are outputs of the FKP code, so they come out without the bias.
    # We can easily put back the bias by multiplying:
    # CrossX = cross_effbias**2 * CrossX
    cross_effbias = np.zeros(ntracers*(ntracers-1)//2)
    index=0
    for nt in range(ntracers):
        for ntp in range(nt+1,ntracers):
            cross_effbias[index] = np.sqrt(effbias[nt]*effbias[ntp])
            index += 1

    # FKP and Cross measurements need to have the bias returned in their definitions
    P0_fkp = np.transpose((effbias**2*np.transpose(P0_fkp,axes=(0,2,1))),axes=(0,2,1))
    P2_fkp = np.transpose((effbias**2*np.transpose(P2_fkp,axes=(0,2,1))),axes=(0,2,1))
    P4_fkp = np.transpose((effbias**2*np.transpose(P4_fkp,axes=(0,2,1))),axes=(0,2,1))

    C0_fkp = np.transpose((cross_effbias**2*np.transpose(Cross0,axes=(0,2,1))),axes=(0,2,1))
    C2_fkp = np.transpose((cross_effbias**2*np.transpose(Cross2,axes=(0,2,1))),axes=(0,2,1))
    C4_fkp = np.transpose((cross_effbias**2*np.transpose(Cross4,axes=(0,2,1))),axes=(0,2,1))

    # Means
    P0_fkp_mean = np.mean(P0_fkp,axis=0)
    P2_fkp_mean = np.mean(P2_fkp,axis=0)
    P4_fkp_mean = np.mean(P4_fkp,axis=0)
    Cross0_mean = np.mean(C0_fkp,axis=0)
    Cross2_mean = np.mean(C2_fkp,axis=0)
    Cross4_mean = np.mean(C4_fkp,axis=0)

elif method == 'MT':

    # Multitracer method: monopole, quadrupole, th. covariance(FKP-like)
    # Original (convolved) spectra:
    P0_data = np.zeros((n_maps,ntracers,num_binsk))
    P2_data = np.zeros((n_maps,ntracers,num_binsk))
    P4_data = np.zeros((n_maps,ntracers,num_binsk))

    # Range where we estimate some parameters
    myk_min = np.argsort(np.abs(kph-0.1))[0]
    myk_max = np.argsort(np.abs(kph-0.2))[0]
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

    fkp_mult = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,effbias_mt,cell_size,n_x_box,n_y_box,n_z_box,n_x_orig,n_y_orig,n_z_orig,MRk,powercentral,mas_power)

    '''
    Because of the very sensitive nature of shot noise subtraction 
    in the Jing de-aliasing, it may be better to normalize the counts of the 
    selection function to each map, and not to the mean of the maps.
    ''' 
    normsel = np.zeros((n_maps,ntracers))

    print ("... done. Starting computations for each map (box) now.")
    print()

    #################################

    est_bias_mt = np.zeros(ntracers)

    for nm in range(n_maps):
        time_start=time()
        print ('Loading simulated box #', nm)
        h5map = h5py.File(mapnames_sims[nm],'r')
        maps = np.asarray(h5map.get(list(h5map.keys())[0]))
        if not maps.shape == (ntracers,n_x,n_y,n_z):
            print()
            print ('Unexpected shape of simulated maps! Found:', maps.shape)
            print ('Check inputs and sims.  Aborting now...')
            print()
            sys.exit(-1)
        h5map.close()

        print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))

        ## !! NEW !! Additional mask from low-cell-count threshold
        if use_cell_low_count_thresh:
            maps = thresh_mask*maps
            #print ("Total number of objects AFTER additional threshold mask:", np.sum(maps,axis=(1,2,3)))
        else:
            pass

        if use_mask:
            maps = maps * mask


        # Apply padding, if it exists
        if use_padding:
            n_box = np.zeros((ntracers,n_x_box,n_y_box,n_z_box))
            n_box[:,padding_length[0]:-padding_length[0],padding_length[1]:-padding_length[1],padding_length[2]:-padding_length[2]] = maps
            maps = np.copy(n_box)
            n_box = None
            del n_box
        else:
            pass

        # Use sum instead of mean to take care of empty cells
        normsel[nm] = np.sum(n_bar_matrix_fid,axis=(1,2,3))/np.sum(maps,axis=(1,2,3))
        # Updated definition of normsel to avoid raising a NaN when there are zero tracers
        normsel[nm,np.isinf(normsel[nm])]=1.0
        normsel[nm,np.isnan(normsel[nm])]=1.0
        
        #################################
        # Now, the multi-tracer method
        print ('  Now estimating multi-tracer spectra...')
        if nm==0:
            FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
            P0_data[nm] = fkp_mult.P0_mu_ret
            P2_data[nm] = fkp_mult.P2_mu_ret
            P4_data[nm] = fkp_mult.P4_mu_ret
        else:
            FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
            P0_data[nm] = fkp_mult.P0_mu_ret
            P2_data[nm] = fkp_mult.P2_mu_ret
            P4_data[nm] = fkp_mult.P4_mu_ret

        #CORRECT FOR BIN AVERAGING
        P0_data[nm] = P0_data[nm]/k_av_corr
        P2_data[nm] = P2_data[nm]/k_av_corr
        P4_data[nm] = P4_data[nm]/k_av_corr

        if nm==0:
            # If data bias is different from mocks
            est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
            print ("  Effective biases of the simulated maps:")
            print ("   Fiducial=", ["%.3f"%b for b in effbias])
            print ("         MT=", ["%.3f"%b for b in est_bias_mt])
            dt = time() - time_start
            print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
            print(".")
        else:
            est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
            print( "  Effective biases of these maps:")
            print( "   Fiducial=", ["%.3f"%b for b in effbias])
            print ("         MT=", ["%.3f"%b for b in est_bias_mt])
            dt = time() - time_start
            print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
            print(".")
    
    #Update nbarbar to reflect actual
    nbarbar = nbarbar/np.mean(normsel,axis=0)

    #del maps
    #maps = None

    ################################################################################
    ################################################################################


    time_end=time()
    print ('Total time cost for estimation of spectra: ', time_end - time_start)

    ################################################################################
    ################################################################################

    tempor=time()

    ################################################################################
    ################################################################################

    print ('Applying mass assignement window function corrections...')

    ################################################################################
    #############################################################################
    # Mass assignement correction

    # For k smaller than the smallest k_phys, we use the Planck power spectrum.
    # For k larger than the highest k_phys value, we extrapolate using the power law at k_N/2 -- or the highest value of k_phys in the range, whichever is smaller
    kph_min = kph[0]
    kN = np.pi/cell_size  # Nyquist frequency

    # Compute the mean power spectra -- I will use the MTOE for the iteration

    P0_mean = np.mean(P0_data,axis=0)

    # Means
    P0_mean = np.mean(P0_data,axis=0)
    P2_mean = np.mean(P2_data,axis=0)
    P4_mean = np.mean(P4_data,axis=0)

    
################################################################################
################################################################################
################################################################################
################################################################################
#
#   SAVE these spectra
#
################################################################################
################################################################################
################################################################################
################################################################################


if method == 'both':
    P0_save=np.reshape(P0_data,(n_maps,ntracers*pow_bins))
    P0_fkp_save=np.reshape(P0_fkp,(n_maps,ntracers*pow_bins))

    P2_save=np.reshape(P2_data,(n_maps,ntracers*pow_bins))
    P2_fkp_save=np.reshape(P2_fkp,(n_maps,ntracers*pow_bins))

    P4_save=np.reshape(P4_data,(n_maps,ntracers*pow_bins))
    P4_fkp_save=np.reshape(P4_fkp,(n_maps,ntracers*pow_bins))

    C0_fkp_save=np.reshape(C0_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
    C2_fkp_save=np.reshape(C2_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
    C4_fkp_save=np.reshape(C4_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))

    # Export data
    np.savetxt(dir_specs + '/' + handle_estimates + '_vec_k.dat',kph,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_P0_MTOE.dat',P0_save,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P0_FKP.dat',P0_fkp_save,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_P2_MTOE.dat',P2_save,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P2_FKP.dat',P2_fkp_save,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P4_FKP.dat',P4_fkp_save,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_C0_FKP.dat',C0_fkp_save,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_C2_FKP.dat',C2_fkp_save,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_C4_FKP.dat',C4_fkp_save,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_nbar_mean.dat',nbarbar,fmt="%2.6f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_bias.dat',gal_bias,fmt="%2.3f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_effbias.dat',effbias,fmt="%2.3f")


    np.savetxt(dir_specs + '/' + handle_estimates + '_P0_MTOE_mean.dat',P0_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P2_MTOE_mean.dat',P2_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P4_MTOE_mean.dat',P4_mean,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_P0_FKP_mean.dat',P0_fkp_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P2_FKP_mean.dat',P2_fkp_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P4_FKP_mean.dat',P4_fkp_mean,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_C0_FKP_mean.dat',Cross0_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_C2_FKP_mean.dat',Cross2_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_C4_FKP_mean.dat',Cross4_mean,fmt="%6.4f")

    #UPDATED SOME PLOT DEFINITIONS
    kN = np.pi/cell_size  # Nyquist frequency
    ikN9 = np.argsort(np.abs(kph-kN))[0]

    xlim = [0.9*kph[0],1.1*kph[ikN9]]
    ylim = [np.min(effbias**2)*np.min(powtrue)/100.,np.max(effbias**2)*np.max(powtrue)*5]

    print(".")
    print("Now just a couple of plots to /figs...")

    index=0
    pl.loglog(kph,powtrue,'k',linewidth=0.3)
    for nt in range(ntracers):
        pl.loglog(kph,P0_fkp_mean[nt],'b',linewidth=0.6)
        pl.loglog(kph,-P0_fkp_mean[nt],'b--',linewidth=0.6)
        pl.loglog(kph,P2_fkp_mean[nt],'g',linewidth=0.5)
        pl.loglog(kph,-P2_fkp_mean[nt],'g--',linewidth=0.5)
        pl.loglog(kph,P4_fkp_mean[nt],'c',linewidth=0.3)
        pl.loglog(kph,-P4_fkp_mean[nt],'c--',linewidth=0.3)
        for ntp in range(nt+1,ntracers):
            pl.loglog(kph,Cross0_mean[index],'b',linewidth=0.2)
            pl.loglog(kph,Cross2_mean[index],'g',linewidth=0.2)
            pl.loglog(kph,Cross4_mean[index],'c',linewidth=0.2)
            index += 1

    pl.xlim(xlim)
    pl.ylim(ylim)
    pl.savefig(dir_figs + "/Mean_spec_fkp.png",dpi=200)
    pl.close('all')

    index=0
    pl.loglog(kph,powtrue,'k',linewidth=0.3)
    for nt in range(ntracers):
        pl.loglog(kph,P0_mean[nt],'b',linewidth=0.6)
        pl.loglog(kph,-P0_mean[nt],'b--',linewidth=0.6)
        pl.loglog(kph,P2_mean[nt],'g',linewidth=0.5)
        pl.loglog(kph,-P2_mean[nt],'g--',linewidth=0.5)
        pl.loglog(kph,P4_mean[nt],'c',linewidth=0.3)
        pl.loglog(kph,-P4_mean[nt],'c--',linewidth=0.3)
        for ntp in range(nt+1,ntracers):
            pl.loglog(kph,Cross0_mean[index],'b',linewidth=0.2)
            pl.loglog(kph,Cross2_mean[index],'g',linewidth=0.2)
            pl.loglog(kph,Cross4_mean[index],'c',linewidth=0.2)
            index += 1
    pl.xlim(xlim)
    pl.ylim(ylim)
    pl.savefig(dir_figs + "/Mean_spec_mt.png",dpi=200)
    pl.close('all')

    print(".")
    print(".")
    print()
    print("Done!")
    print()

    sys.exit(-1)

elif method == 'FKP':
    P0_fkp_save=np.reshape(P0_fkp,(n_maps,ntracers*pow_bins))

    P2_fkp_save=np.reshape(P2_fkp,(n_maps,ntracers*pow_bins))

    P4_fkp_save=np.reshape(P4_fkp,(n_maps,ntracers*pow_bins))

    C0_fkp_save=np.reshape(C0_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
    C2_fkp_save=np.reshape(C2_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
    C4_fkp_save=np.reshape(C4_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))

    # Export data
    np.savetxt(dir_specs + '/' + handle_estimates + '_vec_k.dat',kph,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_P0_FKP.dat',P0_fkp_save,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_P2_FKP.dat',P2_fkp_save,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P4_FKP.dat',P4_fkp_save,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_C0_FKP.dat',C0_fkp_save,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_C2_FKP.dat',C2_fkp_save,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_C4_FKP.dat',C4_fkp_save,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_nbar_mean.dat',nbarbar,fmt="%2.6f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_bias.dat',gal_bias,fmt="%2.3f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_effbias.dat',effbias,fmt="%2.3f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_P0_FKP_mean.dat',P0_fkp_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P2_FKP_mean.dat',P2_fkp_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P4_FKP_mean.dat',P4_fkp_mean,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_C0_FKP_mean.dat',Cross0_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_C2_FKP_mean.dat',Cross2_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_C4_FKP_mean.dat',Cross4_mean,fmt="%6.4f")

    #UPDATED SOME PLOT DEFINITIONS
    kN = np.pi/cell_size  # Nyquist frequency
    ikN9 = np.argsort(np.abs(kph-kN))[0]

    xlim = [0.9*kph[0],1.1*kph[ikN9]]
    ylim = [np.min(effbias**2)*np.min(powtrue)/100.,np.max(effbias**2)*np.max(powtrue)*5]

    print(".")
    print("Now just a couple of plots to /figs...")

    index=0
    pl.loglog(kph,powtrue,'k',linewidth=0.3)
    for nt in range(ntracers):
        pl.loglog(kph,P0_fkp_mean[nt],'b',linewidth=0.6)
        pl.loglog(kph,-P0_fkp_mean[nt],'b--',linewidth=0.6)
        pl.loglog(kph,P2_fkp_mean[nt],'g',linewidth=0.5)
        pl.loglog(kph,-P2_fkp_mean[nt],'g--',linewidth=0.5)
        pl.loglog(kph,P4_fkp_mean[nt],'c',linewidth=0.3)
        pl.loglog(kph,-P4_fkp_mean[nt],'c--',linewidth=0.3)
        for ntp in range(nt+1,ntracers):
            pl.loglog(kph,Cross0_mean[index],'b',linewidth=0.2)
            pl.loglog(kph,Cross2_mean[index],'g',linewidth=0.2)
            pl.loglog(kph,Cross4_mean[index],'c',linewidth=0.2)
            index += 1

    pl.xlim(xlim)
    pl.ylim(ylim)
    pl.savefig(dir_figs + "/Mean_spec_fkp.png",dpi=200)
    pl.close('all')

    print(".")
    print(".")
    print()
    print("Done!")
    print()

    sys.exit(-1)

elif method == 'MT':
    P0_save=np.reshape(P0_data,(n_maps,ntracers*pow_bins))

    P2_save=np.reshape(P2_data,(n_maps,ntracers*pow_bins))

    P4_save=np.reshape(P4_data,(n_maps,ntracers*pow_bins))

    # Export data
    np.savetxt(dir_specs + '/' + handle_estimates + '_vec_k.dat',kph,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_P0_MTOE.dat',P0_save,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_P2_MTOE.dat',P2_save,fmt="%6.4f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_nbar_mean.dat',nbarbar,fmt="%2.6f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_bias.dat',gal_bias,fmt="%2.3f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_effbias.dat',effbias,fmt="%2.3f")

    np.savetxt(dir_specs + '/' + handle_estimates + '_P0_MTOE_mean.dat',P0_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P2_MTOE_mean.dat',P2_mean,fmt="%6.4f")
    np.savetxt(dir_specs + '/' + handle_estimates + '_P4_MTOE_mean.dat',P4_mean,fmt="%6.4f")

    #UPDATED SOME PLOT DEFINITIONS
    kN = np.pi/cell_size  # Nyquist frequency
    ikN9 = np.argsort(np.abs(kph-kN))[0]

    xlim = [0.9*kph[0],1.1*kph[ikN9]]
    ylim = [np.min(effbias**2)*np.min(powtrue)/100.,np.max(effbias**2)*np.max(powtrue)*5]

    print(".")
    print("Now just a couple of plots to /figs...")

    index=0
    pl.loglog(kph,powtrue,'k',linewidth=0.3)
    for nt in range(ntracers):
        pl.loglog(kph,P0_mean[nt],'b',linewidth=0.6)
        pl.loglog(kph,-P0_mean[nt],'b--',linewidth=0.6)
        pl.loglog(kph,P2_mean[nt],'g',linewidth=0.5)
        pl.loglog(kph,-P2_mean[nt],'g--',linewidth=0.5)
        pl.loglog(kph,P4_mean[nt],'c',linewidth=0.3)
        pl.loglog(kph,-P4_mean[nt],'c--',linewidth=0.3)
    pl.xlim(xlim)
    pl.ylim(ylim)
    pl.savefig(dir_figs + "/Mean_spec_mt.png",dpi=200)
    pl.close('all')

    print(".")
    print(".")
    print()
    print("Done!")
    print()

    sys.exit(-1)
