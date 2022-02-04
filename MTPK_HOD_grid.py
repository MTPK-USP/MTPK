#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
------------
This code implements an HOD 
Given a grid containing halos in cells,
it creates a grid containing tracers ("galaxies") in cells.
Can impose a selection function on the galaxies, if needed
------------
"""

#from __future__ import print_function
import numpy as np
import os, sys
import uuid
import h5py
import glob

# Matplotlib
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

from time import time , strftime
from scipy import interpolate
from scipy import special
from scipy.optimize import leastsq
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse import vstack

import grid3D as gr
import pk_multipoles_gauss as pkmg
import pk_crossmultipoles_gauss as pkmg_cross
from camb_spec import camb_spectrum
from cosmo_funcs import matgrow, H
from analytical_selection_function import *


#####################################################
#####################################################
#####################################################


# Specify where the original (halo) grids will be found
dir_grid_halos = "/maps/sims/Highz_z16/"
# Specify where the original (halo) spectra will be found
dir_specs_halos = '/spectra/Highz_z16/'


# Specify where the tracer grids will be saved:
dir_grid_tracer = "/maps/sims/Highz_z16_gals/"
# Specify where the model+theory will be saved
dir_specs_tracer = '/spectra/Highz_z16_gals/'

# If using a data selection function, specify it here
sel_fun_data_HOD = True
sel_fun_file_HOD = "sel_fun_highz_z1.6.hdf5"


#####################################################
#####################################################
#####################################################


# Add path to /inputs directory in order to load inputs
# Change as necessary according to your installation
this_dir = os.getcwd()

dir_grid_halos = this_dir + dir_grid_halos
dir_specs_halos = this_dir + dir_specs_halos
dir_grid_tracer = this_dir + dir_grid_tracer
dir_specs_tracer = this_dir + dir_specs_tracer

input_dir = this_dir + '/inputs/'
sys.path.append(input_dir)


small=1.e-8

# Load specs for this run
from MTPK import *

# Override selection function specs
sel_fun_data = sel_fun_data_HOD
sel_fun_file = sel_fun_file_HOD



###################
print()
print( 'This is the HOD module of the MTPK suite.')

print()
print('Handle/input of this run (inputs, fiducial spectra, biases, etc.): ', handle_estimates)
print()

# Load those properties
input_filename = glob.glob('inputs/*' + handle_estimates + '.py')
if (len(input_filename)==0) or (len(input_filename)>2) :
    print ('Input files not found -- or more than one with same handle in dir!')
    print ('Check on /inputs.  Aborting now...')
    sys.exit(-1)

exec ("from " + handle_estimates + " import *")

# Directory with data and simulated maps
print("Original (halo) grids will be read from:")
print("  > ", dir_grid_halos)

# Directory with saved maps
print("Tracers after applying HOD will be saved in:")
print("  > ",  dir_grid_tracer)
print("More spectra of these tracers will be saved in:")
print("  > " , dir_specs_tracer)

# This is where the data selection function is stored if needed
if (not sims_only) or (sel_fun_data):
    dir_data = this_dir + '/maps/data/' + handle_data

#
##########

if not os.path.exists(dir_grid_tracer):
    os.makedirs(dir_grid_tracer)

if not os.path.exists(dir_specs_tracer):
    os.makedirs(dir_specs_tracer)


#####################################################
try:
    hod = np.loadtxt(input_dir + hod_file)
    # Halo mass function
    mass_fun = mult_sel_fun*np.loadtxt(input_dir + mass_fun_file)
    halo_bias = np.loadtxt(input_dir + halo_bias_file)
    hod_delta = ((hod*mass_fun).T/np.dot(hod,mass_fun)).T
except:
    print ("Something's wrong... did not find HOD and/or mass function:")
    print(hod_file,mass_fun_file)
    print ("Check in the /inputs directory. Aborting now...")
    sys.exit(-1)

# nbar below is in cell units
halo_densities_cell = np.asarray(mass_fun)*(cell_size)**3

# Tracer densities (before selection function)
gal_densities = np.dot(hod,mass_fun)
gal_densities_cell = gal_densities*(cell_size)**3

gal_bias  = np.dot(hod,mass_fun*halo_bias)/gal_densities

print()
print("Bias of the tracers generated with this HOD:", np.around(gal_bias,3) )
answer = input('You may want to save this in a file for later use. Continue? y/n  ')
if answer!='y':
    print ('Aborting now...')
    sys.exit(-1)
print()

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
    # This tells us how the HOD acts on each cell 
    hod_3D = np.zeros((ngals,nhalos,n_x,n_y,n_z),dtype='float32')
    for ng in range(ngals):
        for nh in range(nhalos):
            hod_3D[ng,nh] = np.asarray(hod[ng,nh]/gal_densities_cell[ng]*n_bar_matrix_fid[ng],dtype='float32')
else:
    print ("Att.: using analytical selection function for galaxies (check parameters in input file).")
    nbar = mult_sel_fun*gal_densities_cell
    #if (len(bias) != ngals) or (len(nbar) != ngals) or (len(ncentral) != ngals) or (len(nsigma) != ngals) or (len(bias) != ngals) or (len(adip) != ngals):
    hod_3D = np.zeros((ngals,nhalos,n_x,n_y,n_z),dtype='float32')
    n_bar_matrix_fid = np.zeros((ngals,n_x,n_y,n_z),dtype='float32')
    for ng in range(ngals):
        # Here you can choose how to call the selection function, using the different dependencies
        # Exponential/Gaussian form:
        n_bar_matrix_fid[ng] = selection_func_Gaussian(grid.grid_r, nbar[ng],ncentral[ng],nsigma[ng])
        # Linear form:
        #n_bar_matrix_fid[ng] = selection_func_Linear(grid.RX, grid.RY, grid.RZ, nbar[ng],ax[ng],ay[ng],az[ng])
        # Use this to build galaxy hod, cell by cell, given the halo mass function, for each halo:
        for nh in range(nhalos):
            hod_3D[ng,nh] = mult_sel_fun*np.asarray(hod[ng,nh]/gal_densities_cell[ng] * n_bar_matrix_fid[ng] , dtype='float32')
            #n_bar_matrix_fid[ng] += hod_3D[ng,nh]*halo_densities_cell[nh]



########################## Some other cosmological quantities ######################################
Omegam = Omegac + Omegab
OmegaDE = 1. - Omegam - Omegak
h = H0/100.
clight = 299792.46
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

try:
    # Velocity dispersion. vdisp is defined on inputs with units of km/s
    vdisp = np.asarray(halos_vdisp) #km/s
    sigma_v = vdisp/H(100,Omegam,OmegaDE,-1,0.0,zcentral) #Mpc/h
    a_vdisp = vdisp/clight #Adimensional vdisp

    # Redshift errors. sigz_est is defined on inputs, and is adimensional
    sigz_est = np.asarray(halos_sigz_est)
    sigma_z = sigz_est*clight/H(100,Omegam,OmegaDE,-1,0.0,zcentral) # Mpc/h

    # Joint factor considering v dispersion and z error
    sig_tot = np.sqrt(sigma_z**2 + sigma_v**2) #Mpc/h
    a_sig_tot = np.sqrt(sigz_est**2 + a_vdisp**2) #Adimensional sig_tot
    # Dipole term (if it exists in original map)
    try:
        kdip_phys
    except:
        kdip_phys = 1./(cell_size*(n_z_orig + n_z/2.))
    else:
        print ('ATTENTION: pre-defined (on input) alpha-dipole k_dip [h/Mpc]=', '%1.4f'%kdip_phys)

    try:
        dip = np.asarray(halos_adip) * kdip_phys
    except:
        dip = 0.0

except:
    print("Did not find properties of halos in input file:")
    print("  > ", input_filename)
    print("Notice that you need to specify redshift errors and dipole terms for galaxies AND halos on the input file.")
    print("Aborting now...")
    sys.exit(-1)

###################################################################################################




#############Calling CAMB for calculations of the spectra#################
print('Computing CAMB spectrum...')

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

# These are the theoretical auto-monopoles and auto-quadrupoles for the power spectra of the halos
pk_mg = pkmg.pkmg(halo_bias,dip,matgrowcentral,k_camb,a_sig_tot,cH,zcentral)
monopoles = pk_mg.mono
quadrupoles = pk_mg.quad

# And these are the theoretical cross-monopoles and cross-quadrupoles for the halos
pk_mg_cross = pkmg_cross.pkmg_cross(halo_bias,dip,matgrowcentral,k_camb,a_sig_tot,cH,zcentral)
cross_monopoles = pk_mg_cross.monos
cross_quadrupoles = pk_mg_cross.quads





# Import correction factors for the lognormal halo simulations
# Use these to correct the spectra of the lognormal sims
# (Obs: halo_spec_corr are the corrections in REAL space, not redshift space)
print("Importing halo model spectra...")
try:
    Pk_camb_sim = np.loadtxt(dir_specs_halos + 'Pk_camb.dat')[:,1]
    halo_spec_corr = np.loadtxt(dir_specs_halos + 'spec_corrections.dat')
    halo_mono_model = np.loadtxt(dir_specs_halos + 'monopole_model.dat')
    halo_quad_model = np.loadtxt(dir_specs_halos + 'quadrupole_model.dat')
    halo_mono_theory = np.loadtxt(dir_specs_halos + 'monopole_theory.dat')
    halo_quad_theory = np.loadtxt(dir_specs_halos + 'quadrupole_theory.dat')
    k_corr = halo_spec_corr[:,0]
    nks = len(k_corr)
    if(nhalos>1):
        halo_crossmono_model = np.loadtxt(dir_specs_halos + 'cross_monopole_model.dat')
        halo_crossquad_model = np.loadtxt(dir_specs_halos + 'cross_quadrupole_model.dat')
        halo_crossmono_theory = np.loadtxt(dir_specs_halos + 'cross_monopole_theory.dat')
        halo_crossquad_theory = np.loadtxt(dir_specs_halos + 'cross_quadrupole_theory.dat')
        if len(halo_crossmono_model.shape) == 1:
            halo_crossmono_model = np.reshape(halo_crossmono_model, (nks, 1))
            halo_crossquad_model = np.reshape(halo_crossquad_model, (nks, 1))
            halo_crossmono_theory = np.reshape(halo_crossmono_theory, (nks, 1))
            halo_crossquad_theory = np.reshape(halo_crossquad_theory, (nks, 1))
except:
    print()
    print ("Did not find halo spectral corrections and theory spectra on directory:")
    print (dir_spec_halos)
    print ("[These are files that are created by the lognormal map-creating tool.]")
    print ("Will assume spectral corrections are all unity, and that the model/theory is given by ")
    print ("the auto- and cross-multipoles from CAMB [+ HaloFit] + linear bias + Kaiser RSD.")
    print()
    k_corr = k_camb
    Pk_camb_sim = Pk_camb
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
    if(nhalos>1):
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

# Reorganize auto- and cross-spectra into "all" spectra
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




####################
#
# Now create model/theory for the tracers after imposing the HOD

if(ngals>1):
    cross_mono_model = np.zeros((nks,ngals*(ngals-1)//2))
    cross_quad_model = np.zeros((nks,ngals*(ngals-1)//2))
    cross_mono_theory = np.zeros((nks,ngals*(ngals-1)//2))
    cross_quad_theory = np.zeros((nks,ngals*(ngals-1)//2))

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

mono_model = np.zeros((nks,ngals))
quad_model = np.zeros((nks,ngals))
mono_theory = np.zeros((nks,ngals))
quad_theory = np.zeros((nks,ngals))

for ng in range(ngals):
    mono_model[:,ng] = all_mono_model[:,ng,ng]
    quad_model[:,ng] = all_quad_model[:,ng,ng]
    mono_theory[:,ng] = all_mono_theory[:,ng,ng]
    quad_theory[:,ng] = all_quad_theory[:,ng,ng]



#####################
# Save models for tracers to files in dir_grid_tracer -- in the same format as was done for the halos
spec_corr = np.vstack((k_corr,spec_corr.T)).T

np.savetxt(dir_specs_tracer + 'spec_corrections.dat',spec_corr,fmt="%2.3f")
np.savetxt(dir_specs_tracer + 'monopole_model.dat',mono_model,fmt="%4.3f")
np.savetxt(dir_specs_tracer + 'quadrupole_model.dat',quad_model,fmt="%4.3f")
np.savetxt(dir_specs_tracer + 'monopole_theory.dat',mono_theory,fmt="%4.3f")
np.savetxt(dir_specs_tracer + 'quadrupole_theory.dat',quad_theory,fmt="%4.3f")

if(ngals>1):
    np.savetxt(dir_specs_tracer + 'cross_monopole_model.dat',cross_mono_model,fmt="%4.3f")
    np.savetxt(dir_specs_tracer + 'cross_quadrupole_model.dat',cross_quad_model,fmt="%4.3f")
    np.savetxt(dir_specs_tracer + 'cross_monopole_theory.dat',cross_mono_theory,fmt="%4.3f")
    np.savetxt(dir_specs_tracer + 'cross_quadrupole_theory.dat',cross_quad_theory,fmt="%4.3f")

camb_save = np.array([k_corr,Pk_camb_sim]).T
np.savetxt(dir_specs_tracer + 'Pk_camb.dat', camb_save, fmt="%4.3f")





####################
# Now start transforming halo maps into tracer maps using HOD.
#
# Look for files with maps

mapnames_sims = sorted(glob.glob(dir_grid_halos + '*.hdf5'))
if len(mapnames_sims)==0 :
    print()
    print ('Halo grid files not found! Check your directory', dir_grid_halos)
    print ('Aborting now...')
    print ()
    sys.exit(-1)
if len(mapnames_sims) != n_maps :
    print ('You are using', n_maps, ' mocks out of the existing', len(mapnames_sims))
    answer = input('Continue anyway? y/n  ')
    if answer!='y':
        print ('Aborting now...')
        sys.exit(-1)
print ('Will use the N =', n_maps, ' grids contained in directory', dir_grid_halos)

print ()
print ('Geometry: (nx,ny,nz) = (' +str(n_x)+','+str(n_y)+','+str(n_z)+'),  cell_size=' + str(cell_size) + ' h^-1 Mpc')



for nm in range(n_maps):
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
    print( "Total number of halos in this map:", np.sum(maps_halos,axis=(1,2,3)))
    print ('Creating tracer maps from halo maps...')
    maps = np.zeros((ngals,n_x,n_y,n_z))
    for ng in range(ngals):
        for nh in range(nhalos):
            maps[ng] += hod_3D[ng,nh]*maps_halos[nh]
        # Poisson sampling
        maps[ng] = np.random.poisson(maps[ng])
        print("   [total number of tracer", ng, "in this map:", np.sum(maps[ng]) , "]")

    print("Saving grid with tracers/galaxies...")

    if len(str(nm))==1:
        map_num = '00' + str(nm)
    elif len(str(nm))==2:
        map_num = '0' + str(nm)
    else:
        map_num = str(nm)
    hdf5_map_file = dir_grid_tracer + 'Grid_' + map_num + '.hdf5'
    print('Writing file ', hdf5_map_file)
    h5f = h5py.File(hdf5_map_file,'w')
    #h5f.create_dataset(hdf5_map_file, data=maps_out[nt], dtype='int64')
    h5f.create_dataset('sims', data=maps, dtype='int32',compression='gzip')
    h5f.close()
    print("Ok. Next...")
    print()

print("Done!")

sys.exit(-1)


