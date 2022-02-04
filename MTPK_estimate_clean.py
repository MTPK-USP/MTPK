#General packages
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

from pprint import pprint as pp

#Classes -- functions used by the MTPK suite
import fkp_multitracer as fkpmt
import fkp_class as fkp  # This is the new class, which computes auto- and cross-spectra
import pk_multipoles_gauss as pkmg
import pk_crossmultipoles_gauss as pkmg_cross
from camb_spec import camb_spectrum
from cosmo_funcs import matgrow, H
from analytical_selection_function import *
import grid3D as gr

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


###################################################
#Parameters of input.py file
###################################################

def MTPK_physical_params(**kwargs):
    default_params = { 'OMEGA_K': 0,
                       'W0': - 1.,
                       'W1': 0,
                       'OMEGA_B': 0.05,
                       'OMEGA_C': 0.262,
                       'H0': 67.556,
                       'N_SA': 0.96,
                       'A_S': 2e-9,
                       'Z_RE': 10.0,
                       'GAMMA': 0.5454,
                       'C_LIGHT': 299792.46,
                       'ZCENTRAL': 1.0,
                       'ESTIMATE_MAT_GROW_CENTRAL': True,
                       'MATGROWCENTRAL': 0.00001
    }
    for key, value in kwargs.items():
        if key not in default_params.keys():
            raise KeyError(f"You may not create new parameters. Available parameters are {list(default_params.keys())}. You passed '{key}' as key.")
        if type(default_params[key]) != type(value):
            raise TypeError(f"Expected {type(default_params[key])}, got {type(value)} in key '{key}'")
        default_params[key] = value
    default_params['OMEGA_M'] = default_params['OMEGA_B'] + default_params['OMEGA_C']
    default_params['OMEGA_DE'] = 1. - default_params['OMEGA_M'] - default_params['OMEGA_K']
    if default_params['ESTIMATE_MAT_GROW_CENTRAL'] == True:
        default_params['MATGROWCENTRAL'] = matgrow(default_params['OMEGA_M'], default_params['OMEGA_DE'],
                                                   default_params['W0'], default_params['W1'],
                                                   default_params['ZCENTRAL'], default_params['GAMMA'])

    default_params['H'] = default_params['H0']/100.
    return default_params

# #Example of change
# pp(MTPK_physical_params(H0 = 72, ESTIMATE_MAT_GROW_CENTRAL = False))
# pp(MTPK_physical_params(H0 = 'batata', ESTIMATE_MAT_GROW_CENTRAL = False))
# pp(MTPK_physical_params(ZCENT = 1.0))

def MTPK_code_options(**kwargs):
    default_params = { 'SIMS_ONLY': True,
                       'USE_WINDOW_FUNCTION': False,
                       'MASS_FUN_FILE': "inputs/ExSHalos_MF.dat",
                       'HALO_BIAS_FILE': "inputs/ExSHalos_bias.dat",
                       'NHALOS': 3,
                       'HALOS_IDS': ['h1', 'h2', 'h3'],
                       'N_MAPS': 3,
                       'CELL_SIZE': 1.0,
                       'N_X': 128,
                       'N_Y': 128,
                       'N_Z': 128,
                       'N_X_ORIG': -64.,
                       'N_Y_ORIG': -64.,
                       'N_Z_ORIG': 10000.,
                       'SEL_FUN_DATA': False,
                       'MF_TO_NBAR': np.loadtxt('inputs/ExSHalos_MF.dat'),
                       'CELL_LOW_COUNT_THRESH': 0.0,
                       'MULT_SEL_FUN': 1.0,
                       'SHIFT_SEL_FUN': 0.0,
                       'KMIN_BIAS': 0.05,
                       'KMAX_BIAS': 0.15,
                       'K_MIN_CAMB': 1.e-4,
                       'K_MAX_CAMB': 1.e+0,
                       'KPH_CENTRAL': 0.1,
                       'DKPH_BIN': 0.01,
                       'KMAX_PHYS': 0.6,
                       'KMIN_PHYS': 0.05,
                       'ZBINWIDTH': 0.1,
                       'WHICHSPEC': 1,
                       'JING_DEC_SIMS': True,
                       'POWER_JING_SIMS': 2.0,
                       'POWER_JING_DATA': 2.0,
                       'PLOT_ALL_COV': False,
                       'FUDGE': 0.0000001,
    }
    for key, value in kwargs.items():
        if key not in default_params.keys():
            raise KeyError(f"You may not create new parameters. Available parameters are {list(default_params.keys())}. You passed '{key}' as key.")
        if type(default_params[key]) != type(value):
            raise TypeError(f"Expected {type(default_params[key])}, got {type(value)} in key '{key}'")
        default_params[key] = value
    default_params['NTRACERS'] = default_params['NHALOS']
    default_params['TRACER_IDS'] = default_params['HALOS_IDS']
    default_params['BIAS_FILE'] = default_params['HALO_BIAS_FILE']
    default_params['NBAR'] = list(default_params['MF_TO_NBAR']*(default_params['CELL_SIZE']**3))
    default_params['NCENTRAL'] = default_params['NTRACERS']*[ 20.0 ]
    default_params['NSIGMA'] = default_params['NTRACERS']*[1000000.0]
    default_params['SHOT_FUDGE'] = default_params['NHALOS']*[0.0]
    default_params['SIGZ_EST'] = default_params['NTRACERS']*[0.0000001]
    default_params['ADIP'] = default_params['NTRACERS']*[0.0000000001]
    default_params['VDISP'] = default_params['ADIP']
    default_params['HALOS_SIGZ_EST'] = default_params['NHALOS']*[0.00001]
    default_params['HALOS_ADIP'] = default_params['HALOS_SIGZ_EST']
    default_params['HALOS_VDISP'] = default_params['HALOS_SIGZ_EST']

    return default_params

#Example of change
#pp(MTPK_code_options(CELL_SIZE = 0.5, SEL_FUN_DATA = True))
#pp(MTPK_code_options(NHALOS = 5))
#pp(MTPK_code_options(N_HALOS = 5))

###################################################

    
# Add path to /inputs directory in order to load inputs
# Change as necessary according to your installation
this_dir = os.getcwd()
input_dir = this_dir + '/inputs'
sys.path.append(input_dir)

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

#Parameters
physical_options = MTPK_physical_params()
code_options = MTPK_code_options()
strkph = str(code_options['KPH_CENTRAL'])
########################## Some other cosmological quantities ######################################
h = physical_options['H0']/100.
cH = 299792.458*h/H(physical_options['H0'], physical_options['OMEGA_M'], physical_options['OMEGA_DE'], physical_options['W0'], physical_options['W1'], physical_options['ZCENTRAL'])  # c/H(z) , in units of h^-1 Mpc


# Velocity dispersion. vdisp is defined on inputs with units of km/s
vdisp = np.asarray(code_options['VDISP']) #km/s
sigma_v = code_options['VDISP']/H(100,physical_options['OMEGA_M'], physical_options['OMEGA_DE'],-1,0.0,physical_options['ZCENTRAL']) #Mpc/h
a_vdisp = vdisp/physical_options['C_LIGHT'] #Adimensional vdisp

# Redshift errors. sigz_est is defined on inputs, and is adimensional
sigz_est = np.asarray(code_options['SIGZ_EST'])
sigma_z = sigz_est*physical_options['C_LIGHT']/H(100,physical_options['OMEGA_M'], physical_options['OMEGA_DE'],-1,0.0,physical_options['ZCENTRAL']) # Mpc/h

# Joint factor considering v dispersion and z error
sig_tot = np.sqrt(sigma_z**2 + sigma_v**2) #Mpc/h
a_sig_tot = np.sqrt(sigz_est**2 + a_vdisp**2) #Adimensional sig_tot

#############Calling CAMB for calculations of the spectra#################
print('Beggining CAMB calculations\n')

nklist = 1000
k_camb = np.logspace(np.log10(code_options['K_MIN_CAMB']),np.log10(code_options['K_MAX_CAMB']),nklist)

kc, pkc = camb_spectrum(physical_options['H0'], physical_options['OMEGA_B'], physical_options['OMEGA_C'], physical_options['W0'], physical_options['W1'], physical_options['Z_RE'], physical_options['ZCENTRAL'], physical_options['A_S'], physical_options['N_SA'], code_options['K_MIN_CAMB'], code_options['K_MAX_CAMB'], code_options['WHICHSPEC'])[:2]
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

try:
    gal_bias = np.loadtxt(input_dir + "/" + bias_file)
except:
    print("Could not find bias file:", bias_file,". Please check your /inputs directory.")
    print("Aborting now...")
    sys.exit(-1)

gal_adip = np.asarray(code_options['ADIP'])
gal_sigz_est = np.asarray(code_options['SIGZ_EST'])
gal_vdisp = np.asarray(code_options['VDISP'])
a_gal_sig_tot = np.sqrt((gal_vdisp/physical_options['C_LIGHT'])**2 + gal_sigz_est**2)

#####################################################
# Generate real- and Fourier-space grids for FFTs
#####################################################
# print 'Generating the k-space Grid...'
L_x = code_options['N_X']*code_options['CELL_SIZE']
L_y = code_options['N_Y']*code_options['CELL_SIZE']
L_z = code_options['N_Z']*code_options['CELL_SIZE']
grid = gr.grid3d(code_options['N_X'], code_options['N_Y'], code_options['N_Z'], L_x, L_y, L_z)

print ()
print ('Geometry: (nx,ny,nz) = (' +str(code_options['N_X'])+','+str(code_options['N_Y'])+','+str(code_options['N_Z'])+'),  cell_size=' + str(code_options['CELL_SIZE']) + ' h^-1 Mpc')


print()
if code_options['WHICHSPEC'] == 0:
    print ('Using LINEAR power spectrum from CAMB')
elif code_options['WHICHSPEC'] == 1:
    print ('Using power spectrum from CAMB + HaloFit')
else:
    print ('Using power spectrum from CAMB + HaloFit with PkEqual')

print()
print ('----------------------------------')
print()

#####################################################
# Start computing physical sizes of boxes
#####################################################
box_vol = L_x*L_y*L_z            # Box's volume
L_max = np.sqrt(L_x*L_x + L_y*L_y + L_z*L_z)    


nn = int(np.sqrt(code_options['N_X']**2 + code_options['N_Y']**2 + code_options['N_Z']**2))
kk_bar = np.fft.fftfreq(nn)

try:
    kmax_phys = code_options['KMAX_PHYS']
    kmaxbar = min(0.5,kmax_phys*code_options['CELL_SIZE']/2.0/np.pi)
    kmax_phys = kmaxbar*2*np.pi/code_options['CELL_SIZE']
except:
    kmax_phys = 0.5 # in h/Mpc
    kmaxbar = min(0.4,kmax_phys*code_options['CELL_SIZE']/2.0/np.pi)
    kmax_phys = kmaxbar*2*np.pi/code_options['CELL_SIZE']


dk0 = 3.0/np.power(code_options['N_X']*code_options['N_Y']*code_options['N_Z'],1/3.)/(2.0*np.pi)
dk_phys = 2.0*np.pi*dk0/code_options['CELL_SIZE']

# Ensure that the binning is at least a certain size
dk_phys = max(dk_phys,code_options['DKPH_BIN'])
# Fourier bins in units of frequency
dk0 = dk_phys*code_options['CELL_SIZE']/2.0/np.pi

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
    kmin_phys = code_options['KMIN_PHYS']
    kminbar = kmin_phys*code_options['CELL_SIZE']/2.0/np.pi
except:
    pass

### K_MAX_MIN
num_binsk=np.int((kmaxbar-kminbar)/dk0)
dk_bar = dk0*np.ones(num_binsk)
k_bar = kminbar + dk0*np.arange(num_binsk)
r_bar = 1/2.0 + ((1.0*code_options['N_X'])/num_binsk)*np.arange(num_binsk)

#
# k in physical units
#
kph = k_bar*2*np.pi/code_options['CELL_SIZE']

##############################################
# Define the "effective bias" as the amplitude of the monopole
try:
    kdip_phys
except:
    kdip_phys = 1./(code_options['CELL_SIZE']*(code_options['N_Z_ORIG'] + code_options['N_Z']/2.))
else:
    print ('ATTENTION: pre-defined (on input) alpha-dipole k_dip [h/Mpc]=', '%1.4f'%kdip_phys)

try:
    dip = np.asarray(gal_adip) * kdip_phys
except:
    dip = 0.0

pk_mg = pkmg.pkmg(gal_bias,dip,matgrowcentral,k_camb,a_gal_sig_tot,cH,zcentral)

monopoles = pk_mg.mono
quadrupoles = pk_mg.quad
