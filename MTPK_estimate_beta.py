'''
This code computes the power spectrum for the catalogues

------
Yields
------

ValueError 
 Raised when a function gets an argument of correct type but improper value

NameError
 Raised when an object could not be found

'''

import numpy as np
import os, sys
import h5py
import glob
from time import time
from scipy import interpolate
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy.sparse import vstack
import pandas as pd

# My classes -- functions used by the MTPK suite
import fkp_multitracer_beta as fkpmt
import fkp_class_beta as fkp
import pk_multipoles_gauss as pkmg
import pk_crossmultipoles_gauss as pkmg_cross
from camb_spec import camb_spectrum
from analytical_selection_function import *
import grid3D as gr
from itertools import combinations

def MTPK_estimate(cat_specs, my_cosmology, my_code_options, dir_maps, dir_data, dir_specs, handle_data = "ExSHalos"):
    '''
    Initial code started by Arthur E. da Mota Loureiro, 04/2015 
    Additonal changes by R. Abramo 07/2015, 02/2016
    Added multi-tracer generalization and other changes, 03-12/2016
    Added Halo Model generalization and other changes, 01/2018
    Added cross-spectra -- Francisco Maion, 01/2020
    '''
    
    # # Add path to /inputs directory in order to load inputs
    # # Change as necessary according to your installation
    # this_dir = os.getcwd()
        
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

    ###################################################
    #Defining all the parameters from new classes here
    ###################################################
    #Cosmological quantities
    h = my_cosmology.h
    H0 = h*100.
    clight = my_cosmology.c_light
    cH = clight*h/my_cosmology.H(my_cosmology.zcentral, False) # c/H(z) , in units of h^-1 Mpc
    OmegaDE = my_cosmology.Omega0_DE
    Omegab = my_cosmology.Omega0_b
    Omegac = my_cosmology.Omega0_cdm
    A_s = my_cosmology.A_s
    gamma = my_cosmology.gamma
    matgrow = my_cosmology.matgrow
    w0 = my_cosmology.w0
    w1 = my_cosmology.w1
    z_re = my_cosmology.z_re
    zcentral = my_cosmology.zcentral
    n_SA = my_cosmology.n_s
    vdisp = np.asarray(my_code_options.vdisp) #Velocity dispersion [km/s]
    sigma_v = my_code_options.vdisp/my_cosmology.H(my_cosmology.zcentral, False) #Mpc/h
    a_vdisp = vdisp/my_cosmology.c_light #Adimensional vdisp
    sigz_est = np.asarray(my_code_options.sigz_est) #adimensional
    sigma_z = sigz_est*my_cosmology.c_light/my_cosmology.H(my_cosmology.zcentral, False) # Mpc/h
    #Code options
    verbose = my_code_options.verbose
    whichspec = my_code_options.whichspec
    sig_tot = np.sqrt(sigma_z**2 + sigma_v**2) #Mpc/h
    a_sig_tot = np.sqrt(sigz_est**2 + a_vdisp**2) #Adimensional sig_tot
    gal_bias = my_code_options.bias_file
    adip = my_code_options.adip
    gal_adip = np.asarray(adip)
    gal_sigz_est = np.asarray(sigz_est)
    gal_vdisp = np.asarray(vdisp)
    a_gal_sig_tot = np.sqrt((gal_vdisp/clight)**2 + gal_sigz_est**2)
    mas_method = my_code_options.mas_method
    if mas_method == 'NGP':
        mas_power = 1.0
    elif mas_method == 'CIC':
        mas_power = 2.0
    elif mas_method == 'TSC':
        mas_power = 3.0
    elif mas_method == 'PCS':
        mas_power = 4.0
    else:
        raise ValueError("Wrong gridding method (mas_method)")
    use_mask = my_code_options.use_mask
    sel_fun_data = my_code_options.sel_fun_data
    strkph = str(my_code_options.kph_central) #Save estimations
    use_theory_spectrum = my_code_options.use_theory_spectrum
    theory_spectrum_file = my_code_options.theory_spectrum_file
    k_min_camb = my_code_options.k_min_CAMB
    k_max_camb = my_code_options.k_max_CAMB
    n_x = my_code_options.n_x
    n_y = my_code_options.n_y
    n_z = my_code_options.n_z
    n_x_orig = my_code_options.n_x_orig
    n_y_orig = my_code_options.n_y_orig
    n_z_orig = my_code_options.n_z_orig
    use_padding = my_code_options.use_padding
    padding_length = my_code_options.padding_length
    cell_size = my_code_options.cell_size
    ntracers = my_code_options.ntracers
    nbar = my_code_options.nbar
    ncentral = my_code_options.ncentral
    nsigma = my_code_options.nsigma
    sel_fun_file = my_code_options.sel_fun_file
    mass_fun = my_code_options.mass_fun
    n_maps = cat_specs.n_maps
    use_kmax_phys = my_code_options.use_kmax_phys
    kmax_phys = my_code_options.kmax_phys
    dkph_bin = my_code_options.dkph_bin
    use_kmin_phys = my_code_options.use_kmin_phys
    kmin_phys = my_code_options.kmin_phys
    use_kdip_phys = my_code_options.use_kdip_phys
    kdip_phys = my_code_options.kdip_phys
    method = my_code_options.method
    multipoles_order = my_code_options.multipoles_order
    do_cross_spectra = my_code_options.do_cross_spectra
    ntracers = cat_specs.ntracers

    #Backing to the code
    handle_sims = handle_data
    handle_estimates = handle_data

    ###################
    if verbose:
        print()
        print()
        print( 'This is the Multi-tracer power spectrum estimator')

        print()
        print('Handle of this run (fiducial spectra, biases, etc.): ', handle_estimates)
        print()
    else:
        pass

    # If directories do not exist, create them now
    if not os.path.exists(dir_specs):
        os.makedirs(dir_specs)

    #############Calling CAMB for calculations of the spectra#################
    if verbose:
        print('Beggining CAMB calculations\n')
    else:
        pass

    if use_theory_spectrum:
        if verbose:
            print('Using pre-existing power spectrum in file:',theory_spectrum_file)
        else:
            pass
        kcpkc = np.loadtxt(theory_spectrum_file)
        if kcpkc.shape[1] > kcpkc.shape[0]: 
            k_camb=kcpkc[0]
            Pk_camb=kcpkc[1]
        else:
            k_camb=kcpkc[:,0]
            Pk_camb=kcpkc[:,1]
    else:
        if verbose:
            print('Computing matter power spectrum for given cosmology...\n')
        else:
            pass

        # It is strongly encouraged to use k_min >= 1e-4, since it is a lot faster

        nklist = 1000
        k_camb = np.logspace(np.log10(k_min_camb),np.log10(k_max_camb),nklist)
        
        kc, pkc = camb_spectrum(H0, Omegab, Omegac, w0, w1, z_re, zcentral, A_s, n_SA, k_min_camb, k_max_camb, whichspec)
        Pk_camb = np.asarray( np.interp(k_camb, kc, pkc) )

    # Ended CAMB calculation #####################################

    # Construct spectrum that decays sufficiently rapidly, and interpolate, using an initial ansatz for
    #power-law of P ~ k^(-1) [good for HaloFit]
    k_interp = np.append(k_camb,np.array([2*k_camb[-1],4*k_camb[-1],8*k_camb[-1],16*k_camb[-1],32*k_camb[-1],64*k_camb[-1],128*k_camb[-1]]))
    P_interp = np.append(Pk_camb,np.array([1./2.*Pk_camb[-1],1./4*Pk_camb[-1],1./8*Pk_camb[-1],1./16*Pk_camb[-1],1./32*Pk_camb[-1],1./64*Pk_camb[-1],1./128*Pk_camb[-1]]))
    pow_interp=interpolate.PchipInterpolator(k_interp,P_interp)

    #####################################################
    # Generate real- and Fourier-space grids for FFTs
    #####################################################

    if verbose:
        print('.')
        print('Generating the k-space Grid...')
        print('.')
    else:
        pass

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

    L_x = n_x_box*cell_size ; L_y = n_y_box*cell_size ; L_z = n_z_box*cell_size
    grid = gr.grid3d(n_x_box,n_y_box,n_z_box,L_x,L_y,L_z)

    grid_orig = gr.grid3d(n_x,n_y,n_z,n_x*cell_size,n_y*cell_size,n_z*cell_size)

    # Selection function
    # If given by data
    if sel_fun_data:
        try:
            h5map = h5py.File(dir_data + '/' + sel_fun_file,'r')
            h5data = h5map.get(list(h5map.keys())[0])
            nbm = np.asarray(h5data,dtype='float32')
            h5map.close
        except:
            raise NameError('Files with data selection function not to found! Check your directory ', dir_data)
        if len(nbm.shape)==3:
            n_bar_matrix_fid = np.zeros((1,n_x,n_y,n_z))
            n_bar_matrix_fid[0] = nbm
        elif len(nbm.shape)==4:
            n_bar_matrix_fid = nbm
            if (np.shape(n_bar_matrix_fid)[1] != n_x) or (np.shape(n_bar_matrix_fid)[2] != n_y) or (np.shape(n_bar_matrix_fid)[3] != n_z):
                raise ValueError('Dimensions of data selection function box =', n_bar_matrix_fid.shape, ' , differ from input file!')
        else:
            raise ValueError('Data selection function has funny dimensions:', nbm.shape)

    else:
        try:
            nbar = mass_fun*cell_size**3
            ncentral = 10.0*nbar**0
            nsigma = 10000.0*nbar**0 
        except:
            if verbose:
                print("Att.: using analytical selection function for galaxies (check parameters in input file).")
                print("Using n_bar, n_central, n_sigma  from input file")
            else:
                pass

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
            raise NameError('Mask not found! Check your directory ', dir_data)
        if (np.shape(mask)[0] != n_x) or (np.shape(mask)[1] != n_y) or (np.shape(mask)[2] != n_z):
            raise ValueError('Dimensions of mask, =', mask.shape, ' , differ from input file!')
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

    mapnames_sims = sorted(glob.glob(dir_maps + '/*.hdf5'))
    if len(mapnames_sims)==0 :
        raise NameError('Simulated map files not found! Check input simulation files.')
    if len(mapnames_sims) != n_maps :
        print ('You are using', n_maps, ' mocks out of the existing', len(mapnames_sims))
        answer = input('Continue anyway? y/n  ')
        if answer!='y':
            print ('Aborting now...')
            sys.exit(-1)
    if verbose:
        print ('Will use the N =', n_maps, ' simulation-only maps contained in directory', dir_maps)
    else:
        pass

    if verbose:
        print(".")

        print ('Geometry: (nx,ny,nz) = (' +str(n_x)+','+str(n_y)+','+str(n_z)+'),  cell_size=' + str(cell_size) + ' h^-1 Mpc')
    else:
        pass
    
    # Apply padding, if it exists
    try:
        if verbose:
            print ('Geometry including bounding box: (nx,ny,nz) = (' +str(n_x_box)+','+str(n_y_box)+','+str(n_z_box) + ')')
        else:
            pass
    except:
        pass

    if verbose:
        print(".")
        if whichspec == 0:
            print ('Using LINEAR power spectrum from CAMB')
        elif whichspec == 1:
            print ('Using power spectrum from CAMB + HaloFit')
        else:
            print ('Using power spectrum from CAMB + HaloFit with PkEqual')
    else:
        pass

    if verbose:
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
    if use_kmax_phys:
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
    dk_phys = max(dk_phys,dkph_bin)
    # Fourier bins in units of frequency
    dk0 = dk_phys*cell_size/2.0/np.pi

    #  Physically, the maximal useful k is perhaps k =~ 0.3 h/Mpc (non-linear scale)
    np.set_printoptions(precision=3)

    if verbose:
        print ('Will estimate modes up to k[h/Mpc] = ', '%.4f'% kmax_phys,' in bins with Delta_k =', '%.4f' %dk_phys)

        print(".")
        print ('----------------------------------')
        print(".")
    else:
        pass

    #R This line makes some np variables be printed with less digits
    np.set_printoptions(precision=6)

    #R Here are the k's that will be estimated (in grid units):
    kgrid = grid.grid_k
    kminbar = 1./4.*(kgrid[1,0,0]+kgrid[0,1,0]+kgrid[0,0,1]) + dk0/4.0

    ### K_MAX_MIN
    if use_kmin_phys:
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
    if use_kdip_phys:
        print ('ATTENTION: pre-defined (on input) alpha-dipole k_dip [h/Mpc]=', '%1.4f'%kdip_phys)
        pass
    else:
        kdip_phys = 1./(cell_size*(n_z_orig + n_z/2.))    
    
    try:
        dip = np.asarray(gal_adip) * kdip_phys
    except:
        dip = 0.0

    pk_mg = pkmg.pkmg(gal_bias,dip,matgrow,k_camb,a_gal_sig_tot,cH,zcentral)

    monopoles = pk_mg.mono
    quadrupoles = pk_mg.quad

    # Hexadecapoles only in the Kaiser approximation
    hexa = 8./35*matgrow**2

    hexapoles = hexa*np.power(monopoles,0)

    try:
        pk_mg_cross = pkmg_cross.pkmg_cross(gal_bias,dip,matgrow,k_camb,a_gal_sig_tot,cH,zcentral)
        cross_monopoles = pk_mg_cross.monos
        cross_quadrupoles = pk_mg_cross.quads
    except:
        cross_monopoles = np.zeros((len(k_camb),1))
        cross_quadrupoles = np.zeros((len(k_camb),1))

    cross_hexapoles = hexa*np.power(cross_monopoles,0)

    # Compute effective dipole and bias of tracers
    kph_central = my_code_options.kph_central
    where_kph_central = np.argmin(np.abs(k_camb - kph_central))

    effadip = dip*matgrow/(0.00000000001 + kph_central)
    effbias = np.sqrt(monopoles[:,where_kph_central])

    # Get effective bias (sqrt of monopoles) for final tracers
    pk_mg = pkmg.pkmg(gal_bias,gal_adip,matgrow,k_camb,a_gal_sig_tot,cH,zcentral)

    monopoles = pk_mg.mono
    quadrupoles = pk_mg.quad

    # Compute effective dipole and bias of tracers
    where_kph_central = np.argmin(np.abs(k_camb - kph_central))

    effadip = gal_adip*matgrow/(0.00000000001 + kph_central)
    effbias = np.sqrt(monopoles[:,where_kph_central])

    if verbose:
        print()
        print ('----------------------------------')
        print()
    else:
        pass

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


    if verbose:
        print ('Central physical k values where spectra will be estimated:', kph_central)
    else:
        pass

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

    if verbose:
        print ('Initializing the k-binning matrix...')
    else:
        pass
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

    if verbose:
        print ('Done with k-binning matrices. Time cost: ', np.int32((time()-tempor)*1000)/1000., 's')
        print ('Memory occupied by the binning matrix: ', MRk.nnz)
    else:
        pass

    #R We defined "target" k's , but <k> on bins may be slightly different
    kkav=(((MRk*kflat))/(kkbar_counts+0.00001))
    if verbose:
        print ('Originally k_bar was defined as:', [ "{:1.4f}".format(x) for x in k_bar[10:16:2] ])
        print ('The true mean of k for each bin is:', [ "{:1.4f}".format(x) for x in kkav[10:16:2] ])
        print()
        print ('----------------------------------')
        print()
        print ('Now estimating the power spectra...')
    else:
        pass

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
    if verbose:
        print ('Starting power spectra estimation')
    else:
        pass

    # Initialize outputs

    if method == 'both':
        
        if multipoles_order == 0:
            # Multitracer method: monopole, quadrupole, th. covariance(FKP-like)
            # Original (convolved) spectra:
            P0_data = np.zeros((n_maps,ntracers,num_binsk))

            # Traditional (FKP) method
            P0_fkp = np.zeros((n_maps,ntracers,num_binsk))

            # Cross spectra

            if do_cross_spectra == True and ntracers > 1:
                Cross0 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))

            else:
                pass

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


            if verbose:
                print( "Initializing multi-tracer estimation toolbox...")
            else:
                pass

            fkp_mult = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,effbias_mt,cell_size,n_x_box,n_y_box,n_z_box,n_x_orig,n_y_orig,n_z_orig,MRk,powercentral,mas_power, multipoles_order, verbose)

            ##
            # UPDATED THIS TO NEW FKP CLASS WITH AUTO- AND CROSS-SPECTRA
            if verbose:
                print( "Initializing traditional (FKP) estimation toolbox...")
            else:
                pass
            fkp_many = fkp.fkp_init(num_binsk, n_bar_matrix_fid, effbias, cell_size, n_x_box, n_y_box, n_z_box, n_x_orig, n_y_orig, n_z_orig, MRk, powercentral,mas_power, multipoles_order, do_cross_spectra, ntracers, verbose)

            '''
            Because of the very sensitive nature of shot noise subtraction 
            in the Jing de-aliasing, it may be better to normalize the counts of the 
            selection function to each map, and not to the mean of the maps.
            ''' 
            normsel = np.zeros((n_maps,ntracers))

            if verbose:
                print ("... done. Starting computations for each map (box) now.")
                print()
            else:
                pass

            #################################
            est_bias_fkp = np.zeros(ntracers)
            est_bias_mt = np.zeros(ntracers)

            for nm in range(n_maps):
                time_start=time()
                if verbose:
                    print ('Loading simulated box #', nm)
                else:
                    pass
                h5map = h5py.File(mapnames_sims[nm],'r')
                maps = np.asarray(h5map.get(list(h5map.keys())[0]))
                if not maps.shape == (ntracers,n_x,n_y,n_z):
                    raise ValueError('Unexpected shape of simulated maps! Found:', maps.shape)
                h5map.close()

                if verbose:
                    print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))
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
                if verbose:
                    print ('  Estimating FKP power spectra...')
                else:
                    pass
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

                    if verbose:
                        print(" Normalized selection function/map at:",np.around(normsel[nm],3))
                    else:
                        pass
                    
                if nm==0:
                    FKPmany = fkp_many.fkp((normsel[nm]*maps.T).T)
                    P0_fkp[nm] = fkp_many.P_ret
                    if do_cross_spectra == True and ntracers > 1:
                        Cross0[nm] = fkp_many.cross_spec
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2
                else:
                    FKPmany = fkp_many.fkp((normsel[nm]*maps.T).T)
                    P0_fkp[nm] = fkp_many.P_ret
                    if do_cross_spectra == True and ntracers > 1:
                        Cross0[nm] = fkp_many.cross_spec
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2

                #################################
                # Now, the multi-tracer method
                if verbose:
                    print ('  Now estimating multi-tracer spectra...')
                else:
                    pass
                
                if nm==0:
                    FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
                    P0_data[nm] = fkp_mult.P0_mu_ret
                else:
                    FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
                    P0_data[nm] = fkp_mult.P0_mu_ret

                #CORRECT FOR BIN AVERAGING
                P0_data[nm] = P0_data[nm]/k_av_corr
                P0_fkp[nm] = P0_fkp[nm]/k_av_corr
                if do_cross_spectra == True and ntracers > 1:
                    Cross0[nm] = Cross0[nm]/k_av_corr
                else:
                    pass

                if nm==0:
                    # If data bias is different from mocks
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print ("  Effective biases of the simulated maps:")
                        print ("   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("        FKP=", ["%.3f"%b for b in est_bias_fkp])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
                else:
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print( "  Effective biases of these maps:")
                        print( "   Fiducial=", ["%.3f"%b for b in effbias])
                        print( "        FKP=", ["%.3f"%b for b in est_bias_fkp])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass

            #Update nbarbar to reflect actual
            nbarbar = nbarbar/np.mean(normsel,axis=0)

            # Correct missing factor of 2 in definition
            Theor_Cov_FKP = 2.0*np.mean(ThCov_fkp,axis=0)

            ################################################################################
            ################################################################################
    
            
            time_end=time()
            if verbose:
                print ('Total time cost for estimation of spectra: ', time_end - time_start)
            else:
                pass

            ################################################################################
            ################################################################################

            tempor=time()

            ################################################################################
            ################################################################################

            if verbose:
                print ('Applying mass assignement window function corrections...')
            else:
                pass

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
            if do_cross_spectra == True and ntracers > 1:
                Cross0_mean = np.mean(Cross0,axis=0)
            else:
                pass

            # Cross0 and Cross2 are outputs of the FKP code, so they come out without the bias.
            # We can easily put back the bias by multiplying:
            # CrossX = cross_effbias**2 * CrossX
            if do_cross_spectra == True and ntracers > 1:
                cross_effbias = np.zeros(ntracers*(ntracers-1)//2)#AQUI
                index=0
                for nt in range(ntracers):
                    for ntp in range(nt+1,ntracers):
                        cross_effbias[index] = np.sqrt(effbias[nt]*effbias[ntp])
                        index += 1
            else:
                pass        

            # FKP and Cross measurements need to have the bias returned in their definitions
            P0_fkp = np.transpose((effbias**2*np.transpose(P0_fkp,axes=(0,2,1))),axes=(0,2,1))

            if do_cross_spectra == True and ntracers > 1:
                C0_fkp = np.transpose((cross_effbias**2*np.transpose(Cross0,axes=(0,2,1))),axes=(0,2,1))
            else:
                pass

            #   SAVE these spectra
            P0_save=np.reshape(P0_data,(n_maps,ntracers*pow_bins))
            P0_fkp_save=np.reshape(P0_fkp,(n_maps,ntracers*pow_bins))
            if do_cross_spectra == True and ntracers > 1:
                C0_fkp_save=np.reshape(C0_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
            else:
                pass

            if do_cross_spectra == True and ntracers > 1:
                data = [ kph ]
                columns = ['k']
                comb = list( combinations( [i for i in range(ntracers)], 2 ) )
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )
                        data.append( P0_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_MT_map{i}_tracer{j}' )
                    for k in range( len(comb) ):
                        data.append( C0_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS0_map{i}_tracers{comb[k][0]}{comb[k][1]}' )
                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)
            else:
                data = [ kph ]
                columns = ['k']
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )
                        data.append( P0_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_MT_map{i}_tracer{j}' )
                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)

            
        elif multipoles_order == 2:
            # Multitracer method: monopole, quadrupole, th. covariance(FKP-like)
            # Original (convolved) spectra:
            P0_data = np.zeros((n_maps,ntracers,num_binsk))
            P2_data = np.zeros((n_maps,ntracers,num_binsk))

            # Traditional (FKP) method
            P0_fkp = np.zeros((n_maps,ntracers,num_binsk))
            P2_fkp = np.zeros((n_maps,ntracers,num_binsk))

            # Cross spectra
            if do_cross_spectra == True and ntracers > 1:
                Cross0 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
                Cross2 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
            else:
                pass
            
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

            if verbose:
                print( "Initializing multi-tracer estimation toolbox...")
            else:
                pass

            fkp_mult = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,effbias_mt,cell_size,n_x_box,n_y_box,n_z_box,n_x_orig,n_y_orig,n_z_orig,MRk,powercentral,mas_power, multipoles_order, verbose)

            ##
            # UPDATED THIS TO NEW FKP CLASS WITH AUTO- AND CROSS-SPECTRA
            if verbose:
                print( "Initializing traditional (FKP) estimation toolbox...")
            else:
                pass
            fkp_many = fkp.fkp_init(num_binsk, n_bar_matrix_fid, effbias, cell_size, n_x_box, n_y_box, n_z_box, n_x_orig, n_y_orig, n_z_orig, MRk, powercentral,mas_power, multipoles_order, do_cross_spectra, ntracers, verbose)

            '''
            Because of the very sensitive nature of shot noise subtraction 
            in the Jing de-aliasing, it may be better to normalize the counts of the 
            selection function to each map, and not to the mean of the maps.
            ''' 
            normsel = np.zeros((n_maps,ntracers))

            if verbose:
                print ("... done. Starting computations for each map (box) now.")
                print()
            else:
                pass

            #################################


            est_bias_fkp = np.zeros(ntracers)
            est_bias_mt = np.zeros(ntracers)

            for nm in range(n_maps):
                time_start=time()
                if verbose:
                    print ('Loading simulated box #', nm)
                else:
                    pass
                h5map = h5py.File(mapnames_sims[nm],'r')
                maps = np.asarray(h5map.get(list(h5map.keys())[0]))
                if not maps.shape == (ntracers,n_x,n_y,n_z):
                    raise ValueError('Unexpected shape of simulated maps! Found:', maps.shape)
                h5map.close()

                if verbose:
                    print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))
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
                if verbose:
                    print ('  Estimating FKP power spectra...')
                else:
                    pass
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
                    if do_cross_spectra == True and ntracers > 1:
                        Cross0[nm] = fkp_many.cross_spec
                        Cross2[nm] = fkp_many.cross_spec2
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2
                else:
                    FKPmany = fkp_many.fkp((normsel[nm]*maps.T).T)
                    P0_fkp[nm] = fkp_many.P_ret
                    P2_fkp[nm] = fkp_many.P2_ret
                    if do_cross_spectra == True and ntracers > 1:
                        Cross0[nm] = fkp_many.cross_spec
                        Cross2[nm] = fkp_many.cross_spec2
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2

                #################################
                # Now, the multi-tracer method
                if verbose:
                    print ('  Now estimating multi-tracer spectra...')
                else:
                    pass
                if nm==0:
                    FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
                    P0_data[nm] = fkp_mult.P0_mu_ret
                    P2_data[nm] = fkp_mult.P2_mu_ret
                else:
                    FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
                    P0_data[nm] = fkp_mult.P0_mu_ret
                    P2_data[nm] = fkp_mult.P2_mu_ret

                #CORRECT FOR BIN AVERAGING
                P0_data[nm] = P0_data[nm]/k_av_corr
                P2_data[nm] = P2_data[nm]/k_av_corr
                P0_fkp[nm] = P0_fkp[nm]/k_av_corr
                P2_fkp[nm] = P2_fkp[nm]/k_av_corr
                if do_cross_spectra == True and ntracers > 1:
                    Cross0[nm] = Cross0[nm]/k_av_corr
                    Cross2[nm] = Cross2[nm]/k_av_corr

                if nm==0:
                    # If data bias is different from mocks
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print ("  Effective biases of the simulated maps:")
                        print ("   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("        FKP=", ["%.3f"%b for b in est_bias_fkp])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
                else:
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print( "  Effective biases of these maps:")
                        print( "   Fiducial=", ["%.3f"%b for b in effbias])
                        print( "        FKP=", ["%.3f"%b for b in est_bias_fkp])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass

            #Update nbarbar to reflect actual
            nbarbar = nbarbar/np.mean(normsel,axis=0)

            # Correct missing factor of 2 in definition
            Theor_Cov_FKP = 2.0*np.mean(ThCov_fkp,axis=0)

            ################################################################################
            ################################################################################
    

            time_end=time()
            if verbose:
                print ('Total time cost for estimation of spectra: ', time_end - time_start)
            else:
                pass

            ################################################################################
            ################################################################################

            tempor=time()

            ################################################################################
            ################################################################################

            if verbose:
                print ('Applying mass assignement window function corrections...')
            else:
                pass

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
            if do_cross_spectra == True and ntracers > 1:
                Cross0_mean = np.mean(Cross0,axis=0)
            else:
                pass

            # Cross0 and Cross2 are outputs of the FKP code, so they come out without the bias.
            # We can easily put back the bias by multiplying:
            # CrossX = cross_effbias**2 * CrossX
            if do_cross_spectra == True and ntracers > 1:
                cross_effbias = np.zeros(ntracers*(ntracers-1)//2)
                index=0
                for nt in range(ntracers):
                    for ntp in range(nt+1,ntracers):
                        cross_effbias[index] = np.sqrt(effbias[nt]*effbias[ntp])
                        index += 1
            else:
                pass        

            # FKP and Cross measurements need to have the bias returned in their definitions
            P0_fkp = np.transpose((effbias**2*np.transpose(P0_fkp,axes=(0,2,1))),axes=(0,2,1))
            P2_fkp = np.transpose((effbias**2*np.transpose(P2_fkp,axes=(0,2,1))),axes=(0,2,1))

            if do_cross_spectra == True and ntracers > 1:
                C0_fkp = np.transpose((cross_effbias**2*np.transpose(Cross0,axes=(0,2,1))),axes=(0,2,1))
                C2_fkp = np.transpose((cross_effbias**2*np.transpose(Cross2,axes=(0,2,1))),axes=(0,2,1))
            else:
                pass

            # SAVE these spectra
            P0_save=np.reshape(P0_data,(n_maps,ntracers*pow_bins))
            P0_fkp_save=np.reshape(P0_fkp,(n_maps,ntracers*pow_bins))

            P2_save=np.reshape(P2_data,(n_maps,ntracers*pow_bins))
            P2_fkp_save=np.reshape(P2_fkp,(n_maps,ntracers*pow_bins))

            if do_cross_spectra == True and ntracers > 1:
                C0_fkp_save=np.reshape(C0_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
                C2_fkp_save=np.reshape(C2_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
            else:
                pass

            if do_cross_spectra == True and ntracers > 1:
                data = [ kph ]
                columns = ['k']
                comb = list( combinations( [i for i in range(ntracers)], 2 ) )
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )
                        data.append( P0_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_MT_map{i}_tracer{j}' )

                        data.append( P2_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_FKP_map{i}_tracer{j}' )
                        data.append( P2_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_MT_map{i}_tracer{j}' )
                    for k in range( len(comb) ):
                        data.append( C0_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS0_map{i}_tracers{comb[k][0]}{comb[k][1]}' )
                    for k in range( len(comb) ):
                        data.append( C2_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS2_map{i}_tracers{comb[k][0]}{comb[k][1]}' )
                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)
            else:
                data = [ kph ]
                columns = ['k']
                comb = list( combinations( [i for i in range(ntracers)], 2 ) )
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )
                        data.append( P0_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_MT_map{i}_tracer{j}' )

                        data.append( P2_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_FKP_map{i}_tracer{j}' )
                        data.append( P2_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_MT_map{i}_tracer{j}' )
                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)

        else:
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
            if do_cross_spectra == True and ntracers > 1:
                Cross0 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
                Cross2 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
                Cross4 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
            else:
                pass

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

            if verbose:
                print( "Initializing multi-tracer estimation toolbox...")
            else:
                pass

            fkp_mult = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,effbias_mt,cell_size,n_x_box,n_y_box,n_z_box,n_x_orig,n_y_orig,n_z_orig,MRk,powercentral,mas_power, multipoles_order, verbose)

            ##
            # UPDATED THIS TO NEW FKP CLASS WITH AUTO- AND CROSS-SPECTRA
            if verbose:
                print( "Initializing traditional (FKP) estimation toolbox...")
            else:
                pass
            fkp_many = fkp.fkp_init(num_binsk, n_bar_matrix_fid, effbias, cell_size, n_x_box, n_y_box, n_z_box, n_x_orig, n_y_orig, n_z_orig, MRk, powercentral,mas_power, multipoles_order, do_cross_spectra, ntracers, verbose)

            '''
            Because of the very sensitive nature of shot noise subtraction 
            in the Jing de-aliasing, it may be better to normalize the counts of the 
            selection function to each map, and not to the mean of the maps.
            ''' 
            normsel = np.zeros((n_maps,ntracers))

            if verbose:
                print ("... done. Starting computations for each map (box) now.")
                print()
            else:
                pass

            #################################


            est_bias_fkp = np.zeros(ntracers)
            est_bias_mt = np.zeros(ntracers)

            for nm in range(n_maps):
                time_start=time()
                if verbose:
                    print ('Loading simulated box #', nm)
                else:
                    pass
                h5map = h5py.File(mapnames_sims[nm],'r')
                maps = np.asarray(h5map.get(list(h5map.keys())[0]))
                if not maps.shape == (ntracers,n_x,n_y,n_z):
                    raise ValueError('Unexpected shape of simulated maps! Found:', maps.shape)
                h5map.close()

                if verbose:
                    print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))
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
                if verbose:
                    print ('  Estimating FKP power spectra...')
                else:
                    pass
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
                    if do_cross_spectra == True and ntracers > 1:
                        Cross0[nm] = fkp_many.cross_spec
                        Cross2[nm] = fkp_many.cross_spec2
                        Cross4[nm] = fkp_many.cross_spec4
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2
                else:
                    FKPmany = fkp_many.fkp((normsel[nm]*maps.T).T)
                    P0_fkp[nm] = fkp_many.P_ret
                    P2_fkp[nm] = fkp_many.P2_ret
                    P4_fkp[nm] = fkp_many.P4_ret
                    if do_cross_spectra == True and ntracers > 1:
                        Cross0[nm] = fkp_many.cross_spec
                        Cross2[nm] = fkp_many.cross_spec2
                        Cross4[nm] = fkp_many.cross_spec4
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2

                #################################
                # Now, the multi-tracer method
                if verbose:
                    print ('  Now estimating multi-tracer spectra...')
                else:
                    pass
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
                if do_cross_spectra == True and ntracers > 1:
                    Cross0[nm] = Cross0[nm]/k_av_corr
                    Cross2[nm] = Cross2[nm]/k_av_corr
                    Cross4[nm] = Cross4[nm]/k_av_corr
                else:
                    pass

                if nm==0:
                    # If data bias is different from mocks
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print ("  Effective biases of the simulated maps:")
                        print ("   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("        FKP=", ["%.3f"%b for b in est_bias_fkp])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
                else:
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print( "  Effective biases of these maps:")
                        print( "   Fiducial=", ["%.3f"%b for b in effbias])
                        print( "        FKP=", ["%.3f"%b for b in est_bias_fkp])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
    
            #Update nbarbar to reflect actual
            nbarbar = nbarbar/np.mean(normsel,axis=0)

            # Correct missing factor of 2 in definition
            Theor_Cov_FKP = 2.0*np.mean(ThCov_fkp,axis=0)

            ################################################################################
            ################################################################################

            time_end=time()
            if verbose:
                print ('Total time cost for estimation of spectra: ', time_end - time_start)
            else:
                pass

            ################################################################################
            ################################################################################

            tempor=time()

            ################################################################################
            ################################################################################

            if verbose:
                print ('Applying mass assignement window function corrections...')
            else:
                pass

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
            if do_cross_spectra == True and ntracers > 1:
                Cross0_mean = np.mean(Cross0,axis=0)
            else:
                pass

            # Cross0 and Cross2 are outputs of the FKP code, so they come out without the bias.
            # We can easily put back the bias by multiplying:
            # CrossX = cross_effbias**2 * CrossX
            if do_cross_spectra == True and ntracers > 1:
                cross_effbias = np.zeros(ntracers*(ntracers-1)//2)
                index=0
                for nt in range(ntracers):
                    for ntp in range(nt+1,ntracers):
                        cross_effbias[index] = np.sqrt(effbias[nt]*effbias[ntp])
                        index += 1
            else:
                pass        

            # FKP and Cross measurements need to have the bias returned in their definitions
            P0_fkp = np.transpose((effbias**2*np.transpose(P0_fkp,axes=(0,2,1))),axes=(0,2,1))
            P2_fkp = np.transpose((effbias**2*np.transpose(P2_fkp,axes=(0,2,1))),axes=(0,2,1))
            P4_fkp = np.transpose((effbias**2*np.transpose(P4_fkp,axes=(0,2,1))),axes=(0,2,1))

            if do_cross_spectra == True and ntracers > 1:
                C0_fkp = np.transpose((cross_effbias**2*np.transpose(Cross0,axes=(0,2,1))),axes=(0,2,1))
                C2_fkp = np.transpose((cross_effbias**2*np.transpose(Cross2,axes=(0,2,1))),axes=(0,2,1))
                C4_fkp = np.transpose((cross_effbias**2*np.transpose(Cross4,axes=(0,2,1))),axes=(0,2,1))
            else:
                pass

            # Means
            P0_mean = np.mean(P0_data,axis=0)
            P2_mean = np.mean(P2_data,axis=0)
            P4_mean = np.mean(P4_data,axis=0)
            P0_fkp_mean = np.mean(P0_fkp,axis=0)
            P2_fkp_mean = np.mean(P2_fkp,axis=0)
            P4_fkp_mean = np.mean(P4_fkp,axis=0)
            if do_cross_spectra == True and ntracers > 1:
                Cross0_mean = np.mean(C0_fkp,axis=0)
                Cross2_mean = np.mean(C2_fkp,axis=0)
                Cross4_mean = np.mean(C4_fkp,axis=0)
            else:
                pass

            # SAVE spectra
            P0_save=np.reshape(P0_data,(n_maps,ntracers*pow_bins))
            P0_fkp_save=np.reshape(P0_fkp,(n_maps,ntracers*pow_bins))

            P2_save=np.reshape(P2_data,(n_maps,ntracers*pow_bins))
            P2_fkp_save=np.reshape(P2_fkp,(n_maps,ntracers*pow_bins))

            P4_save=np.reshape(P4_data,(n_maps,ntracers*pow_bins))
            P4_fkp_save=np.reshape(P4_fkp,(n_maps,ntracers*pow_bins))

            if do_cross_spectra == True and ntracers > 1:
                C0_fkp_save=np.reshape(C0_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
                C2_fkp_save=np.reshape(C2_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
                C4_fkp_save=np.reshape(C4_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
            else:
                pass

            if do_cross_spectra == True and ntracers > 1:
                data = [ kph ]
                columns = ['k']
                comb = list( combinations( [i for i in range(ntracers)], 2 ) )
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )
                        data.append( P0_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_MT_map{i}_tracer{j}' )

                        data.append( P2_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_FKP_map{i}_tracer{j}' )
                        data.append( P2_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_MT_map{i}_tracer{j}' )

                        data.append( P4_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P4_FKP_map{i}_tracer{j}' )
                        data.append( P4_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P4_MT_map{i}_tracer{j}' )
                        
                    for k in range( len(comb) ):
                        data.append( C0_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS0_map{i}_tracers{comb[k][0]}{comb[k][1]}' )
                    for k in range( len(comb) ):
                        data.append( C2_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS2_map{i}_tracers{comb[k][0]}{comb[k][1]}' )
                    for k in range( len(comb) ):
                        data.append( C4_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS4_map{i}_tracers{comb[k][0]}{comb[k][1]}' )

                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)
            else:
                data = [ kph ]
                columns = ['k']
                comb = list( combinations( [i for i in range(ntracers)], 2 ) )
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )
                        data.append( P0_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_MT_map{i}_tracer{j}' )

                        data.append( P2_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_FKP_map{i}_tracer{j}' )
                        data.append( P2_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_MT_map{i}_tracer{j}' )

                        data.append( P4_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P4_FKP_map{i}_tracer{j}' )
                        data.append( P4_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P4_MT_map{i}_tracer{j}' )

                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)

    elif method == 'FKP':

        if multipoles_order == 0:

            # Traditional (FKP) method
            P0_fkp = np.zeros((n_maps,ntracers,num_binsk))

            # Cross spectra
            if do_cross_spectra == True and ntracers > 1:
                Cross0 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
            else:
                pass

            # Covariance
            ThCov_fkp = np.zeros((n_maps,ntracers,num_binsk))

            # Range where we estimate some parameters
            myk_min = np.argsort(np.abs(kph-0.1))[0]
            myk_max = np.argsort(np.abs(kph-0.2))[0]
            myran = np.arange(myk_min,myk_max)

            ##
            # UPDATED THIS TO NEW FKP CLASS WITH AUTO- AND CROSS-SPECTRA
            if verbose:
                print( "Initializing traditional (FKP) estimation toolbox...")
            else:
                pass
            fkp_many = fkp.fkp_init(num_binsk, n_bar_matrix_fid, effbias, cell_size, n_x_box, n_y_box, n_z_box, n_x_orig, n_y_orig, n_z_orig, MRk, powercentral,mas_power, multipoles_order, do_cross_spectra, ntracers, verbose)

            '''
            Because of the very sensitive nature of shot noise subtraction 
            in the Jing de-aliasing, it may be better to normalize the counts of the 
            selection function to each map, and not to the mean of the maps.
            ''' 
            normsel = np.zeros((n_maps,ntracers))

            if verbose:
                print ("... done. Starting computations for each map (box) now.")
                print()
            else:
                pass

            #################################

            est_bias_fkp = np.zeros(ntracers)

            for nm in range(n_maps):
                time_start=time()
                if verbose:
                    print ('Loading simulated box #', nm)
                else:
                    pass
                h5map = h5py.File(mapnames_sims[nm],'r')
                maps = np.asarray(h5map.get(list(h5map.keys())[0]))
                if not maps.shape == (ntracers,n_x,n_y,n_z):
                    raise ValueError('Unexpected shape of simulated maps! Found:', maps.shape)
                h5map.close()

                if verbose:
                    print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))
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
                if verbose:
                    print ('  Estimating FKP power spectra...')
                else:
                    pass
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
                    if do_cross_spectra == True and ntracers > 1:
                        Cross0[nm] = fkp_many.cross_spec
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2
                else:
                    FKPmany = fkp_many.fkp((normsel[nm]*maps.T).T)
                    P0_fkp[nm] = fkp_many.P_ret
                    if do_cross_spectra == True and ntracers > 1:
                        Cross0[nm] = fkp_many.cross_spec
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2

                #CORRECT FOR BIN AVERAGING
                P0_fkp[nm] = P0_fkp[nm]/k_av_corr
                if do_cross_spectra == True and ntracers > 1:
                    Cross0[nm] = Cross0[nm]/k_av_corr
                else:
                    pass

                if nm==0:
                    # If data bias is different from mocks
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print ("  Effective biases of the simulated maps:")
                        print ("   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("        FKP=", ["%.3f"%b for b in est_bias_fkp])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
                else:
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print( "  Effective biases of these maps:")
                        print( "   Fiducial=", ["%.3f"%b for b in effbias])
                        print( "        FKP=", ["%.3f"%b for b in est_bias_fkp])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass

            #Update nbarbar to reflect actual
            nbarbar = nbarbar/np.mean(normsel,axis=0)

            # Correct missing factor of 2 in definition
            Theor_Cov_FKP = 2.0*np.mean(ThCov_fkp,axis=0)

            ################################################################################
            ################################################################################

            time_end=time()
            if verbose:
                print ('Total time cost for estimation of spectra: ', time_end - time_start)
            else:
                pass

            ################################################################################
            ################################################################################

            tempor=time()

            ################################################################################
            ################################################################################

            if verbose:
                print ('Applying mass assignement window function corrections...')
            else:
                pass

            ################################################################################
            #############################################################################
            # Mass assignement correction

            # For k smaller than the smallest k_phys, we use the Planck power spectrum.
            # For k larger than the highest k_phys value, we extrapolate using the power law at k_N/2 -- or the highest value of k_phys in the range, whichever is smaller
            kph_min = kph[0]
            kN = np.pi/cell_size  # Nyquist frequency

            # Compute the mean power spectra -- I will use the MTOE for the iteration
            
            P0_fkp_mean = np.mean(P0_fkp,axis=0)
            if do_cross_spectra == True and ntracers > 1:
                Cross0_mean = np.mean(Cross0,axis=0)
            else:
                pass

            # Cross0 and Cross2 are outputs of the FKP code, so they come out without the bias.
            # We can easily put back the bias by multiplying:
            # CrossX = cross_effbias**2 * CrossX
            if do_cross_spectra == True and ntracers > 1:
                cross_effbias = np.zeros(ntracers*(ntracers-1)//2)
                index=0
                for nt in range(ntracers):
                    for ntp in range(nt+1,ntracers):
                        cross_effbias[index] = np.sqrt(effbias[nt]*effbias[ntp])
                        index += 1
            else:
                pass

            # FKP and Cross measurements need to have the bias returned in their definitions
            P0_fkp = np.transpose((effbias**2*np.transpose(P0_fkp,axes=(0,2,1))),axes=(0,2,1))

            if do_cross_spectra == True and ntracers > 1:
                C0_fkp = np.transpose((cross_effbias**2*np.transpose(Cross0,axes=(0,2,1))),axes=(0,2,1))
            else:
                pass

            # Means
            P0_fkp_mean = np.mean(P0_fkp,axis=0)
            if do_cross_spectra == True and ntracers > 1:
                Cross0_mean = np.mean(C0_fkp,axis=0)
            else:
                pass

            # SAVE spectra
            P0_fkp_save=np.reshape(P0_fkp,(n_maps,ntracers*pow_bins))

            if do_cross_spectra == True and ntracers > 1:
                C0_fkp_save=np.reshape(C0_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
            else:
                pass

            if do_cross_spectra == True and ntracers > 1:
                data = [ kph ]
                columns = ['k']
                comb = list( combinations( [i for i in range(ntracers)], 2 ) )
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )
                    for k in range( len(comb) ):
                        print('monopole FKP', C0_fkp_save.shape)
                        data.append( C0_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS0_map{i}_tracers{comb[k][0]}{comb[k][1]}' )

                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)
            else:
                data = [ kph ]
                columns = ['k']
                comb = list( combinations( [i for i in range(ntracers)], 2 ) )
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )

                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)

        elif multipoles_order == 2:

            # Traditional (FKP) method
            P0_fkp = np.zeros((n_maps,ntracers,num_binsk))
            P2_fkp = np.zeros((n_maps,ntracers,num_binsk))

            # Cross spectra
            if do_cross_spectra == True and ntracers > 1:
                Cross0 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
                Cross2 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
            else:
                pass

            # Covariance
            ThCov_fkp = np.zeros((n_maps,ntracers,num_binsk))

            # Range where we estimate some parameters
            myk_min = np.argsort(np.abs(kph-0.1))[0]
            myk_max = np.argsort(np.abs(kph-0.2))[0]
            myran = np.arange(myk_min,myk_max)

            ##
            # UPDATED THIS TO NEW FKP CLASS WITH AUTO- AND CROSS-SPECTRA
            if verbose:
                print( "Initializing traditional (FKP) estimation toolbox...")
            else:
                pass
            fkp_many = fkp.fkp_init(num_binsk, n_bar_matrix_fid, effbias, cell_size, n_x_box, n_y_box, n_z_box, n_x_orig, n_y_orig, n_z_orig, MRk, powercentral,mas_power, multipoles_order, do_cross_spectra, ntracers, verbose)

            '''
            Because of the very sensitive nature of shot noise subtraction 
            in the Jing de-aliasing, it may be better to normalize the counts of the 
            selection function to each map, and not to the mean of the maps.
            ''' 
            normsel = np.zeros((n_maps,ntracers))

            if verbose:
                print ("... done. Starting computations for each map (box) now.")
                print()
            else:
                pass

            #################################

            est_bias_fkp = np.zeros(ntracers)

            for nm in range(n_maps):
                time_start=time()
                if verbose:
                    print ('Loading simulated box #', nm)
                else:
                    pass
                h5map = h5py.File(mapnames_sims[nm],'r')
                maps = np.asarray(h5map.get(list(h5map.keys())[0]))
                if not maps.shape == (ntracers,n_x,n_y,n_z):
                    raise ValueError('Unexpected shape of simulated maps! Found:', maps.shape)
                h5map.close()

                if verbose:
                    print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))
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
                if verbose:
                    print ('  Estimating FKP power spectra...')
                else:
                    pass
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
                    if do_cross_spectra == True and ntracers > 1:
                        Cross0[nm] = fkp_many.cross_spec
                        Cross2[nm] = fkp_many.cross_spec2
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2
                else:
                    FKPmany = fkp_many.fkp((normsel[nm]*maps.T).T)
                    P0_fkp[nm] = fkp_many.P_ret
                    P2_fkp[nm] = fkp_many.P2_ret
                    if do_cross_spectra == True and ntracers > 1:
                        Cross0[nm] = fkp_many.cross_spec
                        Cross2[nm] = fkp_many.cross_spec2
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2

                #CORRECT FOR BIN AVERAGING
                P0_fkp[nm] = P0_fkp[nm]/k_av_corr
                P2_fkp[nm] = P2_fkp[nm]/k_av_corr
                if do_cross_spectra == True and ntracers > 1:
                    Cross0[nm] = Cross0[nm]/k_av_corr
                    Cross2[nm] = Cross2[nm]/k_av_corr

                if nm==0:
                    # If data bias is different from mocks
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print ("  Effective biases of the simulated maps:")
                        print ("   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("        FKP=", ["%.3f"%b for b in est_bias_fkp])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
                else:
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print( "  Effective biases of these maps:")
                        print( "   Fiducial=", ["%.3f"%b for b in effbias])
                        print( "        FKP=", ["%.3f"%b for b in est_bias_fkp])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass

            #Update nbarbar to reflect actual
            nbarbar = nbarbar/np.mean(normsel,axis=0)

            # Correct missing factor of 2 in definition
            Theor_Cov_FKP = 2.0*np.mean(ThCov_fkp,axis=0)

            ################################################################################
            ################################################################################

            time_end=time()
            if verbose:
                print ('Total time cost for estimation of spectra: ', time_end - time_start)
            else:
                pass

            ################################################################################
            ################################################################################

            tempor=time()

            ################################################################################
            ################################################################################

            if verbose:
                print ('Applying mass assignement window function corrections...')
            else:
                pass

            ################################################################################
            #############################################################################
            # Mass assignement correction

            # For k smaller than the smallest k_phys, we use the Planck power spectrum.
            # For k larger than the highest k_phys value, we extrapolate using the power law at k_N/2 -- or the highest value of k_phys in the range, whichever is smaller
            kph_min = kph[0]
            kN = np.pi/cell_size  # Nyquist frequency

            # Compute the mean power spectra -- I will use the MTOE for the iteration

            P0_fkp_mean = np.mean(P0_fkp,axis=0)
            if do_cross_spectra == True and ntracers > 1:
                Cross0_mean = np.mean(Cross0,axis=0)
            else:
                pass
            
            # Cross0 and Cross2 are outputs of the FKP code, so they come out without the bias.
            # We can easily put back the bias by multiplying:
            # CrossX = cross_effbias**2 * CrossX
            if do_cross_spectra == True and ntracers > 1:
                cross_effbias = np.zeros(ntracers*(ntracers-1)//2)
                index=0
                for nt in range(ntracers):
                    for ntp in range(nt+1,ntracers):
                        cross_effbias[index] = np.sqrt(effbias[nt]*effbias[ntp])
                        index += 1

            # FKP and Cross measurements need to have the bias returned in their definitions
            P0_fkp = np.transpose((effbias**2*np.transpose(P0_fkp,axes=(0,2,1))),axes=(0,2,1))
            P2_fkp = np.transpose((effbias**2*np.transpose(P2_fkp,axes=(0,2,1))),axes=(0,2,1))

            if do_cross_spectra == True and ntracers > 1:
                C0_fkp = np.transpose((cross_effbias**2*np.transpose(Cross0,axes=(0,2,1))),axes=(0,2,1))
                C2_fkp = np.transpose((cross_effbias**2*np.transpose(Cross2,axes=(0,2,1))),axes=(0,2,1))
            else:
                pass

            # Means
            P0_fkp_mean = np.mean(P0_fkp,axis=0)
            P2_fkp_mean = np.mean(P2_fkp,axis=0)
            if do_cross_spectra == True and ntracers > 1:
                Cross0_mean = np.mean(C0_fkp,axis=0)
                Cross2_mean = np.mean(C2_fkp,axis=0)
            else:
                pass

            # SAVE spectra
            P0_fkp_save=np.reshape(P0_fkp,(n_maps,ntracers*pow_bins))

            P2_fkp_save=np.reshape(P2_fkp,(n_maps,ntracers*pow_bins))

            if do_cross_spectra == True and ntracers > 1:
                C0_fkp_save=np.reshape(C0_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
                C2_fkp_save=np.reshape(C2_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
            else:
                pass

            if do_cross_spectra == True and ntracers > 1:
                data = [ kph ]
                columns = ['k']
                comb = list( combinations( [i for i in range(ntracers)], 2 ) )
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )

                        data.append( P2_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_FKP_map{i}_tracer{j}' )
                    for k in range( len(comb) ):
                        data.append( C0_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS0_map{i}_tracers{comb[k][0]}{comb[k][1]}' )
                    for k in range( len(comb) ):
                        data.append( C2_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS2_map{i}_tracers{comb[k][0]}{comb[k][1]}' )

                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)
            else:
                data = [ kph ]
                columns = ['k']
                comb = list( combinations( [i for i in range(ntracers)], 2 ) )
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )

                        data.append( P2_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_FKP_map{i}_tracer{j}' )

                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)

        else:
            # Traditional (FKP) method
            P0_fkp = np.zeros((n_maps,ntracers,num_binsk))
            P2_fkp = np.zeros((n_maps,ntracers,num_binsk))
            P4_fkp = np.zeros((n_maps,ntracers,num_binsk))

            # Cross spectra
            if do_cross_spectra == True and ntracers > 1:
                Cross0 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
                Cross2 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
                Cross4 = np.zeros((n_maps,ntracers*(ntracers-1)//2,num_binsk))
            else:
                pass

            # Covariance
            ThCov_fkp = np.zeros((n_maps,ntracers,num_binsk))


            # Range where we estimate some parameters
            myk_min = np.argsort(np.abs(kph-0.1))[0]
            myk_max = np.argsort(np.abs(kph-0.2))[0]
            myran = np.arange(myk_min,myk_max)

            ##
            # UPDATED THIS TO NEW FKP CLASS WITH AUTO- AND CROSS-SPECTRA
            if verbose:
                print( "Initializing traditional (FKP) estimation toolbox...")
            else:
                pass
            fkp_many = fkp.fkp_init(num_binsk, n_bar_matrix_fid, effbias, cell_size, n_x_box, n_y_box, n_z_box, n_x_orig, n_y_orig, n_z_orig, MRk, powercentral,mas_power, multipoles_order, do_cross_spectra, ntracers, verbose)

            '''
            Because of the very sensitive nature of shot noise subtraction 
            in the Jing de-aliasing, it may be better to normalize the counts of the 
            selection function to each map, and not to the mean of the maps.
            ''' 
            normsel = np.zeros((n_maps,ntracers))

            if verbose:
                print ("... done. Starting computations for each map (box) now.")
                print()
            else:
                pass

            #################################

            est_bias_fkp = np.zeros(ntracers)

            for nm in range(n_maps):
                time_start=time()
                if verbose:
                    print ('Loading simulated box #', nm)
                else:
                    pass
                h5map = h5py.File(mapnames_sims[nm],'r')
                maps = np.asarray(h5map.get(list(h5map.keys())[0]))
                if not maps.shape == (ntracers,n_x,n_y,n_z):
                    raise ValueError('Unexpected shape of simulated maps! Found:', maps.shape)
                h5map.close()

                if verbose:
                    print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))
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
                if verbose:
                    print ('  Estimating FKP power spectra...')
                else:
                    pass
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
                    if do_cross_spectra == True and ntracers > 1:
                        Cross0[nm] = fkp_many.cross_spec
                        Cross2[nm] = fkp_many.cross_spec2
                        Cross4[nm] = fkp_many.cross_spec4
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2
                else:
                    FKPmany = fkp_many.fkp((normsel[nm]*maps.T).T)
                    P0_fkp[nm] = fkp_many.P_ret
                    P2_fkp[nm] = fkp_many.P2_ret
                    if do_cross_spectra == True and ntracers > 1:
                        Cross4[nm] = fkp_many.cross_spec4
                        Cross0[nm] = fkp_many.cross_spec
                        Cross2[nm] = fkp_many.cross_spec2
                        Cross4[nm] = fkp_many.cross_spec4
                    else:
                        pass
                    ThCov_fkp[nm] = (fkp_many.sigma)**2

                #CORRECT FOR BIN AVERAGING
                P0_fkp[nm] = P0_fkp[nm]/k_av_corr
                P2_fkp[nm] = P2_fkp[nm]/k_av_corr
                P4_fkp[nm] = P4_fkp[nm]/k_av_corr
                if do_cross_spectra == True and ntracers > 1:
                    Cross0[nm] = Cross0[nm]/k_av_corr
                    Cross2[nm] = Cross2[nm]/k_av_corr
                    Cross4[nm] = Cross4[nm]/k_av_corr
                else:
                    pass

                if nm==0:
                    # If data bias is different from mocks
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print ("  Effective biases of the simulated maps:")
                        print ("   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("        FKP=", ["%.3f"%b for b in est_bias_fkp])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
                else:
                    est_bias_fkp = np.sqrt(np.mean(effbias**2*(P0_fkp[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print( "  Effective biases of these maps:")
                        print( "   Fiducial=", ["%.3f"%b for b in effbias])
                        print( "        FKP=", ["%.3f"%b for b in est_bias_fkp])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
    
            #Update nbarbar to reflect actual
            nbarbar = nbarbar/np.mean(normsel,axis=0)

            # Correct missing factor of 2 in definition
            Theor_Cov_FKP = 2.0*np.mean(ThCov_fkp,axis=0)

            ################################################################################
            ################################################################################

            time_end=time()
            if verbose:
                print ('Total time cost for estimation of spectra: ', time_end - time_start)
            else:
                pass

            ################################################################################
            ################################################################################

            tempor=time()

            ################################################################################
            ################################################################################

            if verbose:
                print ('Applying mass assignement window function corrections...')
            else:
                pass

            ################################################################################
            #############################################################################
            # Mass assignement correction

            # For k smaller than the smallest k_phys, we use the Planck power spectrum.
            # For k larger than the highest k_phys value, we extrapolate using the power law at k_N/2 -- or the highest value of k_phys in the range, whichever is smaller
            kph_min = kph[0]
            kN = np.pi/cell_size  # Nyquist frequency

            # Compute the mean power spectra -- I will use the MTOE for the iteration

            P0_fkp_mean = np.mean(P0_fkp,axis=0)
            if do_cross_spectra == True and ntracers > 1:
                Cross0_mean = np.mean(Cross0,axis=0)
            else:
                pass

            # Cross0 and Cross2 are outputs of the FKP code, so they come out without the bias.
            # We can easily put back the bias by multiplying:
            # CrossX = cross_effbias**2 * CrossX
            if do_cross_spectra == True and ntracers > 1:
                cross_effbias = np.zeros(ntracers*(ntracers-1)//2)
                index=0
                for nt in range(ntracers):
                    for ntp in range(nt+1,ntracers):
                        cross_effbias[index] = np.sqrt(effbias[nt]*effbias[ntp])
                        index += 1
            else:
                pass

            # FKP and Cross measurements need to have the bias returned in their definitions
            P0_fkp = np.transpose((effbias**2*np.transpose(P0_fkp,axes=(0,2,1))),axes=(0,2,1))
            P2_fkp = np.transpose((effbias**2*np.transpose(P2_fkp,axes=(0,2,1))),axes=(0,2,1))
            P4_fkp = np.transpose((effbias**2*np.transpose(P4_fkp,axes=(0,2,1))),axes=(0,2,1))

            if do_cross_spectra == True and ntracers > 1:
                C0_fkp = np.transpose((cross_effbias**2*np.transpose(Cross0,axes=(0,2,1))),axes=(0,2,1))
                C2_fkp = np.transpose((cross_effbias**2*np.transpose(Cross2,axes=(0,2,1))),axes=(0,2,1))
                C4_fkp = np.transpose((cross_effbias**2*np.transpose(Cross4,axes=(0,2,1))),axes=(0,2,1))
            else:
                pass
                
            # Means
            P0_fkp_mean = np.mean(P0_fkp,axis=0)
            P2_fkp_mean = np.mean(P2_fkp,axis=0)
            P4_fkp_mean = np.mean(P4_fkp,axis=0)
            if do_cross_spectra == True and ntracers > 1:
                Cross0_mean = np.mean(C0_fkp,axis=0)
                Cross2_mean = np.mean(C2_fkp,axis=0)
                Cross4_mean = np.mean(C4_fkp,axis=0)
            else:
                pass

            # SAVE spectra
            P0_fkp_save=np.reshape(P0_fkp,(n_maps,ntracers*pow_bins))
            P2_fkp_save=np.reshape(P2_fkp,(n_maps,ntracers*pow_bins))
            P4_fkp_save=np.reshape(P4_fkp,(n_maps,ntracers*pow_bins))

            if do_cross_spectra == True and ntracers > 1:
                C0_fkp_save=np.reshape(C0_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
                C2_fkp_save=np.reshape(C2_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
                C4_fkp_save=np.reshape(C4_fkp,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
            else:
                pass

            if do_cross_spectra == True and ntracers > 1:
                data = [ kph ]
                columns = ['k']
                comb = list( combinations( [i for i in range(ntracers)], 2 ) )
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )

                        data.append( P2_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_FKP_map{i}_tracer{j}' )

                        data.append( P4_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P4_FKP_map{i}_tracer{j}' )
                    for k in range( len(comb) ):
                        data.append( C0_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS0_map{i}_tracers{comb[k][0]}{comb[k][1]}' )
                    for k in range( len(comb) ):
                        data.append( C2_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS2_map{i}_tracers{comb[k][0]}{comb[k][1]}' )
                    for k in range( len(comb) ):
                        data.append( C4_fkp_save[i, k*(kph.shape[0]):(k+1)*(kph.shape[0])] )
                        columns.append( f'CROSS4_map{i}_tracers{comb[k][0]}{comb[k][1]}' )

                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)

            else:
                data = [ kph ]
                columns = ['k']
                comb = list( combinations( [i for i in range(ntracers)], 2 ) )
                for i in range(n_maps):
                    for j in range(ntracers):
                        data.append( P0_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P0_FKP_map{i}_tracer{j}' )

                        data.append( P2_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P2_FKP_map{i}_tracer{j}' )

                        data.append( P4_fkp_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                        columns.append( f'P4_FKP_map{i}_tracer{j}' )

                data = np.array(data).T
                df = pd.DataFrame(data = data, columns = columns)
                df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)

    elif method == 'MT':

        if multipoles_order == 0:

            # Multitracer method: monopole, quadrupole, th. covariance(FKP-like)
            # Original (convolved) spectra:
            P0_data = np.zeros((n_maps,ntracers,num_binsk))

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

            if verbose:
                print( "Initializing multi-tracer estimation toolbox...")
            else:
                pass

            fkp_mult = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,effbias_mt,cell_size,n_x_box,n_y_box,n_z_box,n_x_orig,n_y_orig,n_z_orig,MRk,powercentral,mas_power, multipoles_order, verbose)

            '''
            Because of the very sensitive nature of shot noise subtraction 
            in the Jing de-aliasing, it may be better to normalize the counts of the 
            selection function to each map, and not to the mean of the maps.
            ''' 
            normsel = np.zeros((n_maps,ntracers))

            if verbose:
                print ("... done. Starting computations for each map (box) now.")
                print()
            else:
                pass

            #################################

            est_bias_mt = np.zeros(ntracers)

            for nm in range(n_maps):
                time_start=time()
                if verbose:
                    print ('Loading simulated box #', nm)
                else:
                    pass
                h5map = h5py.File(mapnames_sims[nm],'r')
                maps = np.asarray(h5map.get(list(h5map.keys())[0]))
                if not maps.shape == (ntracers,n_x,n_y,n_z):
                    raise ValueError('Unexpected shape of simulated maps! Found:', maps.shape)
                h5map.close()

                if verbose:
                    print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))
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
                if verbose:
                    print ('  Now estimating multi-tracer spectra...')
                else:
                    pass
                if nm==0:
                    FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
                    P0_data[nm] = fkp_mult.P0_mu_ret
                else:
                    FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
                    P0_data[nm] = fkp_mult.P0_mu_ret

                #CORRECT FOR BIN AVERAGING
                P0_data[nm] = P0_data[nm]/k_av_corr

                if nm==0:
                    # If data bias is different from mocks
                    est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print ("  Effective biases of the simulated maps:")
                        print ("   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
                else:
                    est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print( "  Effective biases of these maps:")
                        print( "   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
    
            #Update nbarbar to reflect actual
            nbarbar = nbarbar/np.mean(normsel,axis=0)

            ################################################################################
            ################################################################################

            time_end=time()
            if verbose:
                print ('Total time cost for estimation of spectra: ', time_end - time_start)
            else:
                pass

            ################################################################################
            ################################################################################

            tempor=time()

            ################################################################################
            ################################################################################

            if verbose:
                print ('Applying mass assignement window function corrections...')
            else:
                pass

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

            # SAVE spectra
            P0_save=np.reshape(P0_data,(n_maps,ntracers*pow_bins))

            data = [ kph ]
            columns = ['k']
            for i in range(n_maps):
                for j in range(ntracers):
                    data.append( P0_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                    columns.append( f'P0_MT_map{i}_tracer{j}' )
            data = np.array(data).T
            df = pd.DataFrame(data = data, columns = columns)
            df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)


        elif multipoles_order == 2:

            # Multitracer method: monopole, quadrupole, th. covariance(FKP-like)
            # Original (convolved) spectra:
            P0_data = np.zeros((n_maps,ntracers,num_binsk))
            P2_data = np.zeros((n_maps,ntracers,num_binsk))

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

            if verbose:
                print( "Initializing multi-tracer estimation toolbox...")
            else:
                pass

            fkp_mult = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,effbias_mt,cell_size,n_x_box,n_y_box,n_z_box,n_x_orig,n_y_orig,n_z_orig,MRk,powercentral,mas_power, multipoles_order, verbose)

            '''
            Because of the very sensitive nature of shot noise subtraction 
            in the Jing de-aliasing, it may be better to normalize the counts of the 
            selection function to each map, and not to the mean of the maps.
            ''' 
            normsel = np.zeros((n_maps,ntracers))

            if verbose:
                print ("... done. Starting computations for each map (box) now.")
                print()
            else:
                pass

            #################################

            est_bias_mt = np.zeros(ntracers)

            for nm in range(n_maps):
                time_start=time()
                if verbose:
                    print ('Loading simulated box #', nm)
                else:
                    pass
                h5map = h5py.File(mapnames_sims[nm],'r')
                maps = np.asarray(h5map.get(list(h5map.keys())[0]))
                if not maps.shape == (ntracers,n_x,n_y,n_z):
                    raise ValueError('Unexpected shape of simulated maps! Found:', maps.shape)
                h5map.close()

                if verbose:
                    print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))
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
                if verbose:
                    print ('  Now estimating multi-tracer spectra...')
                else:
                    pass
                if nm==0:
                    FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
                    P0_data[nm] = fkp_mult.P0_mu_ret
                    P2_data[nm] = fkp_mult.P2_mu_ret
                else:
                    FKPmult = fkp_mult.fkp((normsel[nm]*maps.T).T)
                    P0_data[nm] = fkp_mult.P0_mu_ret
                    P2_data[nm] = fkp_mult.P2_mu_ret

                #CORRECT FOR BIN AVERAGING
                P0_data[nm] = P0_data[nm]/k_av_corr
                P2_data[nm] = P2_data[nm]/k_av_corr

                if nm==0:
                    # If data bias is different from mocks
                    est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print ("  Effective biases of the simulated maps:")
                        print ("   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
                else:
                    est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print( "  Effective biases of these maps:")
                        print( "   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass

            #Update nbarbar to reflect actual
            nbarbar = nbarbar/np.mean(normsel,axis=0)

            ################################################################################
            ################################################################################

            time_end=time()
            if verbose:
                print ('Total time cost for estimation of spectra: ', time_end - time_start)
            else:
                pass

            ################################################################################
            ################################################################################

            tempor=time()

            ################################################################################
            ################################################################################

            if verbose:
                print ('Applying mass assignement window function corrections...')
            else:
                pass

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

            # SAVE spectra
            P0_save=np.reshape(P0_data,(n_maps,ntracers*pow_bins))
            P2_save=np.reshape(P2_data,(n_maps,ntracers*pow_bins))

            data = [ kph ]
            columns = ['k']
            for i in range(n_maps):
                for j in range(ntracers):
                    data.append( P0_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                    columns.append( f'P0_MT_map{i}_tracer{j}' )

                    data.append( P2_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                    columns.append( f'P2_MT_map{i}_tracer{j}' )

            data = np.array(data).T
            df = pd.DataFrame(data = data, columns = columns)
            df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)

        else:

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

            if verbose:
                print( "Initializing multi-tracer estimation toolbox...")
            else:
                pass

            fkp_mult = fkpmt.fkp_init(num_binsk,n_bar_matrix_fid,effbias_mt,cell_size,n_x_box,n_y_box,n_z_box,n_x_orig,n_y_orig,n_z_orig,MRk,powercentral,mas_power, multipoles_order, verbose)

            '''
            Because of the very sensitive nature of shot noise subtraction 
            in the Jing de-aliasing, it may be better to normalize the counts of the 
            selection function to each map, and not to the mean of the maps.
            ''' 
            normsel = np.zeros((n_maps,ntracers))

            if verbose:
                print ("... done. Starting computations for each map (box) now.")
                print()
            else:
                pass

            #################################

            est_bias_mt = np.zeros(ntracers)

            for nm in range(n_maps):
                time_start=time()
                if verbose:
                    print ('Loading simulated box #', nm)
                else:
                    pass
                h5map = h5py.File(mapnames_sims[nm],'r')
                maps = np.asarray(h5map.get(list(h5map.keys())[0]))
                if not maps.shape == (ntracers,n_x,n_y,n_z):
                    raise ValueError('Unexpected shape of simulated maps! Found:', maps.shape)
                h5map.close()

                if verbose:
                    print( "Total number of objects in this map:", np.sum(maps,axis=(1,2,3)))
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
                if verbose:
                    print ('  Now estimating multi-tracer spectra...')
                else:
                    pass
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
                    if verbose:
                        print ("  Effective biases of the simulated maps:")
                        print ("   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
                else:
                    est_bias_mt = np.sqrt(np.mean((P0_data[nm]/powtrue).T [myran],axis=0))
                    if verbose:
                        print( "  Effective biases of these maps:")
                        print( "   Fiducial=", ["%.3f"%b for b in effbias])
                        print ("         MT=", ["%.3f"%b for b in est_bias_mt])
                    else:
                        pass
                    dt = time() - time_start
                    if verbose:
                        print ("Elapsed time for computation of spectra for this map:", np.around(dt,4))
                        print(".")
                    else:
                        pass
    
            #Update nbarbar to reflect actual
            nbarbar = nbarbar/np.mean(normsel,axis=0)

            ################################################################################
            ################################################################################


            time_end=time()
            if verbose:
                print ('Total time cost for estimation of spectra: ', time_end - time_start)
            else:
                pass

            ################################################################################
            ################################################################################

            tempor=time()

            ################################################################################
            ################################################################################

            if verbose:
                print ('Applying mass assignement window function corrections...')
            else:
                pass

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

            # SAVE spectra
            P0_save=np.reshape(P0_data,(n_maps,ntracers*pow_bins))
            P2_save=np.reshape(P2_data,(n_maps,ntracers*pow_bins))
            P4_save=np.reshape(P4_data,(n_maps,ntracers*pow_bins))

            data = [ kph ]
            columns = ['k']
            for i in range(n_maps):
                for j in range(ntracers):
                    data.append( P0_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                    columns.append( f'P0_MT_map{i}_tracer{j}' )

                    data.append( P2_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                    columns.append( f'P2_MT_map{i}_tracer{j}' )

                    data.append( P4_save[i, j*(kph.shape[0]):(j+1)*(kph.shape[0])] )
                    columns.append( f'P4_MT_map{i}_tracer{j}' )

            data = np.array(data).T
            df = pd.DataFrame(data = data, columns = columns)
            df.to_csv(dir_specs + handle_estimates + '-spectra.csv', index = False)

    return print('Done!')
