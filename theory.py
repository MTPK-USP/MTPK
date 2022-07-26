import numpy as np
import camb
import time
import os
import sys
import scipy.integrate

c_light = 299792.458 #km/s

camb_dir = "CAMB"

def power_spectrum(cosmo, whichspec, redshifts, k_min, k_max):
    '''
    Method to compute the matter power-spectrum using CAMB

        Parameters
        ----------
        cosmo : object
            Object containing a predefined cosmology
        
        whichspec : str
            String indicating which spectrum we wish to obtain.
            'Linear'           -> Linear power-spectrum
            'Halofit-Mead'     -> Halofit implementation by A. Mead
            'Halofit-Casarini' -> Halofit implementation by L. Casarini
        
        redshifts : array of floats
            Array containing different redshifts 
        
        k_min : float
            Minimum wave-vector for which we want the spectrum to be computed

        k_max : float 
            Maximum wave-vector for which we want the spectrum to be computed

        Returns
        -------
        
        Pk_dict : dictionary
            Dictionay containing k(h/Mpc),  Pk(Mpc/h)^3, r_s_drag (Mpc) and sigma_8 

            The dictionary's
            Structure is as follows
            Pk_dict = { 'k'        : k_array,
                        'Pk_z1'    : Pk_at_redshift_z1,
                        ...
                        'Pk_zN'    : Pk_at_redshift_zN,
                        'r_s_drag' : r_s_drag_value,
                        'sigma_8'  : sigma_8 value
                       }
                                
    '''
    this_dir = os.getcwd()

    if type(redshifts) == float:
        redshifts = np.asarray([redshifts])
    else:
        redshifts = np.asarray(redshifts)
    
    n_redshifts = len(redshifts)
    
    sort_redshifts = np.argsort(redshifts)[::-1]
    ord_redshifts = redshifts[sort_redshifts]

    # Array of strings with redshifts ordered from high to low
    str_redshifts = (ord_redshifts).astype('str').tolist()

    # Array of file names for transfer functions and power_spectra
    transf_filenames=[]
    matpow_filenames=[]
    for iz in range(n_redshifts):
        transf_filenames.append("transf_" + str_redshifts[iz] + ".dat")
        matpow_filenames.append("matpow_" + str_redshifts[iz] + ".dat")
   
    ks = []
    ps = []

    if(k_min >=1e-4):
        t0 = time.time()
        
        os.chdir(camb_dir)

        # Write the given parameters to the input parameter file
        params = open("params.ini")
        temp = params.readlines()

        temp[3]  = 'output_root     = mtpk \n'

        temp[34] = 'ombh2           = ' + str(cosmo.Omega0_b*cosmo.h**2) + '\n'
        temp[35] = 'omch2           = ' + str(cosmo.Omega0_cdm*cosmo.h**2) + '\n'
        temp[37] = 'omk             = ' + str(cosmo.Omega0_k) + '\n'
        temp[38] = 'hubble          = ' + str(100*cosmo.h) + '\n'
        # Dark Energy Parameters
        temp[41] = 'w               = ' + str(cosmo.w0) + '\n'
        temp[46] = 'wa              = ' + str(cosmo.w1) + '\n'
        # Primordial Spectrum Parameters
        temp[85] = 'scalar_amp(1)   = ' + str(cosmo.A_s) + '\n'
        temp[86] = 'scalar_spectral_index(1) = ' + str(cosmo.n_s) + '\n'
        # Reionization Parameters
        temp[107] = 're_use_optical_depth = F \n'
        temp[110] = 're_redshift         = ' + str(cosmo.z_re) + '\n'
        
        temp[252] = 'transfer_kmax        = ' + str(k_max) + '\n'
        temp[254] = 'transfer_num_redshifts = ' + str(n_redshifts) + '\n'
        
        temp[256] = 'transfer_redshift(1) = ' + str_redshifts[0] + '\n'
        temp[257] = 'transfer_filename(1) = ' + transf_filenames[0] + '\n'
        temp[259] = 'transfer_matterpower(1) = ' + matpow_filenames[0] + '\n'

        for iz in range(1, n_redshifts):
            red_string = "transfer_redshift(" + str(iz+1) + ")   =  " + str_redshifts[iz]  + '\n'
            temp.append(red_string)
            transf_string = "transfer_filename(" + str(iz+1) + ")   =  " + transf_filenames[iz]  + '\n'
            temp.append(transf_string) 
            matpow_string = "transfer_matterpower(" + str(iz+1) + ")   =  " + matpow_filenames[iz]  + '\n'
            temp.append(matpow_string)

        out_name = 'params_tempz.ini'

        out = open(out_name, 'w')
        for r in temp:
            out.write(r)
        out.close()

        t1 = time.time()
        #print("Time elapsed for I/O:",t1-t0)

        t0 = time.time()

        if(whichspec == 0):
            ########### Linear Spectrum #################
            os.system('camb ' + out_name + ' > camb_out.txt')
            os.chdir(this_dir)

        elif(whichspec == 1):
            ########## HaloFit mead version ##############
            params = open('params_tempz.ini')
            temp = params.readlines()

            temp[17] = 'do_nonlinear = 1 \n'

            temp.append('halofit_version = 5 \n')

            out = open('params_tempz.ini','w')
            for r in temp:
                out.write(r)
            out.close()

            os.system('camb ' + out_name + ' > camb_out.txt')

            t1 = time.time()
            print("Time elapsed for CAMB + I/O:",t1-t0)
            ###################################################

            os.chdir(this_dir)

        else:
            ######### HaloFit casarini version ################
            params = open('params_tempz.ini')
            temp = params.readlines()

            temp[17] = 'do_nonlinear = 1 \n'

            temp.append('halofit_version = 7 \n')

            out = open('params_tempz.ini','w')
            for r in temp:
                out.write(r)
            out.close()

            os.system('camb ' + out_name + ' > camb_out.txt')
            ####################################################

            #t1 = time.time()
            # print("Time elapsed:",t1-t0)

            os.chdir(this_dir)
        
        for iz in range(n_redshifts):
            spec = np.loadtxt(camb_dir+'/mtpk_'+matpow_filenames[iz] )
            key = 'Pk_' + str_redshifts[iz]
            if(iz==0):
                Pk_dict = {'k' : spec[:,0]}
                Pk_dict[key] = spec[:,1]
            else:
                Pk_dict[key] = spec[:,1]

            #############################################

        with open(camb_dir+'/camb_out.txt', 'r') as f:
            lines = f.readlines()

            for i in range(len(lines)):
                if(lines[i].split()[0] == "r_s(zdrag)/Mpc" ):
                    r_s_drag = float(lines[i].split()[-1])
                    sigma8   = float(lines[22].split()[-1])

        Pk_dict['r_s_drag'] = r_s_drag
        Pk_dict['sigma_8']  = sigma8

        return Pk_dict
	
    else:
        print('Warning! Minimum k value too small, code will not work')
        sys.exit(-1)

def rsd_params( **kwargs):
    '''
    Method to organize RSD parameters inside a dictionary from which they can be easily accessed

        Parameters
        ----------
        b1 : float
            Linear bias

        sigma_tot : float
            Adimensional parameter representing redshift error at redshift z=0.

        Returns
        -------

        Python dictionary containing the parameters above
    '''

    default_params = {
            'b1'         : np.asarray([1]),
            'sigma_tot'  : np.sqrt(4.7e-4**2 + (150/c_light)**2)
            }
    
    for key, value in kwargs.items():
        if key not in default_params.keys():
            print("WARNING: {} is not a valid keyword".format(key) )
            continue
        default_params[key] = value

    return default_params

def pk_multipoles_gauss(rsd_params, cosmo, redshifts, whichspec, kmin, kmax, Nk, **kwargs):
    '''
    Method to compute the monopole and quadrupole of the redshift-space power-spectrum

        Parameters
        ----------

        cosmo : cosmology object
            Object containing a predefined cosmology
        
        rsd_params : dictionary
            Dictionary containing the main RSD parameters
        
        redsfhits : float list
            List containing the redshifts for which we want the multipoles

        whichspec : integer
            Integer controlling which kind of spectrum we want:
                0 -> linear
                1 -> Halofit Mead version
                2 -> Halofit Casarini version

        kmin : float
            Minimum wave-vector (h/Mpc)

        kmax : float
            Maximum wave-vector (h/Mpc)

        Nk : int
            Number of k bins

        k : float array, optional
            If you wish to, you can provide a vector containing the k
            values. If you choose to do that, kmin, kmax and Nk will be ignored

        matgrowrate : float, optional
            Optional parameter to pass a value of matgrowrate.
            If this is not given, will use cosmological parameters
            to obtain f through f(z) = Omega_m(z)^gamma

        Returns
        -------

        multipoles_dict : dictionary
            This dictionary has the following structure:
            multipoles_dict = { 
                'mono' : { 0 : M0_array_1,
                           ...,
                           N : M0_array_N
                         }
                'quad' : { 0 : M2_array_1,
                           ...,
                           N : M2_array_N
                         }
                }
            The indexes 0, 1, 2 index the different tracers, and Ml_array_i
            are the factors which connect real-space power spectrum and
            redshift-space multipoles.

    '''

    import os, ctypes
    from scipy import integrate, LowLevelCallable

    biases          = rsd_params['b1']
    try:
        len(biases) # If we have only one value, this will issue an error
    except:
        biases = np.asarray([biases])
    for key, value in kwargs.items():
        if(key=='matgrowrate'):
            matgrowrate=value
        if(key=='k'):
            kphys=value

    try:
        kphys
    except:
        kphys = np.logspace(np.log10(kmin), np.log10(kmax), Nk)

    try:
        len(redshifts)
    except:
        redshifts = np.asarray([redshifts])
    try:
        matgrowrate
    except:
        matgrowrate = cosmo.f_evolving(redshifts)
    try:
        len(matgrowrate)
    except:
        matgrowrate = np.asarray([matgrowrate])

    sig_tot         = rsd_params['sigma_tot']
    sig_tot         = sig_tot / cosmo.H(redshifts, True)
    #cH              = c_light / cosmo.H(redshifts, True)

    # Let's try to do it numerically 

    M_num = np.zeros((len(redshifts), len(biases), len(kphys) ))
    Q_num = np.zeros((len(redshifts), len(biases), len(kphys) ))
    H_num = np.zeros((len(redshifts), len(biases), len(kphys) ))

    lib = ctypes.CDLL(os.path.abspath('spec_model.so'))
    lib.P.restype = ctypes.c_double
    lib.P.argtypes = (ctypes.c_int, ctypes.POINTER(ctypes.c_double))

    func = LowLevelCallable(lib.P)

    for zz in range(len(redshifts)):
        for bb in range(len(biases)):
            for kk in range(len(kphys)):

                M_num[zz][bb][kk] = ((2*0+1)/2)*scipy.integrate.nquad(func, [[-1, 1]], args=([kphys[kk], biases[bb], matgrowrate[zz], sig_tot[bb], 0.]))[0]
                Q_num[zz][bb][kk] = ((2*2+1)/2)*scipy.integrate.nquad(func, [[-1, 1]], args=([kphys[kk], biases[bb], matgrowrate[zz], sig_tot[bb], 2.]))[0]
                H_num[zz][bb][kk] = ((2*4+1)/2)*scipy.integrate.nquad(func, [[-1, 1]], args=([kphys[kk], biases[bb], matgrowrate[zz], sig_tot[bb], 4.]))[0]
                
    # This is to regularize the behaviour of the functions at k -> 0
    small = 0.005

    # # k * sigma_z
    KZ = np.zeros( (len(redshifts), len(biases), len(kphys)) )
        
    M_dict = {'mono' : np.zeros(KZ.shape),
              'quad' : np.zeros(KZ.shape),
              'hexa' : np.zeros(KZ.shape)}

    for i in range(len(redshifts)):
        for j in range(len(biases)):
            M_dict['mono'][i][j] = M_num[i][j]
            M_dict['quad'][i][j] = Q_num[i][j]
            M_dict['hexa'][i][j] = H_num[i][j]

    return M_dict


def q_ell(random, cosmo, **kwargs):
    '''
    Computes window function multipoles in real space, using a pair-counting approach.

        We'll take a random catalogue and compute the random-random pairs weighted by the 
        Legendre polynomial of order \ell evaluated on the angle between the two objects, 
        as seen by the observer.
        
        Parameters
        ----------

        random : array of floats
            Array containing the random catalogue. This should have the shape
            (N, 3) in which N is the number of random points and the columns store RA, 
            DEC, z respectively.        

        cosmo : dictionary
            Contains cosmological parameters which can be accessed through keywords

        ell_max : int
            Maximum multipole we wish to compute
        
        rmin : float
            Value of the minimum r for which we'll compute Q_ell(r)

        rmax : float
            Value of the maximum r for which we'll compute Q_ell(r)

        Nr : int
            Number of points in r to be computed
        
        Nmu : int
            Number of bins in mu

        all_multipoles : bool
            Boolean variable to control whether the user wants all the multipoles or only
            the even ones

        zmin : float
            Minimum redshift we wish to consider in the random catalogue
        
        zmax : float
            Maximum redshift we wish to consider in the random catalogue

        mu_max : float
            Maximum value of mu we wish to consider. Should be kept to 1 unless you have
            strong reasons to change it

        autocorr : int
            0 or 1. Controls whether we're computing auto or cross correlations

        N_cores : int 
            Number of cores available in your computer
        
        fraction : float
            Between 0 and 1. This is the fraction of the random catalogue you want to use
            for your computation. Sometimes the random cataloge is huge, and using it all
            leads to very time consuming pair counting
        
        FKP_weights : bool
            Whether to include the FKP weights or not when computing window function

        n_bar_r : str
           Name of the file containing the radial number density of objects in the survey.
           This should have shape (n,2) in which the first column contains values of comoving distance
           and the second contains the number density in units of (Mpc/h)^{-3}. Useful for
           computing FKP and MTOE weights to be included in window function

        P_eff : float
            Value of the power spectrum in units of (Mpc/h)^3 at a chosen scale k_eff. Useful
            for computing FKP and MTOE weights to be included in window function

        Returns
        -------
        
        r_centers : float array
            An array of length N_r containing the center values for the bins in r

        q_ells : float array
            An array of shape (N_ell, N_r) containing all the q_ell requested. N_ell is the 
            number of multipoles and N_r is the number points in r
        
    '''
    import numpy as np
    from Corrfunc.mocks.DDsmu_mocks import DDsmu_mocks
    from coord_transform import coord_transform
    import time
    import scipy.special

    default_kwargs = {
            'ell_max'           : 4,
            'rmin'              : 0.1,
            'rmax'              : 150,
            'Nr'                : 20,
            'Nmu'               : 1000,
            'all_multipoles'    : False,
            'zmin'              : 0,
            'zmax'              : 3,
            'mu_max'            : 1,
            'autocorr'          : 1,
            'N_cores'           : 12,
            'fraction'          : 1,
            'FKP_weights'       : False,
            'n_bar_r'           : None,
            'P_eff'             : 1e4,
            'Normalize'         : True
            }

    # Get the optional keyword arguments 
    for key, value in kwargs.items():
        if key not in default_kwargs.keys():
            continue
        default_kwargs[key] = value
    
    # Check if paramters were provided correctly
    if(isinstance(default_kwargs['ell_max'], int) == False):
        print("ERROR: ell_max has to be an integer!")
        return -1

    if( default_kwargs['fraction'] > 1 or default_kwargs['fraction'] <=0 ):
        print("ERROR: fraction should be a number between 0 and 1, {} is not a valid entry ".format(fraction) )
        return -1

    ell_max        = default_kwargs['ell_max']
    rmin           = default_kwargs['rmin']
    rmax           = default_kwargs['rmax']
    Nr             = default_kwargs['Nr']
    Nmu            = default_kwargs['Nmu']
    all_multipoles = default_kwargs['all_multipoles']
    zmin           = default_kwargs['zmin']
    zmax           = default_kwargs['zmax']
    mu_max         = default_kwargs['mu_max']
    autocorr       = default_kwargs['autocorr']
    N_cores        = default_kwargs['N_cores']
    fraction       = default_kwargs['fraction']
    FKP_weights    = default_kwargs['FKP_weights']
    n_bar_r        = default_kwargs['n_bar_r']
    P_eff          = default_kwargs['P_eff']
    Normalize      = default_kwargs['Normalize']

    comov = np.vectorize(cosmo.comoving)
    z_interp = np.linspace(zmin,zmax,2000)#np.linspace(0.3,1.3,2000)
    d_interp = comov(z_interp, True)
    comov = scipy.interpolate.interp1d(z_interp, d_interp)
    inv_comov = scipy.interpolate.interp1d(d_interp, z_interp)

    rz_min = comov(zmin)
    rz_max = comov(zmax)

    '''
        LOAD N_BAR_R
    '''
    if(FKP_weights):
        print("Loading n_bar")
        n_bar = np.loadtxt(n_bar_r)
        print("Done!") 

        n_bar = n_bar[np.where( (n_bar[:,2]>cosmo.comoving(zmin,True)) & (n_bar[:,2]<cosmo.comoving(zmax,True)) )]

    '''
        LOAD RANDOM CATALOGUE AND PROCESS IT
    '''

    rand_cat = random

    if ( np.abs(rand_cat[:,2]).max() > 100 ):
        print("\n Third column is comoving distance \n")
        rand_cat[np.where( (rand_cat[:,2]>=rz_min) & (rand_cat[:,2]<=rz_max) )]

    else:
        rand_cat = rand_cat[np.where( (rand_cat[:,2]>=zmin) & (rand_cat[:,2]<=zmax) )]
        # Transforming the third axis into comoving distance
        rand_cat[:,2] = comov(rand_cat[:,2])


    print("Considering all redshifts from {} to {}".format(zmin, zmax) )
    
    # Remove a fraction of it, as requested by the user
    npoints = len(rand_cat[:,2])
    rem_list = np.random.choice(range(npoints), int(npoints*fraction), replace=False )
    r_vec = rand_cat[rem_list,2]
    
    if(FKP_weights):
        print("Including FKP weights")
        n_bar_interp = np.interp(r_vec, n_bar[:,2], n_bar[:,4])
        weights_FKP = 1/(1 + P_eff*n_bar_interp)
    else:
        weights_FKP = np.ones(len(rem_list))

    print(" \n Mean of the weights is of {} \n".format(np.mean(weights_FKP)))

    '''
        COUNT THE RANDOM PAIRS IN BINS OF S AND MU
    '''

    print("Computing random pair counts... This could take a while")
    dmu = mu_max / Nmu
    cosmology = 2 # Planck cosmology
    c_light = 299792.458

    rbins = np.logspace(np.log10(rmin), np.log10(rmax), Nr)
    results_DDsmu, c_api_time = DDsmu_mocks(autocorr, cosmology, N_cores, mu_max, Nmu, rbins, rand_cat[rem_list,0], rand_cat[rem_list,1], rand_cat[rem_list,2], is_comoving_dist=True, weights1=weights_FKP, weight_type='pair_product', c_api_timer=True)
    
    r_ctrs = (results_DDsmu['smin'] + results_DDsmu['smax']) / 2.
    r_centers = np.unique(r_ctrs)

    mu = results_DDsmu['mumax'] - dmu/2
    print("Done! It took {} seconds to complete".format(c_api_time) )
   
    '''
        COMPUTE THE Q_ELL
    '''

    # Legendre Weights
    if(all_multipoles==True):
        # Include odd multipoles
        N_poles = ell_max//2 + 1 + 2
        poles = np.linspace(0,N_poles-1,N_poles) # integers indicating which multipoles to compute
        str_poles = poles.astype('str').tolist()

    elif(all_multipoles==False):
        # Only even multipoles
        N_poles = ell_max//2 + 1
        poles = 2.*np.linspace(0,N_poles-1,N_poles)
        str_poles = poles.astype('str').tolist()

    print("Computing the multipoles {} ".format(poles) )
    w_ell = np.zeros( (N_poles, len(mu) ) )
    rr_ell = np.zeros( (N_poles, len(rbins) - 1 ) )
    q_ell = np.zeros( (N_poles, len(rbins) -1 ) )
    
    for i in range(N_poles):
        ell = int(poles[i])
    
        print( "Now in ell = {}".format(ell) )
    
        w_ell[i,:] = scipy.special.eval_legendre(ell, mu)
    
        for j in range(len(rbins)-1):
            pos_j = (r_ctrs == r_centers[j])
            rr_ell[i,j] = 0.5 * ( 2 * ell + 1 ) * np.trapz( results_DDsmu['npairs'][pos_j] * w_ell[i,pos_j] / r_centers[j]**3, mu[pos_j] )
        q_ell[i,:] = rr_ell[i,:]
    
    # Normalize the Q_ell with Q_0(0)
    Q_ell = np.zeros( (N_poles, len(rbins)-1 ) )
    if(Normalize):
        for i in range(N_poles):
            Q_ell[i,:] = q_ell[i,:]/q_ell[0,0]
    else:
        for i in range(N_poles):
            Q_ell[i,:] = q_ell[i,:]
           
    print("Done!")

    return (r_centers, Q_ell)


def convolved_multipoles(rsd_params, cosmo, redshifts, whichspec, r_centers, Q_ell, **kwargs):
    '''
    Code to compute the power-spectrum multipoles convolved with the survey window function

        Parameters
        ----------
        rsd_params  : dictionary
            Dictionary containing values of parameters such as bias, velocity dispersion, etc.

        cosmo       : dictionary
            Dictionary containing values of cosmological parameters

        redshifts   : array of floats
            Array containing values of redshifts 

        whichspec   : integer
            Integer controlling which kind of spectrum we want:
                0 -> linear
                1 -> Halofit Mead version
                2 -> Halofit Casarini version

        kmin        : float
            Minimum wave-vector (h/Mpc)

        kmax        : float
            Maximum wave-vector (h/Mpc)

        Nk          : int
            Number of k bins

        r_centers   : array of floats


        Q_ell       : array of floats
            Array of shape (n, Nk) in which n is the number of multipoles computed
            and Nk is the number of k bins used.

        matgrowrate : float, optional
            Optional parameter to pass a value of matgrowrate.
            If this is not given, will use cosmological parameters
            to obtain f through f(z) = Omega_m(z)^gamma

        Pk_dict     : Dictionary
            Optional parameter to pass a certain pre-computed power-spectrum over which
            we'll place the RSD amplitudes and will convolve it with the relevant window
            function.

    '''

    import mcfit
    str_redshifts = redshifts.astype('str').tolist()

    for key, value in kwargs.items():
        if(key=='matgrowrate'):
            matgrowrate=value
        if(key=='Pk_dict'):
            Pk_dict = value

    try:
        matgrowrate
    except:
        matgrowrate = cosmo.f_evolving(redshifts)
    
    try:
        n_bias = len(rsd_params['b1'])
    except:
        n_bias = len(np.asarray([rsd_params['b1']]))

    try:
        Pk_dict
    except:
        Pk_dict = power_spectrum(cosmo, whichspec, redshifts, 1e-4, 1e2)

    # Compute the W_ell from the Q_ell
    k_ell, W0_temp = mcfit.xi2P(r_centers, l=0)(Q_ell[0,:])
    k_ell, W2_temp = mcfit.xi2P(r_centers, l=2)(Q_ell[1,:])
    
    k       = Pk_dict['k']
    M_dict  = pk_multipoles_gauss(rsd_params, cosmo, redshifts, whichspec, kmin=1e-4, kmax=1e2, Nk=len(k), k=k, matgrowrate=matgrowrate)

    conv_dict = {}

    for i in range(n_bias):
        conv_dict[i] = {}
        for j in range(len(redshifts)):

            P_mono = Pk_dict['Pk_'+str_redshifts[j]]*M_dict['mono'][j][i]
            P_quad = Pk_dict['Pk_'+str_redshifts[j]]*M_dict['quad'][j][i]
            P_hexa = Pk_dict['Pk_'+str_redshifts[j]]*M_dict['hexa'][j][i]

            r, xi0 = mcfit.P2xi(k, l=0, lowring=True)(P_mono)
            r, xi2 = mcfit.P2xi(k, l=2, lowring=True)(P_quad)
            r, xi4 = mcfit.P2xi(k, l=4, lowring=True)(P_hexa)

            if( (i==0) and (j==0) ):
                where = np.where( (r>r_centers.min()) & (r<r_centers.max()) )

                r_interp = r[where]

                Q_ell_interp = np.zeros((Q_ell.shape[0], len(r_interp) ))

                Q_ell_interp[0,:] = np.interp(r_interp, r_centers, Q_ell[0,:])
                Q_ell_interp[1,:] = np.interp(r_interp, r_centers, Q_ell[1,:])
                Q_ell_interp[2,:] = np.interp(r_interp, r_centers, Q_ell[2,:])
                Q_ell_interp[3,:] = np.interp(r_interp, r_centers, Q_ell[3,:])

            xi0_cut = xi0[where]
            xi2_cut = xi2[where]
            xi4_cut = xi4[where]

            xi0_prime = xi0_cut*Q_ell_interp[0,:] + xi2_cut*Q_ell_interp[1,:]/5 + xi4_cut*Q_ell_interp[2,:]/9
            xi2_prime = xi0_cut*Q_ell_interp[1,:] + xi2_cut*(Q_ell_interp[0,:] + 2*Q_ell_interp[1,:]/7 + 2*Q_ell_interp[2,:]/7) + xi4_cut*(2*Q_ell_interp[1,:]/7 + 100*Q_ell_interp[2,:]/693 + 25*Q_ell_interp[3,:]/143)

            kprime, Pk0_prime  = mcfit.xi2P(r_interp, l=0, lowring=True)(xi0_prime)
            kprime, Pk2_prime  = mcfit.xi2P(r_interp, l=2, lowring=True)(xi2_prime)

            if( (i==0) and (j==0)):
                W0_interp = np.interp(kprime, k_ell, W0_temp)
                W2_interp = np.interp(kprime, k_ell, W2_temp)

                W0 = W0_interp/W0_interp[0]
                W2 = W2_interp/W0_interp[0]
            # Apply the Integral Constraint

            Pk0_prime -= np.abs(W0)**2*Pk0_prime[0]
            Pk2_prime -= np.abs(W2)**2*Pk2_prime[0]

            conv_dict[i]['P0_z'+str_redshifts[j]] = Pk0_prime
            conv_dict[i]['P2_z'+str_redshifts[j]] = Pk2_prime

    return kprime, conv_dict

if __name__ == '__main__':
    import doctest
    doctest.testmod()
