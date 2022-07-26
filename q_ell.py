"""Docstring for the q_ell.py module.

This module computes the window function multipoles in real space Q_ell(s).

"""

import numpy as np
from Corrfunc.theory.DDsmu import DDsmu
import coord_transform
import time
import scipy.special

def q_ell(random, cosmo, **kwargs):
    '''
    Computes window function multipoles in real space, using a pair-counting approach.

        We'll take a random catalogue and compute the random-random pairs weighted by the 
        Legendre polynomial of order \ell evaluated on the angle between the two objects, 
        as seen by the observer.
        
        Parameters
        ----------

        random : str
            Name of the file containing the random catalogue. This should have the shape
            (N, 3) in which N is the number of random points and the columns store RA, 
            DEC, and z respectively.        

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
           This should have shape (n,2) in which the first column contains values of redshift
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
    default_kwargs = {
            'ell_max'           : 4,
            'rmin'              : 0.1,
            'rmax'              : 150,
            'Nr'                : 20,
            'Nmu'               : 100,
            'all_multipoles'    : False,
            'zmin'              : 0,
            'zmax'              : 3,
            'mu_max'            : 1,
            'autocorr'          : 1,
            'N_cores'           : 64,
            'fraction'          : 1,
            'FKP_weights'       : False,
            'n_bar_r'           : None,
            'P_eff'             : 1e4
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
    '''
        LOAD N_BAR_R
    '''
    if(FKP_weights):
        print("Loading n_bar")
        n_bar = np.loadtxt(n_bar_r)
        print("Done!") 

    '''
        LOAD RANDOM CATALOGUE AND PROCESS IT
    '''

    print("Loading random catalogue...")
    t0 = time.time()
    rand_cat = np.loadtxt(random, skiprows=1)
    t1 = time.time()
    print("Done! This took {} seconds".format(t1-t0) )

    # Cut random catalogue in relevant redshift interval
    rand_cat = rand_cat[np.where( (rand_cat[:,2]>=zmin) & (rand_cat[:,2]<=zmax) )]
    print("Considering all redshifts from {} to {}".format(rand_cat[:,2].min(), rand_cat[:,2].max()) )
    
    # Remove a fraction of it, as requested by the user
    npoints = len(rand_cat[:,2])
    rem_list = np.random.choice(range(npoints), int(npoints*fraction), replace=False )
    
    if(FKP_weights):
        print("Including FKP weights")
        n_bar_interp = np.interp(rand_cat[rem_list,2], n_bar[:,0], n_bar[:,1])
        weights_FKP = 1/(1 + P_eff*n_bar_interp)
    else:
        weights_FKP = np.ones(len(rem_list))

    # Convert to cartesian coordinates
    transform = coord_transform.coord_transform(cosmo)
    rand_cat = transform.sky2cartesian(rand_cat)
    
    '''
        COUNT THE RANDOM PAIRS IN BINS OF S AND MU
    '''

    print("Computing random pair counts... This could take a while")
    rbins = np.logspace(np.log10(rmin), np.log10(rmax), Nr)
    r_centers = 0.5*(rbins[1:] + rbins[:-1])
    results_DDsmu, c_api_time = DDsmu(autocorr, N_cores, rbins, mu_max, Nmu, rand_cat[rem_list,0], rand_cat[rem_list,1], rand_cat[rem_list,2], periodic=False, c_api_timer=True, weights1=weights_FKP, weights2=weights_FKP)
    print("Done! It took {} seconds to complete".format(c_api_time) )
   
    '''
        COMPUTE THE Q_ELL
    '''

    # Legendre Weights
    if(all_multipoles==True):
        N_poles = ell_max + 1
        poles = np.arange(0,N_poles,1)
    elif(all_multipoles==False):
        N_poles = ell_max//2 + 1
        poles = 2.*np.arange(0,N_poles,1)
    print("Computing the multipoles {} ".format(poles) )

    w_ell = np.zeros( (N_poles, len(results_DDsmu['mu_max']) ) )
    rr_ell = np.zeros( (N_poles, len(rbins) - 1 ) )
    q_ell = np.zeros( (N_poles, len(rbins) -1 ) )
    for i in range(N_poles):
        ell = int(poles[i])
        print( "Now in ell = {}".format(ell) )
        w_ell[i,:] = scipy.special.eval_legendre(ell, results_DDsmu['mu_max'])
        for j in range(len(rbins)-1):
            pos_j = np.where( results_DDsmu['smin'] == rbins[j] )
            rr_ell[i,j] = (2*ell+1)*np.sum( results_DDsmu['npairs'][pos_j]*w_ell[i,pos_j] )
        q_ell[i,:] = rr_ell[i,:]/r_centers**3
    
    # Normalize the Q_ell with Q_0(0)
    Q_ell = np.zeros( (N_poles, len(rbins)-1 ) )
    for i in range(N_poles):
        Q_ell[i,:] = q_ell[i,:]/q_ell[0,0]

    
    print("Done!")

    return (r_centers, Q_ell)
