'''
Class to contain code parameters
'''

import numpy as np

class code_parameters:
    '''
    Function to initialize code parameters class
    
    Parameters
    ----------
    
    All of these parameters are optional. If a parameter is not provided,
    the code will simply use the default parameter defined below

    use_mask : bool

    True -> to include some mask to data
    False -> otherwise

    use_window_function : True or False
       
	=> Eliminate this and 'dec' spectra!
	
    True -> will obtain window function for all tracers from file
    False -> will normalize data ("dec") to theoretical spectra ("model")
        
    mass_fun_file : string
    Path and name of the file that keeps the mass function
            
    halo_bias_file : string
    Path and name of the file that keeps the halo bias
            
    nhalos : integer
    Number of halo bins
            
    halo_ids : list of strings
	
	=> Eliminate this! See gal_ids eliminate too
	
    A list of strings to identify each halo bin
            
    n_maps : integer
    Number maps
            
    cell_size : float
    Cell physical size, in units of h^-1 Mpc
            
    n_x, n_y, nz : integer
    Number of cells in x,y and z directions
            
    n_x_orig, n_y_orig, n_z_orig : float
    Displacement of the origin (0,0,0) of the box with respect to Earth (in cell units)
            
    sel_fun_data : bool
    True -> If the tracer selection function is given by a separate map
    False -> Otherwise

    sel_fun_file : string
   
    => Include right after sel_fun_data
   
    Name of the map of the selection function
        
    mf_to_nbar : string
	
	=> Remove it!
	
    Path and name of the mass function file
                
    cell_low_count_thresh : float
    If low-count cells must be masked out, then cells with counts below this threshold will be eliminated from the mocks AND from the data

    mult_sel_fun, shift_sel_fun : float
    One may add a shift and/or a multiplicative factor, for testing: 
    n_bar --> factor * (n_bar + shift)

    kmin_bias, kmax_bias: float
    Interval in k that to estimate the bias
            
    kph_central : float
    Target value of k where the power estimation will be optimal (units of h/Mpc)

    dkph_bin : float
    Binning in k (in units of h^-1 Mpc) -- this should be ~ 1.4/(smallest side of box)

    kmax_phys : float
    Max k (in units of h^-1 Mpc) -- this should be < 2 \pi * Nyquist frequency = \pi /cell_size

    kmin_phys : float
    Min k (in units of h^-1 Mpc) -- this should be > 1.0/box_size. In units of h^-1 Mpc

    k_max_camb : float
    Max k (in units of h^-1 Mpc) for CAMB

    k_min_CAMB : float
    Min k (in units of h^-1 Mpc) for CAMB

    zcentral : float
    Central (mean, or median) redshift of the catalog or simulated data

    zbinwidth : float
    The width of the central redshift in the simulated data
    
    whichspec : integer
    Which spectrum to use in the ln sims and estimation:
    (0) linear (1) HaloFit (2) PkEqual

    jing_dec_sims: bool
    True -> Use Jing deconvolution for sims
    False -> Otherwise

    power_jing_sims : float
    Power used for deconvolution window function of sims

    power_jing_data : float
    Power used for data window function

    plot_all_cov : bool
    True -> To plot the 2D covariances (FKP v. MT)
    False -> Otherwise

    ntracers : integer
    Number of halo bins as tracers

    tracers_ids : list of strings
    A list of strings to identify each halo bin as tracers

    bias_file : string
    Path and name of the file that keeps the halo bias as tracers

    nbar : list of floats
    It is the mass function times cell_size^3

    ncentral : list of float
    It is the ncentral times a list

    nsigma : list of float
    It is the ncentral times a list
         
    shot_fudge : list of float
    A parameter to subtract shot noise

    shot_fudge_FKP : list of float
    A parameter to subtract shot noise, specific to FKP

    shot_fudge_MT : list of float
    A parameter to subtract shot noise, specific to MT

    sigz_est : list of floats
    Gaussian redshift errors of GALAXIES/TRACERS created by HOD. It is in units of the cell size

    adip : list of floats
    Alpha-type dipoles

    vdisp : list of floats
    Velocity dispersions for RSDs

    halos_sigz_est : list of floats
    edshift errors and dipoles of halos

    
    Yields
    ------
            
    KeyError
    If user passes a key which is not defined in default_params
            
    TypeError
    If user passes a variable whose type is not the one expected
'''
    
    def __init__(self, **kwargs):
        default_params = {
            'use_mask'             : False,
            'use_window_function'  : False,
            'mass_fun_file'        : "inputs/ExSHalos_MF.dat",
            'halo_bias_file'       : "inputs/ExSHalos_bias.dat",
            #Mudar para o caso abaixo
            # 'mass_fun'             : [0.001, 0.002, 0.003],
            # 'halo_bias'            : [3., 2.0, 1.5],
            'nhalos'               : 3,
            'halos_ids'            : ['h1', 'h2', 'h3'],
            'n_maps'               : 3,
            'cell_size'            : 1.0,
            'n_x'                  : 128,
            'n_y'                  : 128,
            'n_z'                  : 128,
            'n_x_orig'             : -64.,
            'n_y_orig'             : -64.,
            'n_z_orig'             : 10000.,
            'sel_fun_data'         : False,
            # Tirar o mf_to_nbar
            'mf_to_nbar'           : "inputs/ExSHalos_MF.dat",
            'kmin_bias'            : 0.05,
            'kmax_bias'            : 0.15,
            'kph_central'          : 0.1,
            'dkph_bin'             : 0.01,
            #Verificar o que é o parâmetro 'dkph_phys'
            'dkph_phys'            : 0.6, #=> What is this?
            'kmin_phys'            : 0.05,
            'kmax_phys'            : 0.6,
            #Tirar o 'zbinwidth'
            'zbinwidth'            : 0.1, #=> Eliminate this!
            'whichspec'            : 1,
            'jing_dec_sims'        : True,
            'power_jing_sims'      : 2.0,
            #Tirar a o 'power_jing_data'
            'power_jing_data'      : 2.0, #=> Eliminate
            'plot_all_cov'         : False,
            'cell_low_count_thresh': 0.0,
            'mult_sel_fun'         : 1.0,
            'shift_sel_fun'        : 0.0,
            'k_min_CAMB'           : 1.e-4,
            'k_max_CAMB'           : 1.e+0,
            'fudge'                : 0.0000001
        }

        #Errors
        for key, value in kwargs.items():
            if key not in default_params.keys():
                raise KeyError(f"You may not create new parameters. Available parameters are {list(default_params.keys())}. You passed '{key}' as key.")
            if type(default_params[key]) != type(value):
                raise TypeError(f"Expected {type(default_params[key])}, got {type(value)} in key '{key}'")
            default_params[key] = value

        #Main Parameters
        self.use_window_function = default_params['use_window_function']
        self.mass_fun_file = default_params['mass_fun_file']
        self.halo_bias_file = default_params['halo_bias_file']
        self.nhalos = default_params['nhalos']
        self.halos_ids = default_params['halos_ids']
        self.n_maps = default_params['n_maps']
        self.cell_size = default_params['cell_size']
        self.n_x = default_params['n_x']
        self.n_y = default_params['n_y']
        self.n_z = default_params['n_z']
        self.n_x_orig = default_params['n_x_orig']
        self.n_y_orig = default_params['n_y_orig']
        self.n_z_orig = default_params['n_z_orig']
        self.sel_fun_data = default_params['sel_fun_data']
        self.mf_to_nbar = default_params['mf_to_nbar']
        self.cell_low_count_thresh = default_params['cell_low_count_thresh']
        self.mult_sel_fun = default_params['mult_sel_fun']
        self.shift_sel_fun = default_params['shift_sel_fun']
        self.kmin_bias = default_params['kmin_bias']
        self.kmax_bias = default_params['kmax_bias']
        self.kph_central = default_params['kph_central']
        self.dkph_phys = default_params['dkph_phys']
        self.kmin_phys = default_params['kmin_phys']
        self.kmax_phys = default_params['kmax_phys']
        self.zbinwidth = default_params['zbinwidth']
        self.whichspec = default_params['whichspec']
        self.jing_dec_sims = default_params['jing_dec_sims']
        self.power_jing_sims = default_params['power_jing_sims']
        self.power_jing_data = default_params['power_jing_data']
        self.plot_all_cov = default_params['plot_all_cov']
        self.fudge = default_params['fudge']

        #Computed Parameters
        self.ntracers = self.nhalos
        self.tracer_ids = self.halos_ids
        self.bias_file = self.halo_bias_file
        self.nbar = list(np.loadtxt(self.mf_to_nbar)*(self.cell_size**3))
        self.ncentral = self.ntracers*[20.0]
        self.nsigma = self.ntracers*[1000000.0]
        self.shot_fudge = self.nhalos*[0.]
        self.shot_fudge_FKP = self.nhalos*[0.]
        self.shot_fudge_MT = self.nhalos*[0.]
        self.sigz_est = self.ntracers*[0.0000001]
        self.adip = self.ntracers*[0.0000000001]
        self.vdisp = self.adip
        self.halos_sigz_est = self.nhalos*[0.00001]
        self.halos_adip = self.halos_sigz_est
        self.halos_vdisp = self.halos_sigz_est

        self.default_params = default_params

    def parameters_print(self):
        '''
        Method to print the code parameters
        '''
        for key in self.default_params:
            print('{} = {}'.format(key, self.default_params[key] ) )
        return ''

    #To print without calling parameters_print to print
    __repr__ = parameters_print
