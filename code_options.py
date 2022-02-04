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

    mass_method : string

    Identifies the mass assignment scheme to use:
      mass_method = 'NGP' -> Nearest Grid Point
      mass_method = 'CIC' -> Cloud In Cell
      mass_method = 'TSC' -> Triangular Shaped Cloud
      mass_method = 'PCS' -> Piecewise Cubic Spline

    use_kdip_phys : bool

    True -> use kdip_phys
    False -> the code will compute it

    kdip_phys : float

    Alpha-dipole k_dip [h/Mpc]

    use_padding : bool

    True -> do padding
    False -> otherwise

    padding_length : list of integer
    List to indicating how much padding in each direction

    use_power_law : bool

    True -> to use power_law to multiply CAMB's spectrum
    False -> otherwise

    power_law : integer
    Number indicating the power law to multiply CAMB's spectrum

    pk_power : integer
    Number indicating the power of pk

    use_theory_spectrum : bool

    True -> to include pre-existing power spectrum in file
    False -> to compute matter power spectrum for given cosmology

    theory_spectrum_file : string

    Name of the theory spectrum file

    use_mask : bool

    True -> to include a mask to data
    False -> otherwise

    mask_filename : string

    Name of the mask filename
        
    mass_fun : ndarray of floats
    It is a list that keeps the mass function of each tracer
            
    halo_bias : ndarray of floats
    It is a list that keeps the bias of each tracer
            
    nhalos : integer
    Number of halo bins
                        
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
    Name of the map of the selection function
                        
    use_cell_low_count_thresh : bool
    True -> to use the cell low count threshold
    False -> otherwise

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

    use_kmax_phys : bool
    True -> use the specified kmax_phys parameter
    False -> will compute it, according to Nyquist frequency

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
    
    whichspec : integer
    Which spectrum to use in the ln sims and estimation:
    (0) linear (1) HaloFit (2) PkEqual

    jing_dec_sims: bool
    True -> Use Jing deconvolution for sims
    False -> Otherwise

    power_jing_sims : float
    Power used for deconvolution window function of sims

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
            'mass_method'          : 'NGP',
            'use_kdip_phys'        : False,
            'kdip_phys'            : 0.005,
            'use_padding'          : False,
            'padding_length'       : [1, 1, 1],
            'use_power_law'        : False,
            'power_law'            : 1,
            'pk_power'             : 1,
            'use_theory_spectrum'  : False,
            'theory_spectrum_file' : "theory_spectrum_file.txt",
            'use_mask'             : False,
            'mask_filename'        : "mask.hdf5",
            'mass_fun'             : np.array([1.56e-02, 4.43e-03, 1.43e-03]),
            'halo_bias'            : np.array([1.572, 1.906, 2.442]),
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
            'sel_fun_file'         : "sel_fun-N128_halos.hdf5",
            'kmin_bias'            : 0.05,
            'kmax_bias'            : 0.15,
            'kph_central'          : 0.1,
            'dkph_bin'             : 0.01,
            'kmin_phys'            : 0.05,
            'use_kmax_phys'        : False,
            'kmax_phys'            : 0.6,
            'whichspec'            : 1,
            'jing_dec_sims'        : True,
            'power_jing_sims'      : 2.0,
            'plot_all_cov'         : False,
            'use_cell_low_count_thresh': False,
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
        self.mass_method = default_params['mass_method']
        self.use_kdip_phys = default_params['use_kdip_phys']
        self.kdip_phys = default_params['kdip_phys']
        self.use_padding = default_params['use_padding']
        self.padding_length = default_params['padding_length']
        self.mask_filename = default_params['mask_filename']
        self.use_power_law = default_params['use_power_law']
        self.power_law = default_params['power_law']
        self.pk_power = default_params['pk_power']
        self.use_theory_spectrum = default_params['use_theory_spectrum']
        self.theory_spectrum_file = default_params['theory_spectrum_file']
        self.mass_fun = default_params['mass_fun']
        self.halo_bias = default_params['halo_bias']
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
        self.sel_fun_file = default_params['sel_fun_file']
        self.use_cell_low_count_thresh = default_params['use_cell_low_count_thresh']
        self.cell_low_count_thresh = default_params['cell_low_count_thresh']
        self.mult_sel_fun = default_params['mult_sel_fun']
        self.shift_sel_fun = default_params['shift_sel_fun']
        self.kmin_bias = default_params['kmin_bias']
        self.kmax_bias = default_params['kmax_bias']
        self.kph_central = default_params['kph_central']
        self.kmin_phys = default_params['kmin_phys']
        self.use_kmax_phys = default_params['use_kmax_phys']
        self.kmax_phys = default_params['kmax_phys']
        self.whichspec = default_params['whichspec']
        self.jing_dec_sims = default_params['jing_dec_sims']
        self.power_jing_sims = default_params['power_jing_sims']
        self.plot_all_cov = default_params['plot_all_cov']
        self.fudge = default_params['fudge']

        #Computed Parameters
        self.ntracers = self.nhalos
        self.tracer_ids = self.halos_ids
        self.bias_file = self.halo_bias
        self.nbar = self.mass_fun*(self.cell_size**3)
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
