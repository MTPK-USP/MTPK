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
    the code will simply use the default parameter defined below.
    However, you have to specify your code options, unless you are running the example.

    verbose : bool
      True -> Print all the stuff from the code
      False -> Otherwise

    method : string
      Choose the method to compute de power spectrum:
       method = 'FKP'
       method = 'MT'
       method = 'both'

    mas_method : string
      Identifies the mass assignment scheme to use:
       mas_method = 'NGP' -> Nearest Grid Point
       mas_method = 'CIC' -> Cloud In Cell
       mas_method = 'TSC' -> Triangular Shaped Cloud
       mas_method = 'PCS' -> Piecewise Cubic Spline

   ntracers : integer                                                                                     
      Number of tracers to consider

    use_kdip_phys : bool
      True -> use kdip_phys
      False -> the code will compute it

    kdip_phys : float
      Alpha-dipole k_dip [h/Mpc]

    multipoles_order : integer
      Parameter to specify the order of the multipoles that you want to compute:
       monopole: 0
       monopole and dipole: 2
       monopole, dipole and quadrupole: 4

    do_cross_spectra : bool
      Up to now we only compute the cross spectra for FKP method
       True -> only for more than one tracer
       False -> if the user do not need it

    use_padding : bool
      When you want to compute your spectra within a box with empty borders
       True -> do padding
       False -> otherwise

    padding_length : list of integer
      List to indicate how much padding do tou want in each direction

    use_theory_spectrum : bool
      True -> to include pre-existing power spectrum in file
      False -> to compute matter power spectrum for given cosmology

    theory_spectrum_file : string
      Name of the theory spectrum file

    use_mask : bool
      When do you want to include survey's or any other mask
       True -> to include a mask to data
       False -> otherwise

    mask_filename : string
      Name of the mask filename
      Remember that your mask has to have the dimensions of the number of grid cells that you are
     using to compute the power spectrum of your box. E.g., for a box of 128 cells in each direction
     and cell_size of 2 Mpc/h, your mask should have the dimension of 64 x 64 x 64.
        
    mass_fun : ndarray of floats
      It is a list that keeps the mass function of each tracer
      Number density in function of the volume: # of objects per tracer/box volume
            
    tracer_bias : ndarray of floats
      It is a list that keeps the bias of each tracer
   
    cell_size : float
      Cell physical size, in units of h^-1 Mpc
            
    n_x, n_y, nz : integer
      Number of cells in x, y and z directions
            
    n_x_orig, n_y_orig, n_z_orig : float
      Displacement of the origin (0,0,0) of the box with respect to Earth (in cell units)
      Used in order to account for the redshift space distortions (RSD)
      Usually, the z displacement is large because we want to use the flat sky approximation
            
    sel_fun_data : bool
      True -> If the tracer selection function is given by a separate map
      False -> Otherwise

    sel_fun_file : string
      Name of the map of the selection function

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

    use_kmin_phys : bool
      True -> Use the specified value for kmin_phys and compute kminbar using it
      False -> Use the computed kminbar

    kmin_phys : float
      Min k (in units of h^-1 Mpc) -- this should be > 1.0/box_size. In units of h^-1 Mpc

    k_min_CAMB, k_max_camb : float
      Min and max k (in units of h^-1 Mpc) for CAMB
    
    whichspec : integer
      Which spectrum we wish to use in the ln sims and estimation:
       (0) linear  -> Linear power-spectrum
       (1) HaloFit -> Halofit implementation by A. Mead
       (2) PkEqual -> Halofit implementation by L. Casarini

    bias_file : string
      Path and name of the file that keeps the tracer bias

    nbar : list of floats
      It is the mass function times cell_size^3

    ncentral : list of float
      It is the ncentral times a list

    nsigma : list of float
      It is the ncentral times a list
         
    sigz_est : list of floats
      Gaussian redshift errors of GALAXIES/TRACERS created by HOD. It is in units of the cell size

    adip : list of floats
      Alpha-type dipoles

    vdisp : list of floats
      Velocity dispersions for RSDs

    tracers_sigz_est : list of floats
      Redshift errors and dipoles of tracers

    split_tracers : bool									       
      Create grids for those tracers from a single catalog					       
       True -> Create										       
       False -> Otherwise									

    tracer_bins : array of floats
      Boundary masses of the tracers							

    mask_spillover_cells : bool
      Mask out cells outside bounds of the box:
       True -> Mask										
       False -> Otherwise								

    batch_size : integer
      Batch size for mass assignement								

    wrap : bool											       
      Boundary method to wrap galaxies around (i.e., use periodic B.C.)				
       True -> Use										
       False -> Otherwise    

    Yields
    ------
            
    KeyError
    If user passes a key which is not defined in default_params
            
    TypeError
    If user passes a variable whose type is not the one expected
'''
    
    def __init__(self, **kwargs):
        default_params = {
            'verbose'              : False,
            'method'               : 'both',
            'mas_method'           : 'CIC',
            'ntracers'             : 3,
            'use_kdip_phys'        : False,
            'kdip_phys'            : 0.005,
            'multipoles_order'     : 4,
            'do_cross_spectra'     : True,
            'use_padding'          : False,
            'padding_length'       : [10, 10, 10],
            'use_theory_spectrum'  : False,
            'theory_spectrum_file' : "theory_spectrum_file.txt",
            'use_mask'             : False,
            'mask_filename'        : "mask.hdf5",
            'mass_fun'             : np.array([1.56e-02, 4.43e-03, 1.43e-03]),
            'tracer_bias'          : np.array([1.572, 1.906, 2.442]),
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
            'use_kmin_phys'        : False,
            'kmin_phys'            : 0.05,
            'use_kmax_phys'        : True,
            'kmax_phys'            : 1.0,
            'whichspec'            : 1,
            'k_min_CAMB'           : 1.e-4,
            'k_max_CAMB'           : 1.e+0,
            'split_tracers'        : False,
            'tracer_bins'          : np.array([3.16e+11, 1.00e+12, 3.16e+12, 1.00e+13]),
            'mask_spillover_cells' : False,
            'batch_size'           : 1000000,
            'wrap'                 : False
        }

        #Error for type and wrong/new parameters
        for key, value in kwargs.items():
            if key not in default_params.keys():
                raise KeyError(f"You may not create new parameters. Available parameters are {list(default_params.keys())}. You passed '{key}' as key.")
            if type(default_params[key]) != type(value):
                raise TypeError(f"Expected {type(default_params[key])}, got {type(value)} in key '{key}'")
            default_params[key] = value

        #Main Parameters
        self.verbose = default_params['verbose']
        self.method = default_params['method']
        self.mas_method = default_params['mas_method']
        self.ntracers = default_params['ntracers']
        self.use_kdip_phys = default_params['use_kdip_phys']
        self.kdip_phys = default_params['kdip_phys']
        self.multipoles_order = default_params['multipoles_order']
        self.do_cross_spectra = default_params['do_cross_spectra']
        self.use_padding = default_params['use_padding']
        self.padding_length = default_params['padding_length']
        self.mask_filename = default_params['mask_filename']
        self.use_theory_spectrum = default_params['use_theory_spectrum']
        self.theory_spectrum_file = default_params['theory_spectrum_file']
        self.use_mask = default_params['use_mask']
        self.mask_filename = default_params['mask_filename']
        self.mass_fun = default_params['mass_fun']
        self.tracer_bias = default_params['tracer_bias']
        self.cell_size = default_params['cell_size']
        self.n_x = default_params['n_x']
        self.n_y = default_params['n_y']
        self.n_z = default_params['n_z']
        self.n_x_orig = default_params['n_x_orig']
        self.n_y_orig = default_params['n_y_orig']
        self.n_z_orig = default_params['n_z_orig']
        self.sel_fun_data = default_params['sel_fun_data']
        self.sel_fun_file = default_params['sel_fun_file']
        self.k_min_CAMB = default_params['k_min_CAMB']
        self.k_max_CAMB = default_params['k_max_CAMB']
        self.kmin_bias = default_params['kmin_bias']
        self.kmax_bias = default_params['kmax_bias']
        self.kph_central = default_params['kph_central']
        self.dkph_bin = default_params['dkph_bin']
        self.use_kmin_phys = default_params['use_kmin_phys']
        self.kmin_phys = default_params['kmin_phys']
        self.use_kmax_phys = default_params['use_kmax_phys']
        self.kmax_phys = default_params['kmax_phys']
        self.whichspec = default_params['whichspec']
        self.split_tracers = default_params['split_tracers']
        self.tracer_bins = default_params['tracer_bins']
        self.mask_spillover_cells = default_params['mask_spillover_cells']
        self.batch_size = default_params['batch_size']
        self.wrap = default_params['wrap']
        

        #Computed Parameters
        self.bias_file = self.tracer_bias
        self.nbar = self.mass_fun*(self.cell_size**3)
        self.ncentral = self.ntracers*[20.0]
        self.nsigma = self.ntracers*[1000000.0]
        self.sigz_est = self.ntracers*[0.0000001]
        self.adip = self.ntracers*[0.0000000001]
        self.vdisp = self.adip
        self.tracers_sigz_est = self.ntracers*[0.00001]
        self.tracers_adip = self.tracers_sigz_est
        self.tracers_vdisp = self.tracers_sigz_est

        self.default_params = default_params

        #Error between number of tracers and cross spec computation
        if self.do_cross_spectra == True and self.ntracers == 1:
            raise KeyError("You can only compute the cross spectrum for more than one tracer!")

    def parameters_print(self):
        '''
        Method to print the code parameters
        '''
        for key in self.default_params:
            print('{} = {}'.format(key, self.default_params[key] ) )
        return

    #To print without calling parameters_print to print
    __repr__ = parameters_print
