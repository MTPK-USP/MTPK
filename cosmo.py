'''
Class to contain cosmological parameters. 
This code also implements certain very common cosmological functions: 
   - matgrow: when not defined as a new value
        1) f_evolving: evolving with z,
        2) f_phenomenological: phenomenological value, 
   - H (z), 
   - comoving distance: comoving
'''

import numpy as np
from scipy.integrate import simps

class cosmo:

    def __init__(self, **kwargs):
        '''
        Function to initialize object of the class cosmo

            Parameters
            ----------

            All of these parameters are optional. 
            If a parameter is not provided, the code will simply 
            use the default parameter defined below

            h : float
                Hubble constant parametrization H0 = 100*h

            Omega0_b : float
                Baryon density fraction at refshift 0
            
            Omega0_cdm : float
                Cold Dark Matter density fraction at redshift 0
            
            Omega0_m : float
                Total matter density fraction at redshift 0
            
            Omega0_k : float
                Curvature density fraction at redshift 0
            
            Omega0_DE : float
                Dark Energy density fraction at redshift 0
            
            A_s : float
                Primordial Spectrum amplitude
            
            n_s : float 
                Primordial spectrum spectral index
            
            w0 : float
                Dark energy EoS parameter
            
            w1 : float
                Parametrization for deviations of DE EoS 
                from Cosmological Constant
            
            z_re : float
                Reionization redshift
            
            flat : Bool
                Boolean variable controling whether the user wishes to 
                enforce a flat cosmology
                
            gamma : float
            	Parameter used in the phenomenological matgrowth factor: 
                f = Omega0_m^gamma

            matgrowcentral : float
            	Parameter of matgrowth.
            	You can use the default value or:
            	0) Define a new value
            	1) Compute using method f_evolving(z)
            	2) Compute using method f_phenomenological()

            zcentral : float
                Central (mean, or median) redshift of the catalog 
                or simulated data

            c_light : float
                Speed of light in vacuum

            Yields
            ------
            
            KeyError
                If user passes a key which is not defined in 
                default_params
            
            TypeError
                If user passes a variable whose type is not the one 
                expected

            ValueError 
                If user selects a flat cosmology but inputs parameters 
                inconsistent with this choice


        '''
        default_params = {
                'h'              : 0.678,
                'Omega0_b'       : 0.048206,
                'Omega0_cdm'     : 0.2589,
                'Omega0_m'       : 0.2589 + 0.048206,
                'Omega0_k'       : 0.0,
                'Omega0_DE'      : 1. - 0.2589 - 0.048206,
                'A_s'            : 2.1867466842075255e-9,
                'n_s'            : 0.96,
                'w0'             : -1.,
                'w1'             : 0.,
                'z_re'           : 9.99999,
                'flat'           : True,
                'gamma'          : 0.5454,
                'matgrowcentral' : 0.00001,
                'zcentral'       : 1.0,
                'c_light'        : 299792.458 #km/s
                }

        for key, value in kwargs.items():
            if key not in default_params.keys():
                raise KeyError(f"You may not create new parameters. Available parameters are {list(default_params.keys())}. You passed '{key}' as key.")
            if type(default_params[key]) != type(value):
                raise TypeError(f"Expected {type(default_params[key])}, got {type(value)} in key '{key}'")
            default_params[key] = value

        self.h          = default_params['h']
        self.Omega0_b   = default_params['Omega0_b']
        self.Omega0_cdm = default_params['Omega0_cdm']
        self.Omega0_m   = self.Omega0_b + self.Omega0_cdm
        self.Omega0_DE  = default_params['Omega0_DE']
        self.Omega0_k   = 1 -self.Omega0_m - self.Omega0_DE
        self.A_s        = default_params['A_s']
        self.n_s        = default_params['n_s']
        self.w0         = default_params['w0']
        self.w1         = default_params['w1']
        self.z_re       = default_params['z_re']
        self.flat       = default_params['flat']
        self.gamma       = default_params['gamma']
        self.c_light       = default_params['c_light']
        
        if(self.flat):
            if(self.Omega0_m + self.Omega0_DE != 1):
                raise ValueError(r"This is not a flat cosmology, Omega0_m + Omega0_DE = {}".format(self.Omega0_m + self.Omega0_DE) )
        	
        self.default_params = default_params
    
    '''
        METHODS
    '''

    def cosmo_print(self):
        ''' 
        Method to print the cosmological parameters
        '''
        for key in self.default_params:
            print('{} = {}'.format(key, self.default_params[key] ) )
        return ''

    #To print without calling cosmo_print to print
    __repr__ = cosmo_print

    def f_evolving(self, z):
        '''
        Matgrowth function - evolving with z
        '''
        Omega0_m = self.Omega0_m
        Omega0_DE = self.Omega0_DE
        w0 = self.w0
        w1 = self.w1
        gamma = self.gamma

        a = 1/(1+z)
        w = w0 + (1-a)*w1

        return( ( Omega0_m*a**(-3.)/( Omega0_m*a**(-3.) + Omega0_DE*a**(-3.*(1.+w)) ) )**(gamma) )

    def f_phenomenological(self):
        '''
        Matgrowth function - phenomenological
        '''
        Omega0_m = self.Omega0_m
        gamma = self.gamma

        return Omega0_m**gamma

    def H(self, z, h_units):
        '''
        Method to compute the Hubble factor at the cosmology 
        defined above
        
        Parameters
        ----------
        z : float
            Redshift at which we wish to compute H
        h_units : Bool
            Boolean value controling whether we wish the output to be 
            in units of km*(s*Mpc)^-1 or km*(s*Mpc/h)^-1

        Returns
        -------
        H : float
            Value of H at the selected redshift and with selected units

        '''
        if(h_units):
            h = 1
        else:
            h = self.h

        Omega0_m   = self.Omega0_m
        Omega0_DE  = self.Omega0_DE
        w0         = self.w0
        w1         = self.w1

        H0 = 100*h
        a  = 1 / (1+z)
        w  = w0 + (1-a)*w1

        return( H0*np.sqrt( Omega0_m*a**(-3) + Omega0_DE*a**(-3*(1+w)) ) )

    def comoving(self, z, h_units):
        # from scipy.integrate import simps
        '''
        Method to compute comoving distance using the cosmology defined 
        above, for a certain redshift
    
        Parameters
        ----------
        z : float
            Redshit
        h_units : Bool
            Boolean variable controlling whether we want our output in 
            units of Mpc or Mpc/h

        Returns
        -------
        d_c : float
            Comoving distance up to redshift z

        '''
    
        c = self.c_light
    
        if(z==0):
            return 0
        else:
            z_temp = np.linspace(0, z, 1000)
            integrand = c / self.H(z_temp, h_units)
            d_c = simps(integrand, z_temp)

        return (d_c)

    def chi_h(self, z):
        '''
        Comoving radial distance in units of h^-1 Mpc. Standalone.
        '''

        Omegam = self.Omega0_m
        OmegaDE = self.Omega0_DE
        w0 = self.w0
        w1 = self.w1

        dz = 0.0002
        zint = np.arange(0.0,5.0,dz)
        aint = 1./(1 + zint)
        w = w0 + (1. - aint)*w1
        chi = np.cumsum ( 2997.9*dz*1./np.sqrt( Omegam*aint**(-3.) + OmegaDE*aint**(-3.*(1.+w)) ) )
        return ( np.interp(z, zint, chi) ) 
