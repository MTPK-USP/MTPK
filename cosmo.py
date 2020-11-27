'''Class to contain cosmological parameters. This code also implements certain 
   very common cosmological functions: matgrow, H, d_comoving, sigma_8
'''

class cosmo:

    def __init__(self, **kwargs):
        '''Function to initialize object of the class cosmo

            Parameters
            ----------

            All of these parameters are optional. If a parameter is not provided,
            the code will simply use the default parameter defined below

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
                Parametrization for deviations of DE EoS from Cosmological Constant
            
            z_re : float
                Reionization redshift
            
            flat : Bool
                Boolean variable controling whether the user wishes to enforce a flat
                cosmology

            Yields
            ------
            
            KeyError
                If user passes a key which is not defined in default_params
            
            TypeError
                If user passes a variable whose type is not the one expected

            ValueError 
                If user selects a flat cosmology but inputs parameters inconsistent
                with this choice


        '''
        default_params = {
                'h'          : 0.678,
                'Omega0_b'   : 0.048206,
                'Omega0_cdm' : 0.2589,
                'Omega0_m'   : 0.2589 + 0.048206,
                'Omega0_k'   : 0.0
                'Omega0_DE'  : 1 - 0.2589 - 0.048206,
                'A_s'        : 2.1867466842075255e-9,
                'n_s'        : 0.96,
                'w0'         : -1,
                'w1'         : 0,
                'z_re'       : 9.99999,
                'flat'       : True
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

        if(flat):
            if(self.Omega0_m + Omega0_DE != 1):
                raise ValueError(f"This is not a flat cosmology, Omega0_m + Omega0_DE = {}".format(Omega0_m+Omega0_DE) )
        

        self.default_params = default_params
    
    '''
        METHODS
    '''

    def cosmo_print(self):
        ''' Method to print the cosmological parameters

        '''
        for key, value in self.default_params.items():
            print('{} = {}'.format(key, vars(self)[key] ) )
        return 0


    def f(self, gamma, z):
        Omega0_m = self.Omega0_m
        Omega0_DE = self.Omega0_DE
        w0 = self.w0
        w1 = self.w1

        a = 1/(1+z)
        w = w0 + (1-a)*w1

        return( ( Omega0_m*a**(-3.)/( Omega0_m*a**(-3.) + Omega0_DE*a**(-3.*(1.+w)) ) )**(gamma) )


    def H(self, z, h_units):
        '''Method to compute the Hubble factor at the cosmology defined above
        
        Parameters
        ----------
        z : float
            Redshift at which we wish to compute H
        h_units : Bool
            Boolean value controling whether we wish the output to be in units of km*(s*Mpc)^-1 
            or km*(s*Mpc/h)^-1

        Returns
        -------
        H : float
            Value of H at the selected redshift and with selected units

        '''

        if(h_units):
            h = 1
        else:
            h = cosmo.h

        Omega0_m   = self.Omega0_m
        Omega0_DE  = self.Omega0_DE
        w0         = self.w0
        w1         = self.w1

        H0 = 100*h
        a  = 1 / (1+z)
        w  = w0 + (1-a)*w1

        return( H0*np.sqrt( Omega0_m*a**(-3) + Omega0_DE*a**(-3*(1+w)) ) )

def comoving(self, z, h_units):
    '''Method to compute comoving distance using the cosmology defined above, for a certain redshift
    
    Parameters
    ----------
        z : float
            Redshit
        h_units : Bool
            Boolean variable controlling whether we want our output in units of Mpc or Mpc/h

    Returns
    -------
        d_c : float
            Comoving distance up to redshift z

    '''
    
    c_light = 299792.458 #km/s
    
    if(z==0):
        return 0
    else:
        z_temp = np.linspace(0, z, 1000)
        integrand = c / self.H(z, h_units)
        d_c = scipy.integrate.simps(integrand, z_temp)

    return (d_c)

        


