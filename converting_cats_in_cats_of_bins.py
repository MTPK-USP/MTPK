'''
This class convert complete halo catalogs (given their positions, velocities, mass, etc.)
in catalogs of halos with different masses to be used in the MTPK code

--------
Inputs
--------
 .dat halo catalogs contaning their masses, ..., and positions

--------
Outputs
--------
 Halo position catalogs splited according to their masses

'''

import numpy as np

class converting_cats_in_cats_of_bins:
    '''
    Function to initialize object of the class converting_cats_in_cats_of_bins

    ------------
    Parameters
    -----------
    cats : list of strings
           Contain the paths to the initial catalogs

    col_m: integer
           It is the collumn in the catalog corresponding to the mass of the halos

    col_x, col_y, col_z: integers
           Represent the collumn corresponding to the positions x, y and z

    m_min, m_max: floats
           Correspond to the mininum and maximum mass to be considered

    nhalos: integer
           Number of bins to split the halos in the catalog

    skiprows: integer
           Number of rows to skip, in the case of cats containing headers or other info

    V: float
           Volume of the box considered

    path_to_save: string
           Path to save the new catalogs

    ------
    Yields
    ------
            
    KeyError
        If user passes a key which is not defined in default_params
            
    TypeError
        If user passes a variable whose type is not the one expected

    '''

    def __init__(self, **kwargs):
        default_params = {
            'cats'          : ['data/ExSHalos/L128_000_halos.dat',
                               'data/ExSHalos/L128_001_halos.dat',
                               'data/ExSHalos/L128_002_halos.dat',
                               'data/ExSHalos/L128_003_halos.dat'],
            'skiprows'      : 1,
            'path_to_save'  : 'data/ExSHalos/'
        }

        self.default_params = default_params
        self.cats = default_params['cats']
        self.skiprows = default_params['skiprows']
        self.path_to_save = default_params['path_to_save']

        for key, value in kwargs.items():
            if key not in default_params.keys():
                raise KeyError(f"You may not create new parameters. Available parameters are {list(default_params.keys())}. You passed '{key}' as key.")
            if type(default_params[key]) != type(value):
                raise TypeError(f"Expected {type(default_params[key])}, got {type(value)} in key '{key}'")
            default_params[key] = value

    '''
        METHODS
    '''

    def params_print(self):
        ''' 
        Method to print the parameters
        '''
        for key in self.default_params:
            print('{} = {}'.format(key, self.default_params[key] ) )
        return ''

    def to_bins(self, my_code_options):
        '''
        Method to split the halos in bins according to their masses

        It return the Mass Function of the catalogs
        '''
        cats = self.default_params['cats']
        col_m = my_code_options.col_m
        col_x = my_code_options.col_x
        col_y = my_code_options.col_y
        col_z = my_code_options.col_z
        m_min = my_code_options.m_min
        m_max = my_code_options.m_max
        nhalos = my_code_options.nhalos
        skiprows = self.default_params['skiprows']
        V = my_code_options.V
        path_to_save = self.default_params['path_to_save']
        
        m_lims = np.logspace(np.log10(m_min), np.log10(m_max), nhalos + 1)
        dataj = {}
        MF = []
        for i in range( len(cats) ):
            data = np.loadtxt(cats[i], skiprows = skiprows)
            for j in range(nhalos):
                dataj[j] = data[ np.where( (data[:, col_m] > m_lims[j] ) & (data[:, col_m] <= m_lims[j+1]) ) ]
                MF.append( dataj[j].shape[0] )
                np.savetxt(path_to_save+f'seed{i}_bin{j}.dat', dataj[j][:, [col_x, col_y, col_z] ])
        MF = np.array(MF)
        MF = MF.reshape( (len(cats), nhalos) )
        print('Catalogs created!')
        return np.mean(MF, axis = 0)/V

    def central_masses(self, my_code_options):
        '''
        Method to given the central masses of halos
        '''
        m_min = my_code_options.m_min
        m_max = my_code_options.m_max
        nhalos = my_code_options.nhalos

        m_lims = np.logspace(np.log10(m_min), np.log10(m_max), nhalos + 1)
        m_ctrs = []
        for i in range(nhalos):
            m_ctrs.append( (m_lims[i+1]+m_lims[i])/2 )

        return m_ctrs
