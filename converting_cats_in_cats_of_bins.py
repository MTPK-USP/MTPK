'''
This class convert complete halo catalogs (given their positions, velocities, mass, etc.)
in catalogs of halos with different masses to be used in the MTPK code
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

    m_col: integer
           It is the collumn in the catalog corresponding to the mass of the halos

    x_col, y_col, z_col: integers
           Represent the collumn corresponding to the positions x, y and z

    m_min, m_max: floats
           Correspond to the mininum and maximum mass to be considered

    n_bins: integer
           Number of bins to split the halos in the catalog

    skiprows: integer
           Number of rows to skip, in the case of cats containing headers or other info

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
            'm_col'         : 6,
            'x_col'         : 0,
            'y_col'         : 1,
            'z_col'         : 2,
            'm_min'         : 10**(11.5),
            'm_max'         : 10**(13),
            'n_bins'        : 3,
            'skiprows'      : 1,
            'path_to_save'  : 'data/ExSHalos/'
        }

        self.default_params = default_params
        self.cats = default_params['cats']
        self.m_col = default_params['m_col']
        self.x_col = default_params['x_col']
        self.y_col = default_params['y_col']
        self.z_col = default_params['z_col']
        self.m_min = default_params['m_min']
        self.m_max = default_params['m_max']
        self.n_bins = default_params['n_bins']
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

    def to_bins(self):
        '''
        Method to split the halos in bins according to their masses
        '''
        cats = self.default_params['cats']
        m_col = self.default_params['m_col']
        x_col = self.default_params['x_col']
        y_col = self.default_params['y_col']
        z_col = self.default_params['z_col']
        m_min = self.default_params['m_min']
        m_max = self.default_params['m_max']
        n_bins = self.default_params['n_bins']
        skiprows = self.default_params['skiprows']
        path_to_save = self.default_params['path_to_save']
        
        m_lims = np.logspace(np.log10(m_min), np.log10(m_max), n_bins + 1)
        dataj = {}
        for i in range( len(cats) ):
            data = np.loadtxt(cats[i], skiprows = skiprows)
            for j in range(n_bins):
                dataj[j] = data[ np.where( (data[:, m_col] > m_lims[j] ) & (data[:, m_col] <= m_lims[j+1]) ) ]
                np.savetxt(path_to_save+f'seed{i}_bin{j}.dat', dataj[j][:, [x_col, y_col, z_col] ])
        return print('Catalogs created!')

    def central_masses(self):
        '''
        Method to given the central masses of halos
        '''
        m_min = self.default_params['m_min']
        m_max = self.default_params['m_max']
        n_bins = self.default_params['n_bins']

        m_lims = np.logspace(np.log10(m_min), np.log10(m_max), n_bins + 1)
        m_ctrs = []
        for i in range(n_bins):
            m_ctrs.append( (m_lims[i+1]+m_lims[i])/2 )

        return m_ctrs
