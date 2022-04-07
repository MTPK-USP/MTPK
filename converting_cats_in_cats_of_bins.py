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
import os

class converting_cats_in_cats_of_bins:
    '''
    Function to initialize object of the class converting_cats_in_cats_of_bins

    ------------
    Parameters
    -----------
    cats : list of strings
           Contain the paths to the initial catalogs

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
        return

    def to_bins(self, cat_specs):
        '''
        Method to split the halos in bins according to their masses

        It return the Mass Function of the catalogs
        '''
        
        cats = self.default_params['cats']
        col_m = cat_specs.col_m
        col_x = cat_specs.col_x
        col_y = cat_specs.col_y
        col_z = cat_specs.col_z
        m_min = cat_specs.m_min
        m_max = cat_specs.m_max
        nhalos = cat_specs.nhalos
        skiprows = self.default_params['skiprows']
        V = cat_specs.V
        path_to_save = self.default_params['path_to_save']

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        
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

    def central_masses(self, cat_specs):
        '''
        Method to given the central masses of halos
        '''
        m_min = cat_specs.m_min
        m_max = cat_specs.m_max
        nhalos = cat_specs.nhalos

        m_lims = np.logspace(np.log10(m_min), np.log10(m_max), nhalos + 1)
        m_ctrs = []
        for i in range(nhalos):
            m_ctrs.append( (m_lims[i+1]+m_lims[i])/2 )

        return m_ctrs
