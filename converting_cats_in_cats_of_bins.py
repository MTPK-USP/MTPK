'''
This class convert complete tracer catalogs (given their positions, velocities, mass, etc.)
in catalogs of tracers with different criterium (e.g., mass) to be used as different tracers 
in the MTPK code

--------
Inputs
--------
 .dat tracer catalogs contaning their masses, ..., and positions

--------
Outputs
--------
 Tracer position catalogs splited according to their criterium

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
        Method to split the tracers in bins according to their criterium

        In the case of mass, it return the Mass Function (MF) of the catalogs
        '''
        
        cats = self.default_params['cats']
        col_m = cat_specs.col_m
        col_x = cat_specs.col_x
        col_y = cat_specs.col_y
        col_z = cat_specs.col_z
        crit_min = cat_specs.crit_min
        crit_max = cat_specs.crit_max
        ntracers = cat_specs.ntracers
        skiprows = self.default_params['skiprows']
        V = cat_specs.V
        path_to_save = self.default_params['path_to_save']

        if not os.path.exists(path_to_save):
            os.makedirs(path_to_save)
        
        crit_lims = np.logspace(crit_min, crit_max, ntracers + 1)
        dataj = {}
        MF = []
        for i in range( len(cats) ):
            data = np.loadtxt(cats[i], skiprows = skiprows)
            for j in range(ntracers):
                dataj[j] = data[ np.where( (data[:, col_m] > crit_lims[j] ) & (data[:, col_m] <= crit_lims[j+1]) ) ]
                MF.append( dataj[j].shape[0] )
                np.savetxt(path_to_save+f'seed{i}_bin{j}.dat', dataj[j][:, [col_x, col_y, col_z] ])
        MF = np.array(MF)
        MF = MF.reshape( (len(cats), ntracers) )
        print('Catalogs created!')
        return np.mean(MF, axis = 0)/V

    def central_criteria(self, cat_specs):
        '''
        Method to give log10 of the central masses of tracers
        '''
        crit_min = cat_specs.crit_min
        crit_max = cat_specs.crit_max
        ntracers = cat_specs.ntracers

        crit_lims = np.logspace(crit_min, crit_max, ntracers + 1)
        crit_ctrs = []
        for i in range(ntracers):
            crit_ctrs.append( (crit_lims[i+1] + crit_lims[i])/2 )

        return crit_ctrs
