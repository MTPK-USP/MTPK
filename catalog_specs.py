'''
Class to contain catalog specifications
'''

class cat_specs:
    '''
    Function to initialize catalog specifications class
    
    Parameters
    ----------
    
    You have to specify catalog properties

    ntracers : integer
      Number of tracers to consider
    
    n_maps : integer
      Number maps

    col_m: integer                                                                                      
      It is the collumn in the catalog corresponding to the mass of the tracers

    col_x, col_y, col_z : integers                                                                      
      Necessary to create the catalogs
      Specify columns for x, y, z in the catalogs 

    x_cat_min , y_cat_min , z_cat_min, x_cat_max , y_cat_max , z_cat_max : float                        
      Min and max of coordinates for the catalogs 

    m_min, m_max: floats                                                                               
      Correspond to the log10 of the mininum and maximum masses to be considered   

    V: float                                                                               
      Correspond to box volume, without padding, without mask and everything
      It is computed as: V = (x_cat_max - x_cat_min)*(y_cat_max - y_cat_min)*(z_cat_max - z_cat_min)

    Yields
    ------
            
    KeyError
    If user passes a key which is not defined in default_params
            
    TypeError
    If user passes a variable whose type is not the one expected
'''
    
    def __init__(self, **kwargs):
        default_params = {
            'ntracers'             : 3,
            'n_maps'               : 4,
            'col_m'                : 6,
            'col_x'                : 0,
            'col_y'                : 1,
            'col_z'                : 2,
            'x_cat_min'            : 0.,
            'y_cat_min'            : 0.,
            'z_cat_min'            : 0.,
            'x_cat_max'            : 128.,
            'y_cat_max'            : 128.,
            'z_cat_max'            : 128.,
            'm_min'                : 11.5,
            'm_max'                : 13.
        }

        #Error for type and wrong/new parameters
        for key, value in kwargs.items():
            if key not in default_params.keys():
                raise KeyError(f"You may not create new parameters. Available parameters are {list(default_params.keys())}. You passed '{key}' as key.")
            if type(default_params[key]) != type(value):
                raise TypeError(f"Expected {type(default_params[key])}, got {type(value)} in key '{key}'")
            default_params[key] = value

        #Main Parameters
        self.ntracers = default_params['ntracers']
        self.n_maps = default_params['n_maps']
        self.col_m = default_params['col_m']
        self.col_x = default_params['col_x']
        self.col_y = default_params['col_y']
        self.col_z = default_params['col_z']
        self.x_cat_min = default_params['x_cat_min']
        self.y_cat_min = default_params['y_cat_min']
        self.z_cat_min = default_params['z_cat_min']
        self.x_cat_max = default_params['x_cat_max']
        self.y_cat_max = default_params['y_cat_max']
        self.z_cat_max = default_params['z_cat_max']
        self.m_min = default_params['m_min']
        self.m_max = default_params['m_max']

        #Computed Parameters
        self.V = (self.x_cat_max - self.x_cat_min)*(self.y_cat_max - self.y_cat_min)*(self.z_cat_max - self.z_cat_min)

        self.default_params = default_params

    def parameters_print(self):
        '''
        Method to print the code parameters
        '''
        for key in self.default_params:
            print('{} = {}'.format(key, self.default_params[key] ) )
        return

    #To print without calling parameters_print to print
    __repr__ = parameters_print
