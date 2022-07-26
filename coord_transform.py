import numpy as np
import scipy.interpolate
import dask.array as da

class coord_transform:
    '''
    Class to transform between sky and cartesian coordinate systems    
    '''

    def __init__(self, cosmo, **kwargs):
        default_params = {
                'zmax'     : 3,
                'N_interp' : 10000 
                }

        for key, value in kwargs.items():
            if key not in default_params.keys():
                continue
            default_params[key] = value
        self.zmax = default_params['zmax']
        self.N_interp = default_params['N_interp']
        
        # Interpolating the function to compute comoving distance
        comov = np.vectorize(cosmo.comoving)
        z_interp = np.linspace(0, self.zmax, self.N_interp)
        d_interp = comov(z_interp, True)
        self.comov = scipy.interpolate.interp1d(z_interp,d_interp)
        
        # Interpolating the inverse function to compute redshift from distance
        self.inv_comov = scipy.interpolate.interp1d(d_interp, z_interp)
        

    def sky2cartesian(self, cat):
        '''
        Function to convert from sky coordinates (RA, DEC, z) to cartesian coordinates (x, y, z)
            
        Parameters
        ----------

        cat : array of floats 
            Has dimensions (N, 3) in which N is the number of objects. 
            The columns contain RA, DEC, z respectively
    
        Returns
        -------

        out_cat : array of floats 
            Has dimensions (N, 3). Columns store x, y, z respectively
        '''

        alpha = cat[:,0]*np.pi/180.
        theta = cat[:,1]*np.pi/180.
        z     = cat[:,2]
        
        N = len(z)
        out_cat = np.zeros((N,3))

        out_cat[:,0] = self.comov(z)*np.cos(theta)*np.cos(alpha)
        out_cat[:,1] = self.comov(z)*np.cos(theta)*np.sin(alpha)
        out_cat[:,2] = self.comov(z)*np.sin(theta)

        return out_cat
   
    def cartesian2sky(self, cat):
        '''
        Function to convert from cartesian coordinates (x, y, z) to sky coordinates (RA, DEC, z)

        Parameters
        ----------
        
        cat : array of floats
            Has dimensions (N, 3) in which N is the number of objects. 
            The columns contain x, y, z respectively

        Returns
        -------

        out_cat : array of floats
            Has dimensions (N, 3). Columns store RA, DEC, z respectively

        '''

        R     = np.sqrt( cat[:,0]**2 + cat[:,1]**2 + cat[:,2]**2 )
        rho = np.sqrt( cat[:,0]**2 + cat[:,1]**2 )
        alpha = np.arctan2( cat[:,1], cat[:,0] )
        theta = np.arctan2( cat[:,2], rho )
        
        N = len(R)

        z = self.inv_comov(R)
        ra = alpha*180/np.pi
        ra = da.mod(ra - 360., 360.)
        dec = theta*180/np.pi
        
        out_cat = np.zeros((N,3))

        out_cat[:,0] = ra
        out_cat[:,1] = dec
        out_cat[:,2] = z

        return out_cat
