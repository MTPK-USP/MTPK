import numpy as np

# Matter growth rate given effective gamma parameter
def matgrow(Omegam, OmegaDE, w0, w1, zcentral, gamma):
    a = 1./(1.+zcentral)
    w = w0 + (1.-a)*w1
    return( ( Omegam*a**(-3.)/( Omegam*a**(-3.) + OmegaDE*a**(-3.*(1.+w)) ) )**(gamma) )

# Hubble function
def H(H0, Omegam, OmegaDE, w0, w1, z):
    a = 1./(1.+z)
    w = w0 + (1.-a)*w1
    return( H0*np.sqrt( Omegam*a**(-3.) + OmegaDE*a**(-3.*(1.+w)) ) )

# Vector with the comoving radial distance in units of h^-1 Mpc, in intervals of dz. 
# Should be used in interpolation function
def chi_h_vec(Omegam, OmegaDE, w0, w1, zend, dz):
    zint=np.arange(0.0, zend+dz, dz)
    aint=1./(1+zint)
    w = w0 + (1.-aint)*w1
    dchi=2997.9*dz*1./np.sqrt( Omegam*aint**(-3.) + OmegaDE*aint**(-3.*(1.+w)) )
    return( np.cumsum(dchi) ) 

# Interpolation function for chi(z) . 
def chi_h_interp(chi_h_vector,zend,dz,z):
	zint=np.arange(0.0,zend+dz,dz)
	return np.interp(z,zint,chivec)


# Comoving radial distance in units of h^-1 Mpc. Standalone.
def chi_h(Omegam, OmegaDE, w0, w1, z):
    dz=0.0002
    zint=np.arange(0.0,5.0,dz)
    aint=1./(1+zint)
    w = w0 + (1.-aint)*w1
    chi=np.cumsum ( 2997.9*dz*1./np.sqrt( Omegam*aint**(-3.) + OmegaDE*aint**(-3.*(1.+w)) ) )
    return( np.interp(z,zint,chi) ) 

