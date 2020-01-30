import numpy as np

def matgrow(Omegam, OmegaDE, w0, w1, zcentral, gamma):
    a = 1./(1.+zcentral)
    w = w0 + (1.-a)*w1
    return( ( Omegam*a**(-3.)/( Omegam*a**(-3.) + OmegaDE*a**(-3.*(1.+w)) ) )**(gamma) )

def H(H0, Omegam, OmegaDE, w0, w1, z):
    a = 1./(1.+z)
    w = w0 + (1.-a)*w1
    return( H0*np.sqrt( Omegam*a**(-3.) + OmegaDE*a**(-3.*(1.+w)) ) )
