import numpy as np

gamma = 3./14.

def multipoles_1(kphys, kxhat, kyhat, kzhat, rxhat, ryhat, rzhat, delta_k_half, n_k, n_mu, s):
    # 1 - kphys is the kgrid in physical units (h/Mpc)
    # 2, 3, 4 - kxhat, kyhat, kzhat are adimensional unit vectors in Fourier space
    # 5, 6, 7 - rxhat, ryhat, rzhat are adimensional unit vectors in Real space
    # 8 - delta_k_half is the density field in Fourier space
    # 9 - n_k is the exponent of k in our expressions below
    # 10 - n_mu is the exponent of \mu in our expressions below
    # 11 - s is the FOGs factor, including redshift errors and pairwise velocity (Mpc/h)

    n_x, n_y, n_z = kphys.shape

    kxhat_half = kxhat[:,:,:n_z//2+1]
    kyhat_half = kyhat[:,:,:n_z//2+1]
    kzhat_half = kzhat[:,:,:n_z//2+1]

    k_half = kphys[:,:,:n_z//2+1]

    # mu = 0
    if(n_mu==0):
        return np.fft.irfftn(delta_k_half*k_half**n_k/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
    
    if(n_mu==2):
        Bxx = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**2/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Byy = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**2/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Bzz = np.fft.irfftn(delta_k_half*k_half**n_k*kzhat_half**2/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Bxy = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half*kyhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Bxz = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half*kzhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Byz = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half*kzhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Bterm = rxhat**2*Bxx + ryhat**2*Byy + rzhat**2*Bzz + 2.0*rxhat*ryhat*Bxy + 2.0*rxhat*rzhat*Bxz + 2.0*ryhat*rzhat*Byz
        
        return Bterm
    
    if(n_mu==4):
        Qxxxx = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**4/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Qyyyy = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**4/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Qzzzz = np.fft.irfftn(delta_k_half*k_half**n_k*kzhat_half**4/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        
        Qxxxy = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**3*kyhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Qxxxz = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**3*kzhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Qyyyx = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**3*kxhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Qyyyz = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**3*kzhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Qzzzx = np.fft.irfftn(delta_k_half*k_half**n_k*kzhat_half**3*kxhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Qzzzy = np.fft.irfftn(delta_k_half*k_half**n_k*kzhat_half**3*kyhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        
        Qxxyy = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**2*kyhat_half**2/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Qxxzz = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**2*kzhat_half**2/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Qyyzz = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**2*kzhat_half**2/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        
        Qxxyz = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**2*kyhat_half*kzhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Qyyxz = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**2*kxhat_half*kzhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        Qzzxy = np.fft.irfftn(delta_k_half*k_half**n_k*kzhat_half**2*kxhat_half*kyhat_half/(1.+gamma*(k_half*s)**2), s=[n_x,n_y,n_z])
        
        Qterm_1 = rxhat**4*Qxxxx + ryhat**4*Qyyyy + rzhat**4*Qzzzz
        Qterm_2 = 4*(rxhat**3*ryhat*Qxxxy + rxhat**3*rzhat*Qxxxz + ryhat**3*rxhat*Qyyyx + ryhat**3*rzhat*Qyyyz+ rzhat**3*rxhat*Qzzzx+ rzhat**3*ryhat*Qzzzy)
        Qterm_3 = 6*(rxhat**2*ryhat**2*Qxxyy + rxhat**2*rzhat**2*Qxxzz + ryhat**2*rzhat**2*Qyyzz)
        Qterm_4 = 12*(rxhat**2*ryhat*rzhat*Qxxyz + ryhat**2*rxhat*rzhat*Qyyxz+ rzhat**2*rxhat*ryhat*Qzzxy)
        
        return Qterm_1 + Qterm_2 + Qterm_3 + Qterm_4

def multipoles_2(kphys, kxhat, kyhat, kzhat, rxhat, ryhat, rzhat, delta_k_half, n_k, n_mu, s):
    # 1 - kphys is the kgrid in physical units (h/Mpc)
    # 2, 3, 4 - kxhat, kyhat, kzhat are adimensional unit vectors in Fourier space
    # 5, 6, 7 - rxhat, ryhat, rzhat are adimensional unit vectors in Real space
    # 8 - delta_k_half is the density field in Fourier space
    # 9 - n_k is the exponent of k in our expressions below
    # 10 - n_mu is the exponent of \mu in our expressions below
    # 11 - s is the FOGs factor, including redshift errors and pairwise velocity (Mpc/h)

    n_x, n_y, n_z = kphys.shape

    kxhat_half = kxhat[:,:,:n_z//2+1]
    kyhat_half = kyhat[:,:,:n_z//2+1]
    kzhat_half = kzhat[:,:,:n_z//2+1]

    k_half = kphys[:,:,:n_z//2+1]

    # mu = 0
    if(n_mu==0):
        return np.fft.irfftn(delta_k_half*k_half**n_k/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
    
    if(n_mu==2):
        Bxx = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**2/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Byy = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**2/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Bzz = np.fft.irfftn(delta_k_half*k_half**n_k*kzhat_half**2/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Bxy = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half*kyhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Bxz = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half*kzhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Byz = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half*kzhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Bterm = rxhat**2*Bxx + ryhat**2*Byy + rzhat**2*Bzz + 2.0*rxhat*ryhat*Bxy + 2.0*rxhat*rzhat*Bxz + 2.0*ryhat*rzhat*Byz
        
        return Bterm
    
    if(n_mu==4):
        Qxxxx = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**4/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Qyyyy = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**4/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Qzzzz = np.fft.irfftn(delta_k_half*k_half**n_k*kzhat_half**4/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        
        Qxxxy = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**3*kyhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Qxxxz = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**3*kzhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Qyyyx = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**3*kxhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Qyyyz = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**3*kzhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Qzzzx = np.fft.irfftn(delta_k_half*k_half**n_k*kzhat_half**3*kxhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Qzzzy = np.fft.irfftn(delta_k_half*k_half**n_k*kzhat_half**3*kyhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        
        Qxxyy = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**2*kyhat_half**2/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Qxxzz = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**2*kzhat_half**2/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Qyyzz = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**2*kzhat_half**2/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        
        Qxxyz = np.fft.irfftn(delta_k_half*k_half**n_k*kxhat_half**2*kyhat_half*kzhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Qyyxz = np.fft.irfftn(delta_k_half*k_half**n_k*kyhat_half**2*kxhat_half*kzhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        Qzzxy = np.fft.irfftn(delta_k_half*k_half**n_k*kzhat_half**2*kxhat_half*kyhat_half/(1.-(gamma*(k_half*s)**2)**2), s=[n_x,n_y,n_z])
        
        Qterm_1 = rxhat**4*Qxxxx + ryhat**4*Qyyyy + rzhat**4*Qzzzz
        Qterm_2 = 4*(rxhat**3*ryhat*Qxxxy + rxhat**3*rzhat*Qxxxz + ryhat**3*rxhat*Qyyyx + ryhat**3*rzhat*Qyyyz+ rzhat**3*rxhat*Qzzzx+ rzhat**3*ryhat*Qzzzy)
        Qterm_3 = 6*(rxhat**2*ryhat**2*Qxxyy + rxhat**2*rzhat**2*Qxxzz + ryhat**2*rzhat**2*Qyyzz)
        Qterm_4 = 12*(rxhat**2*ryhat*rzhat*Qxxyz + ryhat**2*rxhat*rzhat*Qyyxz+ rzhat**2*rxhat*ryhat*Qzzxy)
        
        return Qterm_1 + Qterm_2 + Qterm_3 + Qterm_4

def delta_r(kphys, kxhat, kyhat, kzhat, rxhat, ryhat, rzhat, delta_k_half, s, bias, f):
    gamma = 3./14.
    beta = f/bias
    
    c1 = multipoles_1(kphys,kxhat,kyhat,kzhat,rxhat,ryhat,rzhat,delta_k_half,0,0,s)- 0.5*s**2*multipoles_1(kphys,kxhat,kyhat,kzhat,rxhat,ryhat,rzhat,delta_k_half,2,2,s) + beta*multipoles_1(kphys,kxhat,kyhat,kzhat,rxhat,ryhat,rzhat,delta_k_half,0,2,s) - 0.5*beta*s**2*multipoles_1(kphys,kxhat,kyhat,kzhat,rxhat,ryhat,rzhat,delta_k_half,2,4,s) 
    c2 = s**2*gamma*multipoles_2(kphys,kxhat,kyhat,kzhat,rxhat,ryhat,rzhat,delta_k_half,2,0,s) + beta*s**2*gamma*multipoles_2(kphys,kxhat,kyhat,kzhat,rxhat,ryhat,rzhat,delta_k_half,2,2,s) -(37.*s**4*gamma/140.)*multipoles_2(kphys,kxhat,kyhat,kzhat,rxhat,ryhat,rzhat,delta_k_half,4,0,s) - (37.*beta*s**4*gamma/140.)*multipoles_2(kphys,kxhat,kyhat,kzhat,rxhat,ryhat,rzhat,delta_k_half,4,2,s)
    
    return c1+c2