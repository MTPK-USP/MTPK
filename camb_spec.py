import camb
from camb import model
import numpy as np
import time
import os, sys

this_dir = os.getcwd()
camb_dir = 'CAMB'

def camb_spectrum(H0, Omegab, Omegac, w0, w1, z_re, zcentral, A_s, n_SA, k_min, k_max, whicspec):
	'''
		Inputs:
			H0 		-> Hubble constant at z=0;
			Omegab		-> Baryon density at z=0;
			Omegac		-> CDM density at z=0;
			w0		-> Dark energy exponent;
			w1		-> Parametrization of DE exponent;
			z_re		-> Reionization Redshift;
			zcentral	-> Redshift at which spectrum will be computed;
			A_s		-> Scalar amplitude;
			n_SA 		-> Spectral index;
			k_min		-> Minimum k for which spectrum will be computed;
			k_max		-> Maximum k for which spectrum will be computed;
			whichspec	-> Parameter to specify power-spectrum model:
								0 - Linear
								1 - HaloFit Mead
								2 - HaloFit Casarini
						
 
		This code uses CAMB to compute the matter power-spectrum and r_s_drag for the input cosmology.

		User can use the CAMB implementation given in the subdirectory MTPK/CAMB, or provide the location for his own CAMB distribution in the variable camb_dir, defined above.		

		It returns, in this order:
			kh         -> k vector (h/Mpc);
			pk         -> matter power spectrum (Mpc/h)^3;
			r_s_drag   -> (Mpc)	

	'''

	if(k_min >= 1e-4):
		t0 = time.time()
		h = H0/100.

		os.chdir(camb_dir)

		print(os.system('pwd'))

		# Write the given parameters to the input parameter file
		params = open('params.ini')
		temp = params.readlines()

		temp[9] = 'get_transfer                 = T \n'
		temp[13] = 'do_lensing                  = F \n'
		temp[34] = 'ombh2                       = ' + str(Omegab*h**2) + '\n'
		temp[35] = 'omch2                       = ' + str(Omegac*h**2) + '\n'
		temp[37] = 'omk                         = ' + str(0.0) + '\n'
		temp[38] = 'hubble                      = ' + str(H0) + '\n'
		# Dark Energy Paramters
		temp[41] = 'w                           = ' + str(w0) + '\n'
		temp[49] = 'wa                          = ' + str(w1) + '\n'
		# Primordial Spectrum Parameters
		temp[85] = 'scalar_amp(1)               = ' + str(A_s) + '\n'
		temp[86] = 'scalar_spectral_index(1)    = ' + str(n_SA) + '\n'
		# Reionization Parameters 
		# temp[107] = 're_use_optical_depth       = F \n'
		temp[110] = 're_redshift                = ' + str(z_re) + '\n' 
		temp[161] = 'transfer_kmax              = ' + str(k_max) + '\n'
		temp[165] = 'transfer_redshift(1)       = ' + str(zcentral) + '\n'
		# temp[225] = 'accurate_polarization      = F \n'
		out_name = 'params_tempz.ini'

		out = open(out_name, 'w')
		for r in temp:
		    out.write(r)
		out.close()

		if(whicspec == 0):
			########### Linear Spectrum #################
			os.system('./camb ' + out_name + ' > ' + this_dir + "/camb_out.txt")

			spec = np.loadtxt('test_matpow.dat')
			kh = spec[:,0]
			pk = spec[:,1]
			#############################################
			t1 = time.time()
			print("Time elapsed:",t1-t0)

			os.chdir(this_dir)

		elif(whicspec == 1):
			########## HaloFit mead version ##############
			params = open('params_tempz.ini')
			temp = params.readlines()

			temp[17] = 'do_nonlinear = 1 \n'

			temp[261] = 'halofit_version = 5 \n'
			print("CALLING CAMB NOW - HALOFIT")

			out = open('params_tempz.ini','w')
			for r in temp:
			    out.write(r)
			out.close()

			os.system('./camb ' + out_name + ' > ' + this_dir + "/camb_out.txt")

			spec = np.loadtxt('test_matpow.dat')
			kh = spec[:,0]
			pk = spec[:,1]

			###################################################

			t1 = time.time()
			print("Time elapsed:",t1-t0)

			os.chdir(this_dir)

		else:
			######### HaloFit casarini version ################
			params = open('params_tempz.ini')
			temp = params.readlines()

			temp[17] = 'do_nonlinear = 1 \n'

			temp[261] = 'halofit_version = 7 \n'

			out = open('params_tempz.ini','w')
			for r in temp:
			    out.write(r)
			out.close()

			os.system('./camb ' + out_name + ' > ' + this_dir + "/camb_out.txt")

			spec = np.loadtxt(camb_dir + '/test_matpow.dat')
			kh = spec[:,0]
			pk = spec[:,1]
			####################################################

			t1 = time.time()
			print("Time elapsed:",t1-t0)

			os.chdir(this_dir)


		with open('camb_out.txt', 'r') as f:
			lines = f.readlines()

			for i in range(len(lines)):
				if(lines[i].split()[0] == "r_s(zdrag)/Mpc" ):
					r_s_drag = float(lines[i].split()[-1])
			sigma8 = float(lines[23].split()[-1])

		return np.asarray([kh, pk, r_s_drag, sigma8])
	
	else:
		h = H0/100.

		pars = camb.CAMBparams()
		
		pars.omk = 0.0
		pars.DarkEnergy.w = w0
		pars.DarkEnergy.wa = w1
		pars.ombh2 = Omegab*h**2
		pars.omch2 = Omegac*h**2
		pars.H0 = H0
		pars.InitPower.As = A_s
		pars.InitPower.ns = n_SA
		pars.Reion.redshift = z_re
		pars.set_matter_power(redshifts = [zcentral], kmax = k_max)

		if(whicspec == 0):
			results = camb.get_results(pars)
			kh, z, pk = results.get_matter_power_spectrum(minkh = k_min, maxkh = k_max, npoints = 1000)
			pk = pk[0,:]


		elif(whicspec == 1):
			pars.NonLinear = model.NonLinear_both
			results = camb.get_results(pars)
			results.calc_power_spectra(pars)
			kh, z_nl, pk = results.get_matter_power_spectrum(minkh = k_min, maxkh = k_max, npoints = 1000)
			pk = pk[0,:]

		else:
			pars.NonLinear = model.NonLinear_both
			pars.NonLinearModel.halofit_version = 'casarini'
			results = camb.get_results(pars)
			results.calc_power_spectra(pars)
			kh, z, pk = results.get_matter_power_spectrum(minkh = k_min, maxkh = k_max, npoints = 1000)
			pk = pk[0,:]



