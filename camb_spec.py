import camb
from camb import model
import numpy as np
import time
import os, sys

#this_dir = os.getcwd()
camb_dir = '/Users/lrwa/Documents/bin/camb'
MTPK_dir = '/Users/lrwa/trabalho/Multi-tracer estimators/New MTPK'

def camb_spectrum(H0, Omegab, Omegac, w0, w1, z_re, zcentral, n_SA, k_min, k_max, whicspec):
	if(k_min >= 1e-4):
		t0 = time.time()
		h = H0/100.

		os.chdir(camb_dir)

		print(os.system('pwd'))

		# Write the given parameters to the input parameter file
		params = open('params.ini')
		temp = params.readlines()

		temp[9] = 'get_transfer = T \n'
		temp[13] = 'do_lensing = F \n'
		temp[34] = 'ombh2 = ' + str(Omegab*h**2) + '\n'
		temp[35] = 'omch2 = ' + str(Omegac*h**2) + '\n'
		temp[37] = 'omk = ' + str(0.0) + '\n'
		temp[38] = 'hubble = ' + str(H0) + '\n'
		# Dark Energy Paramters
		temp[41] = 'w = ' + str(w0) + '\n'
		temp[49] = 'wa = ' + str(w1) + '\n'
		# Primordial Spectrum Parameters
		temp[86] = 'scalar_spectral_index(1) = ' + str(n_SA) + '\n'
		# Reionization Parameters 
		temp[107] = 're_use_optical_depth = F \n'
		temp[110] = 're_redshift = ' + str(z_re) + '\n' 
		temp[161] = 'transfer_kmax = ' + str(k_max) + '\n'
		temp[165] = 'transfer_redshift(1) = ' + str(zcentral) + '\n'
		temp[225] = 'accurate_polarization = F \n'
		out_name = 'params_tempz.ini'

		out = open(out_name, 'w')
		for r in temp:
		    out.write(r)
		out.close()

		if(whicspec == 0):
			os.system('./camb ' + out_name)
			#######################################################################
			########### Linear Spectrum #################
			spec = np.loadtxt('test_matpow.dat')
			kh = spec[:,0]
			pk = spec[:,1]
			#############################################
			t1 = time.time()
			print("Time elapsed:",t1-t0)

			os.chdir(MTPK_dir)

			return( np.asarray([kh,pk]) )


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

			os.system('./camb ' + out_name)

			spec = np.loadtxt('test_matpow.dat')
			kh_nl = spec[:,0]
			pk_nl = spec[:,1]

			t1 = time.time()
			print("Time elapsed:",t1-t0)

			os.chdir(MTPK_dir)

			return(np.asarray([kh_nl,pk_nl]))

			##############################################
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

			os.system('./camb ' + out_name)

			spec = np.loadtxt(camb_dir + '/test_matpow.dat')
			kh_eq = spec[:,0]
			pk_eq = spec[:,1]
			####################################################

			t1 = time.time()
			print("Time elapsed:",t1-t0)

			os.chdir(MTPK_dir)

			return(np.asarray([kh_eq, pk_eq]))
	
	else:
		h = H0/100.

		pars = camb.CAMBparams()
		
		pars.omk = 0.0
		pars.DarkEnergy.w = w0
		pars.DarkEnergy.wa = w1
		pars.ombh2 = Omegab*h**2
		pars.omch2 = Omegac*h**2
		pars.H0 = H0
		pars.InitPower.As = 2.1867
		pars.InitPower.ns = n_SA
		pars.Reion.redshift = z_re
		pars.set_matter_power(redshifts = [zcentral], kmax = k_max)

		if(whicspec == 0):
			results = camb.get_results(pars)
			kh, z, pk = results.get_matter_power_spectrum(minkh = k_min, maxkh = k_max, npoints = 1000)
			return(kh,pk[0,:])


		elif(whicspec == 1):
			pars.NonLinear = model.NonLinear_both
			results = camb.get_results(pars)
			results.calc_power_spectra(pars)
			kh_nl, z_nl, pk_nl = results.get_matter_power_spectrum(minkh = k_min, maxkh = k_max, npoints = 1000)
			return(kh_nl, pk_nl[0,:])

		else:
			pars.NonLinear = model.NonLinear_both
			pars.NonLinearModel.halofit_version = 'casarini'
			results = camb.get_results(pars)
			results.calc_power_spectra(pars)
			kh_eq, z_eq, pk_eq = results.get_matter_power_spectrum(minkh = k_min, maxkh = k_max, npoints = 1000)
			return(kh_eq, pk_eq[0,:])






