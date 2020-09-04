#
#
# RUN THIS TO RE-ESTIMATE THEORY AND RE-MAKE PLOTS
#
#
#


a_gal_sig_tot = np.array([0.0005,0.0005])
shot_fudge = [1.0, 1.0]



# Re-compute the model/theory here.
# If you are trying to fit the spectra with some theory, you may want to change
# the parameters and then running the code from this point on
#
# NOTICE that this is repeated from above; it is appearing here just for simplicity:
# copy-and-paste everything from below in order to re-compute the theory and
# compare with the estimated spectra. (Do this after changing some of the parameters, of course)
#

pk_mg = pkmg.pkmg(gal_bias,dip,matgrowcentral,k_camb,a_gal_sig_tot,cH,zcentral)

monopoles = pk_mg.mono
quadrupoles = pk_mg.quad

try:
    pk_mg_cross = pkmg_cross.pkmg_cross(gal_bias,dip,matgrowcentral,k_camb,a_gal_sig_tot,cH,zcentral)
    cross_monopoles = pk_mg_cross.monos
    cross_quadrupoles = pk_mg_cross.quads
except:
    cross_monopoles = np.zeros((len(k_camb),1))
    cross_quadrupoles = np.zeros((len(k_camb),1))

try:
    Pk_camb_sim = np.loadtxt(dir_spec_corr_sims + '/Pk_camb.dat')[:,1]
    spec_corr = np.loadtxt(dir_spec_corr_sims + '/spec_corrections.dat')
    k_corr = spec_corr[:,0]
    nks = len(k_corr)
    try:
        mono_model = np.loadtxt(dir_spec_corr_sims + '/monopole_model.dat')
        quad_model = np.loadtxt(dir_spec_corr_sims + '/quadrupole_model.dat')
        mono_theory = np.loadtxt(dir_spec_corr_sims + '/monopole_theory.dat')
        quad_theory = np.loadtxt(dir_spec_corr_sims + '/quadrupole_theory.dat')
        if(ntracers>1):
            crossmono_model = np.loadtxt(dir_spec_corr_sims + '/cross_monopole_model.dat')
            crossquad_model = np.loadtxt(dir_spec_corr_sims + '/cross_quadrupole_model.dat')
            crossmono_theory = np.loadtxt(dir_spec_corr_sims + '/cross_monopole_theory.dat')
            crossquad_theory = np.loadtxt(dir_spec_corr_sims + '/cross_quadrupole_theory.dat')
            if len(crossmono_model.shape) == 1:
                crossmono_model = np.reshape(crossmono_model, (nks, 1))
                crossquad_model = np.reshape(crossquad_model, (nks, 1))
                crossmono_theory = np.reshape(crossmono_theory, (nks, 1))
                crossquad_theory = np.reshape(crossquad_theory, (nks, 1))
    except:
        pk_mg2 = pkmg.pkmg(gal_bias,dip,matgrowcentral,k_corr,a_gal_sig_tot,cH,zcentral)
        monopoles2 = pk_mg2.mono
        quadrupoles2 = pk_mg2.quad
        try:
            pk_mg_cross2 = pkmg_cross.pkmg_cross(gal_bias,dip,matgrowcentral,k_corr,a_gal_sig_tot,cH,zcentral)
            cross_monopoles2 = pk_mg_cross2.monos
            cross_quadrupoles2 = pk_mg_cross2.quads
        except:
            cross_monopoles = np.zeros((len(k_corr),1))
            cross_quadrupoles = np.zeros((len(k_corr),1))
        mono_model = np.ones((nks,ntracers))
        quad_model = np.ones((nks,ntracers))
        mono_theory = np.ones((nks,ntracers))
        quad_theory = np.ones((nks,ntracers))
        if(ntracers>1):
            crossmono_model = np.ones((nks,ntracers*(ntracers-1)//2))
            crossquad_model = np.ones((nks,ntracers*(ntracers-1)//2))
            crossmono_theory = np.ones((nks,ntracers*(ntracers-1)//2))
            crossquad_theory = np.ones((nks,ntracers*(ntracers-1)//2))
        index=0
        for nt in range(ntracers):
            mono_model[:,nt]= monopoles2[nt]*Pk_camb_sim
            quad_model[:,nt]= quadrupoles2[nt]*Pk_camb_sim
            mono_theory[:,nt]= monopoles2[nt]*Pk_camb_sim
            quad_theory[:,nt]= quadrupoles2[nt]*Pk_camb_sim
            for ntp in range(nt+1,ntracers):
                crossmono_model[:,index] = cross_monopoles2[index]*Pk_camb_sim
                crossmono_theory[:,index] = cross_monopoles2[index]*Pk_camb_sim
                crossquad_model[:,index] = cross_quadrupoles2[index]*Pk_camb_sim
                crossquad_theory[:,index] = cross_quadrupoles2[index]*Pk_camb_sim
                index += 1
except:
    k_corr = k_camb
    nks = len(k_camb)
    spec_corr = np.ones((nks,ntracers+1))
    mono_model = np.ones((nks,ntracers))
    quad_model = np.ones((nks,ntracers))
    mono_theory = np.ones((nks,ntracers))
    quad_theory = np.ones((nks,ntracers))
    #mono_model[:,0]=k_camb
    #quad_model[:,0]=k_camb
    #mono_theory[:,0]=k_camb
    #mono_theory[:,0]=k_camb
    if(ntracers>1):
        crossmono_model = np.ones((nks,ntracers*(ntracers-1)//2))
        crossquad_model = np.ones((nks,ntracers*(ntracers-1)//2))
        crossmono_theory = np.ones((nks,ntracers*(ntracers-1)//2))
        crossquad_theory = np.ones((nks,ntracers*(ntracers-1)//2))

    index=0
    for nt in range(ntracers):
        mono_model[:,nt]= monopoles[nt]*Pk_camb
        quad_model[:,nt]= quadrupoles[nt]*Pk_camb
        mono_theory[:,nt]= monopoles[nt]*Pk_camb
        quad_theory[:,nt]= quadrupoles[nt]*Pk_camb
        for ntp in range(nt+1,ntracers):
            crossmono_model[:,index] = cross_monopoles[index]*Pk_camb
            crossmono_theory[:,index] = cross_monopoles[index]*Pk_camb
            crossquad_model[:,index] = cross_quadrupoles[index]*Pk_camb
            crossquad_theory[:,index] = cross_quadrupoles[index]*Pk_camb
            index += 1

# Discard the first column of spec_corr, since it just gives the values of k
spec_corr = spec_corr[:,1:]

# NOW INCLUDING CROSS-CORRELATIONS
all_mono_model = np.zeros((nks,ntracers,ntracers))
all_quad_model = np.zeros((nks,ntracers,ntracers))
all_mono_theory = np.zeros((nks,ntracers,ntracers))
all_quad_theory = np.zeros((nks,ntracers,ntracers))

index=0
for i in range(ntracers):
    all_mono_model[:,i,i] = mono_model[:,i]
    all_mono_theory[:,i,i] = mono_theory[:,i]
    all_quad_model[:,i,i] = quad_model[:,i]
    all_quad_theory[:,i,i] = quad_theory[:,i]
    for j in range(i+1,ntracers):
        all_mono_model[:,i,j] = crossmono_model[:,index]
        all_mono_theory[:,i,j] = crossmono_theory[:,index]
        all_quad_model[:,i,j] = crossquad_model[:,index]
        all_quad_theory[:,i,j] = crossquad_theory[:,index]
        all_mono_model[:,j,i] = crossmono_model[:,index] 
        all_mono_theory[:,j,i] = crossmono_theory[:,index] 
        all_quad_model[:,j,i] = crossquad_model[:,index]
        all_quad_theory[:,j,i] = crossquad_theory[:,index]
        index += 1

k_spec_corr = np.append(np.append(0.,k_corr),k_corr[-1] + dkph_bin )
pk_ln_spec_corr = np.vstack((np.vstack((np.asarray(spec_corr[0]),np.asarray(spec_corr))),spec_corr[-1]))

pk_ln_mono_model = np.vstack((np.vstack((np.asarray(mono_model[0]),np.asarray(mono_model))),mono_model[-1]))
pk_ln_quad_model = np.vstack((np.vstack((np.asarray(quad_model[0]),np.asarray(quad_model))),quad_model[-1]))

pk_ln_mono_theory = np.vstack((np.vstack((np.asarray(mono_theory[0]),np.asarray(mono_theory))),mono_theory[-1]))
pk_ln_quad_theory = np.vstack((np.vstack((np.asarray(quad_theory[0]),np.asarray(quad_theory))),quad_theory[-1]))

if (ntracers>1):
    pk_ln_crossmono_model = np.vstack((np.vstack((np.asarray(crossmono_model[0]),np.asarray(crossmono_model))),crossmono_model[-1]))
    pk_ln_crossmono_theory = np.vstack((np.vstack((np.asarray(crossmono_theory[0]),np.asarray(crossmono_theory))),crossmono_theory[-1]))

    pk_ln_crossquad_model = np.vstack((np.vstack((np.asarray(crossquad_model[0]),np.asarray(crossquad_model))),crossquad_model[-1]))
    pk_ln_crossquad_theory = np.vstack((np.vstack((np.asarray(crossquad_theory[0]),np.asarray(crossquad_theory))),crossquad_theory[-1]))


# Theory monopoles of spectra (as realized on the rectangular box)
pk_ln_spec_corr_kbar=np.zeros((ntracers,pow_bins))
P0_theory=np.zeros((ntracers,pow_bins))
P2_theory=np.zeros((ntracers,pow_bins))
P0_model=np.zeros((ntracers,pow_bins))
P2_model=np.zeros((ntracers,pow_bins))

Cross_P0_model=np.zeros((ntracers*(ntracers-1)//2,pow_bins))
Cross_P2_model=np.zeros((ntracers*(ntracers-1)//2,pow_bins))
Cross_P0_theory=np.zeros((ntracers*(ntracers-1)//2,pow_bins))
Cross_P2_theory=np.zeros((ntracers*(ntracers-1)//2,pow_bins))

index=0
for i in range(ntracers):
    pk_ln_spec_corr_kbar[i] = np.interp(kph,k_spec_corr,pk_ln_spec_corr[:,i])
    P0_model[i] = np.interp(kph,k_spec_corr,pk_ln_mono_model[:,i])
    P2_model[i] = np.interp(kph,k_spec_corr,pk_ln_quad_model[:,i])
    P0_theory[i] = np.interp(kph,k_spec_corr,pk_ln_mono_theory[:,i])
    P2_theory[i] = np.interp(kph,k_spec_corr,pk_ln_quad_theory[:,i])
    for j in range(i+1,ntracers):
        Cross_P0_model[index] = np.interp(kph,k_spec_corr,pk_ln_crossmono_model[:,index])
        Cross_P2_model[index] = np.interp(kph,k_spec_corr,pk_ln_crossquad_model[:,index])
        Cross_P0_theory[index] = np.interp(kph,k_spec_corr,pk_ln_crossmono_theory[:,index])
        Cross_P2_theory[index] = np.interp(kph,k_spec_corr,pk_ln_crossquad_theory[:,index])
        index += 1

# Corrections for cross-spectra
cross_pk_ln_spec_corr_kbar=np.zeros((ntracers*(ntracers-1)//2,pow_bins))
index = 0
for i in range(ntracers):
    for j in range(i+1,ntracers):
        cross_pk_ln_spec_corr_kbar[index] = np.sqrt(pk_ln_spec_corr_kbar[i]*pk_ln_spec_corr_kbar[j])
        index +=1




################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################




################################################################################
# 2) Extra shot noise/1-halo-term subtraction
print()
print ('Reducing extra shot noise...')

ksn1 = (3*pow_bins)//4
ksn2 = -1
spec_index = np.mean(np.diff(np.log(powtrue[ksn1:ksn2]))/np.diff(np.log(kph[ksn1:ksn2])))

P1h_fkp_data = np.zeros(ntracers)
P1h_fkp_sims = np.zeros(ntracers)
P1h_MT_data = np.zeros(ntracers)
P1h_MT_sims = np.zeros(ntracers)

for nt in range(ntracers):
    #print
    #print "Data"
    #print "Tracer", nt
    spec_index = np.mean(np.diff(np.log(P0_model[nt,ksn1:ksn2]))/np.diff(np.log(kph[ksn1:ksn2])))

    frac_p1h = 1.0 - 1./spec_index*np.mean(np.diff(np.log(np.abs(P0_fkp[0,nt,ksn1:ksn2])))/np.diff(np.log(kph[ksn1:ksn2])))
    #print 'FKP:', frac_p1h
    frac_p1h = np.min([0.9, np.max([frac_p1h,-0.9])])
    P1h_fkp_data[nt] = frac_p1h * np.mean(P0_fkp[0,nt,ksn1:ksn2])
    if np.isnan(P1h_fkp_data[nt]):
        P1h_fkp_data[nt] = 0.0

    frac_p1h = 1.0 - 1./spec_index*np.mean(np.diff(np.log(np.abs(P0_data[0,nt,ksn1:ksn2])))/np.diff(np.log(kph[ksn1:ksn2])))
    #print 'MT:', frac_p1h
    frac_p1h = np.min([0.9, np.max([frac_p1h,-0.9])])
    P1h_MT_data[nt] = frac_p1h * np.mean(P0_data[0,nt,ksn1:ksn2])
    if np.isnan(P1h_MT_data[nt]):
        P1h_MT_data[nt] = 0.0

    frac_p1h = 1.0 - 1./spec_index*np.mean(np.diff(np.log(np.abs(P0_fkp_mean[nt,ksn1:ksn2])))/np.diff(np.log(kph[ksn1:ksn2])))
    #print "Sims"
    #print 'FKP:', frac_p1h
    frac_p1h = np.min([0.9, np.max([frac_p1h,-0.9])])
    P1h_fkp_sims[nt] = frac_p1h * np.mean(P0_fkp_mean[nt,ksn1:ksn2])
    if np.isnan(P1h_fkp_sims[nt]):
        P1h_fkp_sims[nt] = 0.0

    frac_p1h = 1.0 - 1./spec_index*np.mean(np.diff(np.log(np.abs(P0_mean[nt,ksn1:ksn2])))/np.diff(np.log(kph[ksn1:ksn2])))
    #print ' MT:', frac_p1h
    frac_p1h = np.min([0.9, np.max([frac_p1h,-0.9])])
    P1h_MT_sims[nt] = frac_p1h * np.mean(P0_mean[nt,ksn1:ksn2])
    if np.isnan(P1h_MT_sims[nt]):
        P1h_MT_sims[nt] = 0.0

print("   P_shot FKP=",P1h_fkp_sims)
print("  P_shot MTOE=",P1h_MT_sims)
print()

# Here subtract the shot noise, using the shot_fudge defined in the input file
P0_fkp_dec[0] = P0_fkp[0] - np.outer(shot_fudge*P1h_fkp_data,np.ones_like(kph))
P0_data_dec[0] = P0_data[0] - np.outer(shot_fudge*P1h_MT_data,np.ones_like(kph))
for nt in range(ntracers):
    #P0_fkp_dec[1:,nt] = P0_fkp[1:,nt] - np.outer(shot_fudge[nt]*P1h_fkp_sims[nt],np.ones_like(kph))
    #P0_data_dec[1:,nt] = P0_data[1:,nt] - np.outer(shot_fudge[nt]*P1h_MT_sims[nt],np.ones_like(kph))
    P0_fkp_dec[:,nt] = P0_fkp[:,nt] - np.outer(shot_fudge[nt]*P1h_fkp_sims[nt],np.ones_like(kph))
    P0_data_dec[:,nt] = P0_data[:,nt] - np.outer(shot_fudge[nt]*P1h_MT_sims[nt],np.ones_like(kph))


P0_fkp_mean_dec = np.mean(P0_fkp_dec[1:],axis=0)
P0_mean_dec = np.mean(P0_data_dec[1:],axis=0)

P2_fkp_dec = np.copy(P2_fkp)
P2_data_dec = np.copy(P2_data)

P2_fkp_mean_dec = np.mean(P2_fkp_dec[1:],axis=0)
P2_mean_dec = np.mean(P2_data_dec[1:],axis=0)


#
# Plot MT estimates along with theory -- convolved spectra

pl.rcParams["axes.titlesize"] = 8
cm_subsection = np.linspace(0, 1, ntracers)
mycolor = [ cm.jet(x) for x in cm_subsection ]

pl.xscale('log')
pl.yscale('log')
xlow=0.99*kph[2]
xhigh=1.01*kph[-1]

for nt in range(ntracers):
    ddk = dkph_bin*0.1*(nt-ntracers/2.0+0.25)
    # Monopole
    color1=mycolor[nt]
    p = P0_data_dec[0,nt]
    errp = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt].T)))
    pl.errorbar(kph+ddk,p,errp,color=color1,linestyle='None', marker='s', capsize=3, markersize=3)
    pl.errorbar(kph+ddk,-p,errp,color=color1,linestyle='None', marker='s', capsize=1, markersize=1)
    p = effbias[nt]**2*P0_fkp_dec[0,nt]
    errp = np.sqrt(np.diag(np.cov(effbias[nt]**2*P0_fkp_dec[1:,nt].T)))
    pl.errorbar(kph+2*ddk,p,errp,color=color1,linestyle='None', marker='^', capsize=3, markersize=3)
    pl.errorbar(kph+2*ddk,-p,errp,color=color1,linestyle='None', marker='^', capsize=1, markersize=1)
    #pl.plot(kph,gp*P0_model[nt],color=color1,linewidth=1.2)
    pl.plot(kph,P0_mean_dec[nt],color=color1,linewidth=0.3)
    pl.plot(kph,P0_theory[nt],color=color1,linewidth=1.0)
    p = P2_data_dec[0,nt]
    errp = np.sqrt(np.diag(np.cov(P2_data_dec[1:,nt].T)))
    pl.errorbar(kph+ddk,p,errp,color=color1,linestyle='None', marker='s', capsize=3, markersize=3)
    pl.errorbar(kph+ddk,-p,errp,color=color1,linestyle='None', marker='s', capsize=1, markersize=1)
    p = effbias[nt]**2*P2_fkp_dec[0,nt]
    errp = np.sqrt(np.diag(np.cov(effbias[nt]**2*P2_fkp_dec[1:,nt].T)))
    pl.errorbar(kph+2*ddk,p,errp,color=color1,linestyle='None', marker='^', capsize=3, markersize=3)
    pl.errorbar(kph+2*ddk,-p,errp,color=color1,linestyle='None', marker='^', capsize=1, markersize=1)
    #pl.plot(kph,gp*P0_model[nt],color=color1,linewidth=1.2)
    pl.plot(kph,np.abs(P2_mean_dec[nt]),color=color1,linewidth=0.3)
    pl.plot(kph,np.abs(P2_theory[nt]),color=color1,linewidth=1.0)


pl.legend()
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(\ell)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.xlim([xlow,xhigh])
pl.ylim([10,10**4])
pl.title(r'Convolved, after shot noise subtraction',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_errbar_Data_Conv_corr_kbar=' + str(kph_central) + '.pdf')
pl.close('all')


# Update bias estimate
kb_cut_min = np.argsort(np.abs(kph-kmin_bias))[0]
kb_cut_max = np.argsort(np.abs(kph-kmax_bias))[0]

def residuals(norm,data):
    return norm*data[1] - data[0]

# The measured monopoles and the flat-sky/theory monopoles are different; find relative normalization
normmonos = np.zeros(ntracers)
chi2_red = np.zeros(ntracers)

for nt in range(ntracers):
    err = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt,kb_cut_min:kb_cut_max].T)))
    data = [ P0_data_dec[0,nt,kb_cut_min:kb_cut_max]/err , P0_mean_dec[nt,kb_cut_min:kb_cut_max]/err ]
    this_norm, success = leastsq(residuals,1.0,args=data)
#    print weights_mono
    normmonos[nt] = np.sqrt(this_norm)
#    norm_monos[nt] = np.mean(np.sqrt(P0_data_dec[0,nt,kb_cut_min:kb_cut_max]/P0_mean_dec[nt,kb_cut_min:kb_cut_max]))
    chi2_red[nt] = np.sum(residuals(this_norm,data)**2)/(len(err)-1.)

print()
print ("Fiducial bias of the mocks, compared with data, before deconvolution:")
for nt in range(ntracers):
    print ('Fiducial (mocks):', np.around(gal_bias[nt],3), '; based on these mocks, data bias should be:', np.around(normmonos[nt]*gal_bias[nt],3), ' (chi^2 = ', np.around(chi2_red[nt],3), ')')

print ()



################################################################################
# 3) Corrections for lognormal maps / sim bias
if sims_only:
    P0_fkp_dec = (P0_fkp_dec*pk_ln_spec_corr_kbar)
    P2_fkp_dec = (P2_fkp*pk_ln_spec_corr_kbar)
    Cross0_dec = (Cross0*cross_pk_ln_spec_corr_kbar)
    Cross2_dec = (Cross2*cross_pk_ln_spec_corr_kbar)

    P0_data_dec = (P0_data_dec*pk_ln_spec_corr_kbar)
    P2_data_dec = (P2_data*pk_ln_spec_corr_kbar)
else:
    P0_fkp_dec[1:] = (P0_fkp_dec[1:]*pk_ln_spec_corr_kbar)
    P2_fkp_dec[1:] = (P2_fkp[1:]*pk_ln_spec_corr_kbar)
    Cross0_dec[1:] = (Cross0[1:]*cross_pk_ln_spec_corr_kbar)
    Cross2_dec[1:] = (Cross2[1:]*cross_pk_ln_spec_corr_kbar)

    P0_data_dec[1:] = (P0_data_dec[1:]*pk_ln_spec_corr_kbar)
    P2_data_dec[1:] = (P2_data[1:]*pk_ln_spec_corr_kbar)


P0_fkp_mean_dec = np.mean(P0_fkp_dec[1:],axis=0)
P0_mean_dec = np.mean(P0_data_dec[1:],axis=0)
Cross0_mean_dec = np.mean(Cross0_dec[1:],axis=0)
Cross2_mean_dec = np.mean(Cross2_dec[1:],axis=0)

P2_fkp_mean_dec = np.mean(P2_fkp_dec[1:],axis=0)
P2_mean_dec = np.mean(P2_data_dec[1:],axis=0)




################################################################################
# 4) Window functions corrections (computed from simulations or from theory)

winfun0 = np.ones((ntracers,pow_bins))
winfun0_cross = np.ones((ntracers*(ntracers-1)//2,pow_bins))

winfun2 = np.ones((ntracers,pow_bins))
winfun2_cross = np.ones((ntracers*(ntracers-1)//2,pow_bins))


# In order to be consistent with older versions:
try:
	use_window_function
	is_n_body_sims = use_window_function
except:
	use_window_function = is_n_body_sims


if use_window_function:
    try:
        win_fun_dir = this_dir + "/spectra/" + win_fun_dir
    except:
        print("You must define the directory where the window functions will be found.")
        print("Please check your input file. Aborting now...")
        print()
        sys.exit(-1)
    print()
    print("Using window function from directory:",win_fun_dir)
    print()
    win_fun_file=glob.glob(win_fun_dir+"*WinFun02*")
    win_fun_file_cross0=glob.glob(win_fun_dir+"*WinFun_Cross0*")
    win_fun_file_cross2=glob.glob(win_fun_dir+"*WinFun_Cross2*")
    if (len(win_fun_file)!=1) | (len(win_fun_file_cross0)!=1) | (len(win_fun_file_cross2)!=1):
        print ("Could not find (or found more than one) specified window functions at", "spectra/", win_fun_dir)
        print ("Using no window function")
        wf = np.ones((pow_bins,2*ntracers))
        wf_c0 = np.ones((pow_bins,ntracers*(ntracers-1)//2))
        wf_c2 = np.ones((pow_bins,ntracers*(ntracers-1)//2))
    else:
        win_fun_file = win_fun_file[0]
        win_fun_file_cross0 = win_fun_file_cross0[0]
        win_fun_file_cross2 = win_fun_file_cross2[0]

        wf = np.loadtxt(win_fun_file)

        if (len(wf) != pow_bins) | (len(wf.T) != 2*ntracers):
            print ("Dimensions of window functions (P0 auto & P2 auto) do not match those of this estimation code!")
            print ("Please check that window function, or create a new one. Aborting now...")
            sys.exit(-1)

        if(ntracers>1):
            wf_c0 = np.loadtxt(win_fun_file_cross0)
            wf_c2 = np.loadtxt(win_fun_file_cross2)

            if ( (ntracers>2) and (len(wf_c0) != pow_bins) | (len(wf_c0.T) != ntracers*(ntracers-1)//2) ):
                print ("Dimensions of window function of cross spectra for P0 do not match those of this estimation code!")
                print ("Please check that window function, or create a new one. Aborting now...")
                sys.exit(-1)
            if (len(wf_c0.T)!=pow_bins):
                print ("Dimensions of window function of cross spectra for P0 do not match those of this estimation code!")
                print ("Please check that window function, or create a new one. Aborting now...")
                sys.exit(-1)
            wf_c0 = np.reshape(wf_c0, (pow_bins,1))

            if ((ntracers>2) and (len(wf_c2) != pow_bins) | (len(wf_c2.T) != ntracers*(ntracers-1)//2)):
                print ("Dimensions of window function of cross spectra for P2 do not match those of this estimation code!")
                print ("Please check that window function, or create a new one. Aborting now...")
                sys.exit(-1)

            if (len(wf_c2.T)!=pow_bins):
                print ("Dimensions of window function of cross spectra for P0 do not match those of this estimation code!")
                print ("Please check that window function, or create a new one. Aborting now...")
                sys.exit(-1)
            wf_c2 = np.reshape(wf_c2, (pow_bins,1))


    mean_win0 = wf[:,:ntracers]
    mean_win2 = wf[:,ntracers:]
    # Deconvolve FKP
    P0_fkp_dec = P0_fkp_dec / (small + mean_win0.T)
    P2_fkp_dec = P2_fkp_dec / (small + mean_win2.T)
    P0_fkp_mean_dec = np.mean(P0_fkp_dec,axis=0)
    P2_fkp_mean_dec = np.mean(P2_fkp_dec,axis=0)
    # Deconvolve MT
    P0_data_dec = P0_data_dec / (small + mean_win0.T)
    P2_data_dec = P2_data_dec / (small + mean_win2.T)
    P0_mean_dec = np.mean(P0_data_dec,axis=0)
    P2_mean_dec = np.mean(P2_data_dec,axis=0)

    index = 0
    for i in range(ntracers):
        for j in range(i+1,ntracers):
            Cross0_dec[:,index] = Cross0_dec[:,index] / (small + wf_c0[:,index])
            Cross2_dec[:,index] = Cross2_dec[:,index] / (small + wf_c2[:,index])
            index += 1
else:
    wf_c0 = np.ones((pow_bins,ntracers*(ntracers-1)//2))
    wf_c2 = np.ones((pow_bins,ntracers*(ntracers-1)//2))
    for nt in range(ntracers):
        # FKP has its window function...
        winfun0[nt] = effbias[nt]**2*P0_fkp_mean_dec[nt]/(small + P0_model[nt])
        winfun2[nt] = effbias[nt]**2*P2_fkp_mean_dec[nt]/(small + P2_model[nt])
        P0_fkp_dec[:,nt] = P0_fkp_dec[:,nt] / (small + winfun0[nt])
        P2_fkp_dec[:,nt] = P2_fkp_dec[:,nt] / (small + winfun2[nt])
        P0_fkp_mean_dec[nt] = np.mean(P0_fkp_dec[:,nt],axis=0)
        P2_fkp_mean_dec[nt] = np.mean(P2_fkp_dec[:,nt],axis=0)
        # MT has its window function... which is the only one we store... so, same name
        winfun0[nt] = P0_mean_dec[nt]/(small + P0_model[nt])
        winfun2[nt] = P2_mean[nt]/(small + P2_model[nt])
        P0_data_dec[:,nt] = P0_data_dec[:,nt] / (small + winfun0[nt])
        P2_data_dec[:,nt] = P2_data[:,nt] / (small + winfun2[nt])
        P0_mean_dec[nt] = np.mean(P0_data_dec[:,nt],axis=0)
        P2_mean_dec[nt] = np.mean(P2_data_dec[:,nt],axis=0)
    # Cross spectra        
    index = 0
    for i in range(ntracers):
        for j in range(i+1,ntracers):
            model0 = np.sqrt( P0_model[i]*P0_model[j] )
            model2 = np.sqrt( P2_model[i]*P0_model[j] )
            wf_c0[:,index] = effbias[i]*effbias[j]*Cross0_mean_dec[index] / (small + model0)
            wf_c2[:,index] = effbias[i]*effbias[j]*Cross2_mean_dec[index] / (small + model2)
            index += 1



################################################################################
################################################################################

# Compute the theoretical covariance:
# first, compute the Fisher matrix, then invert it.
# Do this for for each k bin


# free up memory
n_bar_matrix_fid=None
del n_bar_matrix_fid



################################################################################
################################################################################
################################################################################
################################################################################




print ('Now computing data covariances of the simulated spectra...')
tempor=time()

# First, compute total effective spectrum of all species combined
P0tot_MT = np.zeros((n_maps,num_binsk),dtype="float16")
P0tot_FKP = np.zeros((n_maps,num_binsk),dtype="float16")
for i in range(n_maps):
    P0tot_MT[i] = np.sum(nbarbar*P0_data_dec[i].T,axis=1)/ntot
    P0tot_FKP[i] = np.sum(nbarbar*effbias**2*P0_fkp_dec[i].T,axis=1)/ntot

P0tot_mean_MT = np.mean(P0tot_MT,axis=0)
P0tot_mean_FKP = np.mean(P0tot_FKP,axis=0)

# These are the RELATIVE covariances between all tracers .
# We have Dim P0,2_data : [nmaps,ntracers,num_binsk]
relcov0_MT  = np.zeros((ntracers,ntracers,num_binsk,num_binsk),dtype="float16")
relcov0_FKP = np.zeros((ntracers,ntracers,num_binsk,num_binsk),dtype="float16")
relcov2_MT  = np.zeros((ntracers,ntracers,num_binsk,num_binsk),dtype="float16")
relcov2_FKP = np.zeros((ntracers,ntracers,num_binsk,num_binsk),dtype="float16")

# Covariance of the RATIOS between tracers -- there are n*(n-1)/2 of them -- like the cross-covariances
fraccov0_MT  = np.zeros((ntracers*(ntracers-1)//2,num_binsk,num_binsk),dtype="float16")
fraccov0_FKP = np.zeros((ntracers*(ntracers-1)//2,num_binsk,num_binsk),dtype="float16")
fraccov2_MT  = np.zeros((ntracers*(ntracers-1)//2,num_binsk,num_binsk),dtype="float16")
fraccov2_FKP = np.zeros((ntracers*(ntracers-1)//2,num_binsk,num_binsk),dtype="float16")


# Covariance calculations
# Use the convolved estimators for the amplitudes of the spectra;
# Use the DEconvolved estimators for the ratios of spectra

# Build "super array" containing Ptot and ratios of spectra
P0_tot_ratios_FKP = P0tot_FKP[1:]
P0_tot_ratios_MT = P0tot_MT[1:]
ntcount=0
for nt in range(ntracers):
    rc0_MT = np.cov(P0_data_dec[1:,nt].T)/(small+np.abs(np.outer(P0_mean_dec[nt],P0_mean_dec[nt])))
    rc0_FKP = np.cov(P0_fkp_dec[1:,nt].T)/(small+np.abs(np.outer(P0_fkp_mean_dec[nt],P0_fkp_mean_dec[nt])))
    dd_rc = np.diag(dd_P_spec_kbar[nt]**2)
    norm_rc0_nt = np.var(dd_P0_rel_kbar[nt])
    dd_rc0 = norm_rc0_nt*np.diag(dd_P0_rel_kbar[nt]**2)
    relcov0_MT[nt,nt] = rc0_MT + dd_rc + dd_rc0
    relcov0_FKP[nt,nt] = rc0_FKP + dd_rc + dd_rc0
    dd_rc2 = np.diag(dd_P2_rel_kbar[nt])
    rc2_MT = np.cov(P2_data_dec[1:,nt].T)/(small+np.abs(np.outer(P2_mean_dec[nt],P2_mean_dec[nt])))
    rc2_FKP = np.cov(P2_fkp_dec[1:,nt].T)/(small+np.abs(np.outer(P2_fkp_mean_dec[nt],P2_fkp_mean_dec[nt])))
    norm_rc2_nt = np.var(dd_P2_rel_kbar[nt])
    dd_rc2 = norm_rc2_nt*np.diag(dd_P2_rel_kbar[nt]**2)
    relcov2_MT[nt,nt] = rc2_MT + dd_rc + dd_rc2
    relcov2_FKP[nt,nt] = rc2_FKP + dd_rc + dd_rc2
    for ntp in range(nt+1,ntracers):
        dd_rc = np.diag(dd_P_spec_kbar[nt]*dd_P_spec_kbar[ntp])
        norm_rc0_ntp = np.var(dd_P0_rel_kbar[ntp])
        dd_rc0 = np.sqrt(norm_rc0_nt*norm_rc0_ntp)*np.diag(dd_P0_rel_kbar[nt]*dd_P0_rel_kbar[ntp])
        norm_rc2_ntp = np.var(dd_P2_rel_kbar[ntp])
        dd_rc2 = np.sqrt(norm_rc2_nt*norm_rc2_ntp)*np.diag(dd_P2_rel_kbar[nt]*dd_P2_rel_kbar[ntp])
        ppmt = small + np.abs(np.outer(P0_mean_dec[nt],P0_mean_dec[ntp]))
        ppmt2 = small + np.abs(np.outer(P2_mean_dec[nt],P2_mean_dec[ntp]))
        ppfkp = small + np.abs(np.outer(P0_fkp_mean_dec[nt],P0_fkp_mean_dec[ntp]))
        ppfkp2 = small + np.abs(np.outer(P2_fkp_mean_dec[nt],P2_fkp_mean_dec[ntp]))
        relcov0_MT[nt,ntp] = ((np.cov(P0_data_dec[1:,nt].T,P0_data_dec[1:,ntp].T))[num_binsk:,:num_binsk])/(small+ppmt) + dd_rc + dd_rc0
        relcov2_MT[nt,ntp] = ((np.cov(P2_data_dec[1:,nt].T,P2_data_dec[1:,ntp].T))[num_binsk:,:num_binsk])/(small+ppmt2) + dd_rc + dd_rc2
        relcov0_FKP[nt,ntp] = ((np.cov(P0_fkp_dec[1:,nt].T,P0_fkp_dec[1:,ntp].T))[num_binsk:,:num_binsk])/(small+ppfkp) + dd_rc + dd_rc0
        relcov2_FKP[nt,ntp] = ((np.cov(P2_fkp_dec[1:,nt].T,P2_fkp_dec[1:,ntp].T))[num_binsk:,:num_binsk])/(small+ppfkp2) + dd_rc + dd_rc2
        relcov0_MT[ntp,nt] = relcov0_MT[nt,ntp]
        relcov2_MT[ntp,nt] = relcov2_MT[nt,ntp]
        relcov0_FKP[ntp,nt] = relcov0_FKP[nt,ntp]
        relcov2_FKP[ntp,nt] = relcov2_FKP[nt,ntp]
        rat0_MT = P0_data_dec[1:,nt].T/(small+P0_data_dec[1:,ntp].T)
        rat2_MT = P2_data_dec[1:,nt].T/(small+P2_data_dec[1:,ntp].T)
        fraccov0_MT[ntcount] = np.cov(rat0_MT)
        fraccov2_MT[ntcount] = np.cov(rat2_MT)
        rat0_FKP = effbias[nt]**2*P0_fkp_dec[1:,nt].T/(small+effbias[ntp]**2*P0_fkp_dec[1:,ntp].T)
        rat2_FKP = effbias[nt]**2*P2_fkp_dec[1:,nt].T/(small+effbias[ntp]**2*P2_fkp_dec[1:,ntp].T)
        fraccov0_FKP[ntcount] = np.cov(rat0_FKP)
        fraccov2_FKP[ntcount] = np.cov(rat2_FKP)
        P0_tot_ratios_MT = np.hstack((P0_tot_ratios_MT,rat0_MT.T))
        P0_tot_ratios_FKP = np.hstack((P0_tot_ratios_FKP,rat0_FKP.T))
        ntcount = ntcount + 1


# Correlation matrix of total effective power spectrum and ratios of spectra
cov_Pt_ratios_MT = np.cov(P0_tot_ratios_MT.T)
cov_Pt_ratios_FKP = np.cov(P0_tot_ratios_MT.T)
cov_Pt_MT = np.cov(P0tot_MT.T)
cov_Pt_FKP = np.cov(P0tot_FKP.T)

print ('Done computing data covariances. Time spent: ', np.int((time()-tempor)*1000)/1000., 's')
print ()
print ('----------------------------------')
print ()


#print('Mean D0D0/true:',np.mean(frac00))

print ('Results:')

Vfft_to_Vk = 1.0/((n_x*n_y)*(n_z/2.))

## Compare theory with sims and with data
eff_mono_fkp = np.median(effbias**2*(P0_fkp_mean_dec/powtrue).T [myran],axis=0)
eff_mono_mt = np.median((P0_mean_dec/powtrue).T [myran],axis=0)
eff_quad_fkp = np.median(effbias**2*(P2_fkp_mean_dec/powtrue).T [myran],axis=0)
eff_quad_mt = np.median((P2_mean_dec/powtrue).T [myran],axis=0)

eff_mono_fkp_data = np.median(effbias**2*(P0_fkp_dec[0]/powtrue).T [myran],axis=0)
eff_mono_mt_data = np.median((P0_data_dec[0]/powtrue).T [myran],axis=0)
eff_quad_fkp_data = np.median(effbias**2*(P2_fkp_dec[0]/powtrue).T [myran],axis=0)
eff_quad_mt_data = np.median((P2_data_dec[0]/powtrue).T [myran],axis=0)

mono_theory = np.median((P0_model/powtrue).T [myran],axis=0)
quad_theory = np.median((P2_model/powtrue).T [myran],axis=0)


print('At k=', kph[myk_min], '...', kph[myk_max])
for nt in range(ntracers):
    print('----------------------------------')
    print('Tracer:', nt )
    print('    Theory averaged monopole = ', 0.001*np.int( 1000.0*mono_theory[nt]))
    print('FKP (sims) averaged monopole = ', 0.001*np.int( 1000.0*eff_mono_fkp[nt]))
    print(' MT (sims) averaged monopole = ', 0.001*np.int( 1000.0*eff_mono_mt[nt]))
    print('FKP (data) averaged monopole = ', 0.001*np.int( 1000.0*eff_mono_fkp_data[nt]))
    print(' MT (data) averaged monopole = ', 0.001*np.int( 1000.0*eff_mono_mt_data[nt]))
    print('    Theory averaged quadrupole = ', 0.001*np.int( 1000.0*quad_theory[nt]))
    print('FKP (sims) averaged quadrupole = ', 0.001*np.int( 1000.0*eff_quad_fkp[nt]))
    print(' MT (sims) averaged quadrupole = ', 0.001*np.int( 1000.0*eff_quad_mt[nt]))
    print('FKP (data) averaged quadrupole = ', 0.001*np.int( 1000.0*eff_quad_fkp_data[nt]))
    print(' MT (data) averaged quadrupole = ', 0.001*np.int( 1000.0*eff_quad_mt_data[nt]))

print('----------------------------------')
print ()






################################################################################
################################################################################
################################################################################
################################################################################
#
# Computations are basically completed.
# Now create plots, outputs, etc.
#
################################################################################
################################################################################
################################################################################
################################################################################



pl.rcParams["axes.titlesize"] = 8
cm_subsection = np.linspace(0, 1, ntracers)
mycolor = [ cm.jet(x) for x in cm_subsection ]

#colorsequence=['darkred','r','darkorange','goldenrod','y','yellowgreen','g','lightseagreen','c','deepskyblue','b','darkviolet','m']
#jumpcolor=np.int(len(colorsequence)/ntracers)


print('Plotting results to /figures...')

if plot_all_cov:
    # Plot 2D correlation of Ptot and ratios
    nblocks=1+ntracers*(ntracers-1)//2
    indexcov = np.arange(0,nblocks*num_binsk,np.int(num_binsk//4))
    nameindex = nblocks*[str(0.001*np.round(1000*kin)) for kin in kph[0:-1:np.int(num_binsk//4)]]
    onesk=np.diag(np.ones((nblocks*num_binsk)))
    dF=np.sqrt(np.abs(np.diag(cov_Pt_ratios_FKP)))
    dM=np.sqrt(np.abs(np.diag(cov_Pt_ratios_MT)))
    dF2 = small + np.outer(dF,dF)
    dM2 = small + np.outer(dM,dM)
    fullcov = np.tril(np.abs(cov_Pt_ratios_FKP)/dF2) + np.triu(np.abs(cov_Pt_ratios_MT.T)/dM2) - onesk
    pl.imshow(fullcov,origin='lower',interpolation='none')
    pl.title("Covariance of total effective power spectrum and ratios of spectra (monopoles only)")
    pl.xticks(indexcov,nameindex,size=6,name='monospace')
    pl.yticks(indexcov,nameindex,size=8,name='monospace')
    pl.annotate('FKP',(np.int(pow_bins/5.),2*pow_bins),fontsize=20)
    pl.annotate('Multi-tracer',(2*pow_bins,np.int(pow_bins/5.)),fontsize=20)
    pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
    pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
    pl.colorbar()
    pl.savefig(dir_figs + '/' + handle_estimates + '_2D_tot_ratios_corr.pdf')
    pl.close('all')

    # Plot 2D correlation coefficients
    indexcov = np.arange(0,num_binsk,np.int(num_binsk//5))
    nameindex = [str(0.001*np.round(1000*kin)) for kin in kph[0:-1:np.int(num_binsk//5)]]
    onesk=np.diag(np.ones(num_binsk))
    for nt in range(ntracers):
        for ntp in range(nt,ntracers):
            kk = np.outer(kph,kph)
            FKPcov=relcov0_FKP[nt,ntp]
            MTcov=relcov0_MT[nt,ntp]
            dF=np.sqrt(np.abs(np.diag(FKPcov)))
            dM=np.sqrt(np.abs(np.diag(MTcov)))
            FKPcorr=FKPcov/(small+np.outer(dF,dF))
            MTcorr=MTcov/(small+np.outer(dM,dM))
            fullcov = np.tril(np.abs(FKPcorr)) + np.triu(np.abs(MTcorr.T)) - onesk
            thistitle = 'Corr(P_' + str(nt) + ',P_' + str(ntp) + ') '
            pl.imshow(fullcov,origin='lower',interpolation='none')
            pl.title(thistitle)
            pl.xticks(indexcov,nameindex,size=20,name='monospace')
            pl.yticks(indexcov,nameindex,size=20,name='monospace')
            pl.annotate('FKP',(np.int(pow_bins//10),np.int(pow_bins/2.5)),fontsize=20)
            pl.annotate('Multi-tracer',(np.int(pow_bins//2),np.int(pow_bins//10)),fontsize=20)
            pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
            pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
            pl.colorbar()
            pl.savefig(dir_figs + '/' + handle_estimates + '_2Dcorr_' + str(nt) + '_' + str(ntp) + '.pdf')
            pl.close('all')


# Marker: open circle
mymark=[r'$\circ$']
smoothfilt = np.ones(5)/5.0
kphsmooth = np.convolve(kph,smoothfilt,mode='valid')

# Plot relative errors (diagonal of covariance)
# of the monopoles and quadrupoles for SINGLE tracers
# Use the convolved estimator for the variances/covariances
for nt in range(ntracers):
    tcfkp = np.convolve(Theor_Cov_FKP[nt]/(small+P0_fkp_mean[nt]**2),smoothfilt,mode='valid')
    pl.plot(kphsmooth,np.sqrt(np.abs(tcfkp)),'r-',linewidth=0.5)
    fkpcov= np.sqrt(np.abs(np.diagonal(relcov0_FKP[nt,nt])))
    fkpcov2= np.sqrt(np.abs(np.diagonal(relcov2_FKP[nt,nt])))
    label = str(nt)
    pl.semilogy(kph,fkpcov,marker='x',color='r',linestyle='', label = 'FKP Mono Tracer ' + label)
    pl.semilogy(kph,fkpcov2,marker='+',color='g',linestyle='', label = 'FKP Quad Tracer' + label)
    #tcmt = np.convolve(Theor_Cov_MT[nt,nt]/(small+P0_mean[nt]**2),smoothfilt,mode='valid')
    #pl.semilogy(kphsmooth,np.abs(tcmt),'k-',linewidth=2.5)
    mtcov = np.sqrt(np.abs(np.diagonal(relcov0_MT[nt,nt])))
    mtcov2 = np.sqrt(np.abs(np.diagonal(relcov2_MT[nt,nt])))
    pl.semilogy(kph,mtcov,marker='o',color='k',linestyle='', label = 'MT Mono Tracer ' + label)
    pl.semilogy(kph,mtcov2,marker='.',color='b',linestyle='', label = 'MT Quad Tracer ' + label)
    pl.legend()
    pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
    pl.ylabel(r'$\sigma [P^{(0)}]/[P^{(0)}]$ , $\sigma [P^{(2)}]/[P^{(2)}] $',fontsize=14)
    thistitle = 'Variance of tracer ' + str(nt)
    pl.title(thistitle,fontsize=16)
    pl.savefig(dir_figs + '/' + handle_estimates + '_sigmas_' + str(nt+1) + '.pdf')
    pl.close('all')


# Plot relative errors (diagonal of covariance)
# of the monopoles and quadrupoles for the CROSS-COVARIANCE between tracers
if plot_all_cov:
    if ntracers > 1:
        for nt in range(ntracers):
            for ntp in range(nt+1,ntracers):
                fkpcov= np.diagonal(relcov0_FKP[nt,ntp])
                fkpcov2= np.diagonal(relcov2_FKP[nt,ntp])
                pl.semilogy(kph,np.abs(fkpcov),marker='x',color='r',linestyle='')
                pl.semilogy(kph,np.abs(fkpcov2),marker='+',color='g',linestyle='')
                mtcov = np.diagonal(relcov0_MT[nt,ntp])
                mtcov2 = np.diagonal(relcov2_MT[nt,ntp])
                pl.semilogy(kph,np.abs(mtcov),marker='o',color='k',linestyle='')
                pl.semilogy(kph,np.abs(mtcov2),marker='.',color='b',linestyle='')
                ymin, ymax = 0.00000001, 10.0
                pl.xlim([kph[2],kph[-2]])
                pl.ylim([ymin,ymax])
                pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
                pl.ylabel('Cross-covariances (relative)',fontsize=12)
                thistitle = 'Cross-covariance of tracers ' + str(nt) + ' , ' + str(ntp)
                pl.title(thistitle,fontsize=16)
                pl.savefig(dir_figs + '/'  + handle_estimates + '_cross_cov_' + str(nt+1) + str(ntp+1) +'.pdf')
                pl.close('all')


# Plot relative errors (diagonal of covariance) of the RATIOS
# between monopoles and quadrupoles between tracers.
# N.B.: we are using the DEconcolved spectra for these error estimations!
if plot_all_cov:
    ntcount=0
    if ntracers > 1:
        for nt in range(ntracers):
            for ntp in range(nt+1,ntracers):
                fkpcov= np.sqrt(np.diagonal(fraccov0_FKP[ntcount]))
                fkpcov2= np.sqrt(np.diagonal(fraccov2_FKP[ntcount]))
                pl.semilogy(kph,fkpcov,marker='x',color='r',linestyle='')
                pl.semilogy(kph,fkpcov2,marker='+',color='g',linestyle='')
                mtcov = np.sqrt(np.diagonal(fraccov0_MT[ntcount]))
                mtcov2 = np.sqrt(np.diagonal(fraccov2_MT[ntcount]))
                pl.semilogy(kph,mtcov,marker='o',color='k',linestyle='')
                pl.semilogy(kph,mtcov2,marker='.',color='b',linestyle='')
                ymin, ymax = np.min(mtcov)*0.5, 10.0
                pl.xlim([kph[2],kph[-5]])
                pl.ylim([ymin,ymax])
                pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
                pl.ylabel(r'$\sigma(P_\mu/P_\nu)$',fontsize=12)
                thistitle = 'Relative variance of the ratios between tracers ' + str(nt+1) + ' , ' + str(ntp+1)
                pl.title(thistitle,fontsize=16)
                pl.savefig(dir_figs + '/' + handle_estimates + '_frac_cov_' + str(nt) + str(ntp) +'.pdf')
                pl.close('all')
                ntcount = ntcount + 1


# Plot DECONVOLVED estimates along with theory and data
# Monopole only
pl.xscale('log')
pl.yscale('log')
ylow=np.median((kph*P0_mean_dec)[0,-10:])*0.2
yhigh=np.mean(np.abs((kph*P0_mean_dec)[-1,2:10]))*5.0
xlow=0.99*kph[2]
xhigh=1.01*kph[-1]
pl.ylim([ylow,yhigh])
pl.xlim([xlow,xhigh])
# Plot total effective power spectrum: theory, model (box) and data
s0=np.sqrt(np.diag(cov_Pt_MT))
p0plus = np.minimum(yhigh,np.abs(kph*(P0tot_model + s0)))
p0minus = np.maximum(ylow,np.abs(kph*(P0tot_model - s0)))
pl.fill_between( kph, p0minus, p0plus, color = 'k', alpha=0.15)
pl.plot(kph, kph*P0tot_model, color='k', linestyle='-', linewidth=0.4)
# plot means of sims
pl.plot(kph, kph*P0tot_mean_MT, color='k', linestyle='--', linewidth=0.5)
pl.plot(kph, kph*P0tot_mean_FKP, color='k', linestyle='--', linewidth=0.2)
# plot data for total effective power
pl.plot(kph, kph*P0tot_MT[0], color='k', marker='.', linestyle='none')
pl.plot(kph, kph*P0tot_FKP[0], color='k', marker='x', linestyle='none')
for nt in range(ntracers):
    color1=mycolor[nt]
    # plot error bars as filled regions
    s0=np.sqrt(np.diag(relcov0_MT[nt,nt]))
    p0plus = np.minimum(yhigh,np.abs(np.abs(kph*P0_model[nt])*(1.0 + s0)))
    p0minus = np.maximum(ylow,np.abs(np.abs(kph*P0_model[nt])*(1.0 - s0)))
    pl.fill_between( kph, p0minus, p0plus, color = color1, alpha=0.15)
    # plot means of sims
    pl.plot(kph, kph*effbias[nt]**2*P0_fkp_mean_dec[nt], color=color1, linestyle='-', linewidth=0.2)
    pl.plot(kph,-kph*effbias[nt]**2*P0_fkp_mean_dec[nt], color=color1, linestyle='-.', linewidth=0.2)
    pl.plot(kph, kph*P0_mean_dec[nt], color=color1, linestyle='-', linewidth=0.5)
    pl.plot(kph,-kph*P0_mean_dec[nt], color=color1, linestyle='-.', linewidth=0.5)
    # Plot theory
    pl.plot(kph, kph*P0_model[nt], color=color1, linestyle='-', linewidth=1)
    # plot data -- markers
    pl.plot(kph, kph*effbias[nt]**2*P0_fkp_dec[0,nt], color=color1, marker='x', linestyle='none')
    pl.plot( kph, kph*P0_data_dec[0,nt], color=color1, marker='.', linestyle='none')

pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$k \, P^{(0)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Deconvolved spectra - $P_i^{(0)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_P0_deconv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')



# Plot DECONVOLVED estimates along with theory and data
# Monopole and quadrupole
pl.xscale('log')
pl.yscale('log')
ylow=np.median(P0_mean_dec[0,-10:])*0.2
yhigh=np.mean(np.abs(P0_mean_dec[:,2:10]))*5.0
xlow=0.99*kph[2]
xhigh=1.01*kph[-1]
pl.ylim([ylow,yhigh])
pl.xlim([xlow,xhigh])
for nt in range(ntracers):
    color1=mycolor[nt]
    # plot error bars as filled regions
    s0=np.sqrt(np.diag(relcov0_MT[nt,nt]))
    s2=np.sqrt(np.diag(relcov2_MT[nt,nt]))
    p0plus = np.minimum(yhigh,np.abs(P0_data_dec[0,nt]*(1.0 + s0)))
    p0minus = np.maximum(ylow,np.abs(P0_data_dec[0,nt]*(1.0 - s0)))
    p2plus = np.minimum(yhigh,np.abs(P2_data_dec[0,nt]*(1.0 + s2)))
    p2minus = np.maximum(ylow,np.abs(P2_data_dec[0,nt]*(1.0 - s2)))
    pl.fill_between( kph, p0minus, p0plus, color = color1, alpha=0.15)
    pl.fill_between( kph, p2minus, p2plus, color = color1, alpha=0.15)
    # plot means of sims
    pl.plot(kph, effbias[nt]**2*P0_fkp_mean_dec[nt], color=color1, linestyle='-', linewidth=0.2)
    pl.plot(kph,-effbias[nt]**2*P0_fkp_mean_dec[nt], color=color1, linestyle='-.', linewidth=0.2)
    pl.plot(kph,-effbias[nt]**2*P2_fkp_mean_dec[nt], color=color1, linestyle='-.',linewidth=0.2)
    pl.plot(kph, effbias[nt]**2*P2_fkp_mean_dec[nt], color=color1, linestyle='-', linewidth=0.2)
    pl.plot(kph, P0_mean_dec[nt], color=color1, linestyle='-', linewidth=0.5)
    pl.plot(kph,-P0_mean_dec[nt], color=color1, linestyle='-.', linewidth=0.5)
    pl.plot(kph,-P2_mean_dec[nt], color=color1, linestyle='-.',linewidth=0.5)
    pl.plot(kph, P2_mean_dec[nt], color=color1, linestyle='-', linewidth=0.5)
    # Plot theory
    pl.plot(kph, P0_model[nt], color=color1, linestyle='-', linewidth=1)
    pl.plot(kph, P2_model[nt], color=color1, linestyle='-', linewidth=1)
    # plot data -- markers
    pl.plot(kph, effbias[nt]**2*P0_fkp_dec[0,nt], color=color1, marker='x', linestyle='none')
    pl.plot(kph, effbias[nt]**2*P2_fkp_dec[0,nt], color=color1, marker='x', linestyle='none')
    pl.plot(kph,-effbias[nt]**2*P2_fkp_dec[0,nt], color=color1, marker='1', linestyle='none')
    pl.plot( kph, P0_data_dec[0,nt], color=color1, marker='.', linestyle='none')
    pl.plot( kph, P2_data_dec[0,nt], color=color1, marker='.', linestyle='none')
    pl.plot( kph, -P2_data_dec[0,nt], color=color1, marker='v', linestyle='none')

pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(\ell)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Deconvolved spectra - $P_i^{(0)} (k)$ , $P_i^{(2)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_deconv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')




# Plot CONVOLVED spectra; same limits as above
# Monopole and quadrupole
pl.xscale('log')
pl.yscale('log')
for nt in range(ntracers):
    color1=mycolor[nt]
    # plot error bars as filled regions
    s0=np.sqrt(np.diag(relcov0_MT[nt,nt]))
    s2=np.sqrt(np.diag(relcov2_MT[nt,nt]))
    p0plus = np.minimum(yhigh,np.abs(P0_data[0,nt]*(1.0 + s0)))
    p0minus = np.maximum(ylow,np.abs(P0_data[0,nt]*(1.0 - s0)))
    p2plus = np.minimum(yhigh,np.abs(P2_data[0,nt]*(1.0 + s2)))
    p2minus = np.maximum(ylow,np.abs(P2_data[0,nt]*(1.0 - s2)))
    pl.fill_between( kph, p0minus, p0plus, color = color1, alpha=0.15)
    pl.fill_between( kph, p2minus, p2plus, color = color1, alpha=0.15)
    # plot means of sims
    pl.plot(kph, effbias[nt]**2*P0_fkp_mean[nt], color=color1, linestyle='-', linewidth=0.2)
    pl.plot(kph,-effbias[nt]**2*P0_fkp_mean[nt], color=color1, linestyle='-.', linewidth=0.2)
    pl.plot(kph,-effbias[nt]**2*P2_fkp_mean[nt], color=color1, linestyle='-.',linewidth=0.2)
    pl.plot(kph, effbias[nt]**2*P2_fkp_mean[nt], color=color1, linestyle='-', linewidth=0.2)
    pl.plot(kph, P0_mean[nt], color=color1, linestyle='-', linewidth=0.5)
    pl.plot(kph,-P0_mean[nt], color=color1, linestyle='-.', linewidth=0.5)
    pl.plot(kph,-P2_mean[nt], color=color1, linestyle='-.',linewidth=0.5)
    pl.plot(kph, P2_mean[nt], color=color1, linestyle='-', linewidth=0.5)
    # Plot theory
    pl.plot(kph, P0_model[nt], color=color1, linestyle='-', linewidth=1)
    pl.plot(kph, P2_model[nt], color=color1, linestyle='-', linewidth=1)
    # plot FKP data
    pl.plot(kph, effbias[nt]**2*P0_fkp[0,nt], color=color1, marker='x', linestyle='none')
    pl.plot(kph, effbias[nt]**2*P2_fkp[0,nt], color=color1, marker='x', linestyle='none')
    pl.plot(kph, -effbias[nt]**2*P2_fkp[0,nt], color=color1, marker='1', linestyle='none')
    # plot MT data
    pl.plot( kph, P0_data[0,nt], color=color1, marker='.', linestyle='none')
    pl.plot( kph, P2_data[0,nt], color=color1, marker='.', linestyle='none')
    pl.plot( kph, -P2_data[0,nt], color=color1, marker='v', linestyle='none')

pl.ylim([ylow,yhigh])
pl.xlim([xlow,xhigh])
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(\ell)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Convolved spectra - $\hat{P}_i^{(0)} (k)$ , $ \hat{P}_i^{(2)} (k)$ ',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_conv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')


# Error bar plots
# Plot MT estimates along with theory -- convolved spectra
pl.xscale('log')
pl.yscale('log')
ylow=np.median(P0_mean_dec[0,-10:])*0.1
yhigh=np.mean(np.abs(P0_mean_dec[:,2:10]))*3.0

for nt in range(ntracers):
    gk = 1.+0.01*(nt-1.5)
    # Monopole
    color1=mycolor[nt]
    p = P0_data[0,nt]
    errp = np.sqrt(np.diag(np.cov(P0_data[1:,nt].T)))
    label = str(nt)
    pl.errorbar(gk*kph,p,errp,color=color1,linestyle='None', marker='s', markersize=3,capsize=3, label = 'Monopole Tracer ' + label)
    #pl.plot(kph,gp*P0_model[nt],color=color1,linewidth=1.2)
    pl.plot(kph,P0_mean[nt],color=color1,linewidth=0.6)
    # Quadrupole
    p = P2_data[0,nt]
    errp = np.sqrt(np.diag(np.cov(P2_data[1:,nt].T)))
    pl.errorbar(gk*kph,p,errp,color=color1,linestyle='None',marker = '^',markersize=3,capsize=3, label = 'Quadrupole Tracer ' + label)
    #pl.plot(kph,gp*P2_model[nt],color=color1,linewidth=1.2)
    pl.plot(kph,P2_mean[nt],color=color1,linewidth=0.6)

pl.legend()
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(\ell)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Convolved spectra - $P_i^{(0)} (k)$ , $P_i^{(2)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_errbar_conv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')


# Plot estimates along with theory -- deconvolved spectra
pl.xscale('log')
pl.yscale('log')
ylow=np.median(P0_mean_dec[0,-10:])*0.1
yhigh=np.mean(np.abs(P0_mean_dec[:,2:10]))*3.0
for nt in range(ntracers):
    gp = 1.+0.02*(nt-1.5)
    gk = 1.+0.01*(nt-1.5)
    # Monopole
    color1=mycolor[nt]
    p = gp*P0_data_dec[0,nt]
    errp = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt].T)))
    pl.errorbar(gk*kph,p,errp,color=color1,linestyle='None',linewidth=0.6, marker = 's', markersize = 3, capsize=3,label='MT - Monopole')
    pl.plot(kph,gp*P0_model[nt],color=color1,linewidth=0.4)
    # Quadrupole
    p = gp*P2_data_dec[0,nt]
    errp = np.sqrt(np.diag(np.cov(P2_data[1:,nt].T)))
    pl.errorbar(gk*kph,p,errp,color=color1,linestyle='None',linewidth=0.6, marker = '^', markersize=3,capsize=3,label='MT - Quadrupole')
    pl.plot(kph,gp*P2_model[nt],color=color1,linewidth=0.4)

pl.legend()
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(\ell)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Deconvolved spectra - $P_i^{(0)} (k)$ , $P_i^{(2)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_errbar_deconv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')


# Plot estimates along with theory -- deconvolved spectra
# Monopole only
pl.xscale('log')
pl.yscale('log')
for nt in range(ntracers):
    gk = 1.+0.01*(nt-1.5)
    # Monopole
    color1=mycolor[nt]
    p = np.abs(P0_data_dec[0,nt])
    errp = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt].T)))
    pl.errorbar(gk*kph,p,errp,color=color1,linestyle='None',linewidth=0.6, marker = 's', markersize = 3, capsize=3, label = 'MT Tracer ' + str(nt))
    pl.plot(kph,P0_model[nt],color=color1,linewidth=0.4, label = 'Model Tracer ' + str(nt))

pl.legend()
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(0)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
pl.title(r'Deconvolved spectra - $P_i^{(0)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_errbar_P0_deconv_kbar=' + str(kph_central) + '.pdf')
pl.close('all')


# Plot the ratios of the spectra with respect to the theoretical expectation
#pl.xscale('log', nonposy='clip')
pl.xscale('log')
#pl.yscale('log', nonposy='clip')
ylow = 0
yhigh= 2*ntracers + 1.

for nt in range(ntracers):
    color1=mycolor[nt]
    s0=np.sqrt(np.diag(relcov0_MT[nt,nt]))
    s2=np.sqrt(np.diag(relcov2_MT[nt,nt]))
    pp=2*nt+P0_mean_dec[nt]*(1.0 + s0)/(P0_model[nt])
    pm=2*nt+P0_mean_dec[nt]*(1.0 - s0)/(P0_model[nt])
    p0plus = np.minimum(yhigh,pp)
    p0minus = np.maximum(ylow,pm)
    pp=2*nt+1+P2_mean_dec[nt]*(1.0 + s2)/(P2_model[nt])
    pm=2*nt+1+P2_mean_dec[nt]*(1.0 - s2)/(P2_model[nt])
    p2plus = np.minimum(yhigh,pp)
    p2minus = np.maximum(ylow,pm)
    pl.fill_between( kph, p0minus, p0plus, color = color1, alpha=0.15)
    pl.fill_between( kph, p2minus, p2plus, color = color1, alpha=0.15)
    pl.plot(kph,2*nt+P0_mean_dec[nt]/(P0_model[nt]),color=color1,linestyle='-')
    pl.plot(kph,2*nt+effbias[nt]**2*P0_fkp_mean_dec[nt]/(P0_model[nt]),color=color1,linestyle='--')
    pl.plot(kph,2*nt+P0_data_dec[0,nt]/(P0_model[nt]),color=color1,marker='.', linestyle='none')
    pl.plot(kph,2*nt+1+P2_mean_dec[nt]/(P2_model[nt]),color=color1,linestyle='-')
    pl.plot(kph,2*nt+1+effbias[nt]**2*P2_fkp_mean_dec[nt]/(P2_model[nt]),color=color1,linestyle='--')
    pl.plot(kph,2*nt+1+P2_data_dec[0,nt]/(P2_model[nt]),color=color1,marker='x', linestyle='none')

pl.ylim([ylow,yhigh])
pl.xlim([xlow,xhigh])
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{\ell, D} (k)/P^{\ell, T}_i $',fontsize=14)
pl.title('Data/Theory',fontsize=16)
pl.savefig(dir_figs + '/' + handle_estimates + '_relative_errors_kbar=' + str(kph_central) + '.pdf')
pl.close('all')



#NEW FIGURES, stored in /selected directory
if not os.path.exists(dir_figs + '/selected'):
    os.makedirs(dir_figs + '/selected')

        
if plot_all_cov:
    # Plot 2D correlation of Ptot and ratios
    nblocks=1+ntracers*(ntracers-1)//2
    indexcov = np.arange(0,nblocks*num_binsk,np.int(num_binsk//4))
    nameindex = nblocks*[str(0.001*np.round(1000*kin)) for kin in kph[0:-1:np.int(num_binsk//4)]]
    onesk=np.diag(np.ones((nblocks*num_binsk)))
    dF=np.sqrt(np.abs(np.diag(cov_Pt_ratios_FKP)))
    dM=np.sqrt(np.abs(np.diag(cov_Pt_ratios_MT)))
    dF2 = small + np.outer(dF,dF)
    dM2 = small + np.outer(dM,dM)
    fullcov = np.tril(np.abs(cov_Pt_ratios_FKP)/dF2) + np.triu(np.abs(cov_Pt_ratios_MT.T)/dM2) - onesk
    pl.imshow(fullcov,origin='lower',interpolation='none')
    pl.title("Covariance of total effective power spectrum and ratios of spectra (monopoles only)")
    pl.xticks(indexcov,nameindex,size=6,name='monospace')
    pl.yticks(indexcov,nameindex,size=8,name='monospace')
    pl.annotate('FKP',(np.int(pow_bins/5.),2*pow_bins),fontsize=20)
    pl.annotate('Multi-tracer',(2*pow_bins,np.int(pow_bins/5.)),fontsize=20)
    pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
    pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=20)
    pl.colorbar()
    pl.savefig(dir_figs + '/' +'selected/' + handle_estimates + '_2D_tot_ratios_corr_newformat.pdf')
    pl.close('all')

    # Plot 2D correlation coefficients
    num_binsk0 = 26
    kph0=kph[:num_binsk0]
    indexcov = np.arange(0,num_binsk0+1,np.int(num_binsk0//5))
    #nameindex = [str(0.001*np.round(1000*kin)) for kin in kph0[0:-1:np.int(num_binsk0/6)]]
    nameindex = [str(0.001*np.round(1000*kin))[0:5] for kin in kph0[0:num_binsk0:np.int(num_binsk0/5)]]
    onesk=np.diag(np.ones(num_binsk))
    for nt in range(ntracers):
        for ntp in range(nt,ntracers):
            kk = np.outer(kph,kph)
            FKPcov=relcov0_FKP[nt,ntp]
            MTcov=relcov0_MT[nt,ntp]
            dF=np.sqrt(np.abs(np.diag(FKPcov)))
            dM=np.sqrt(np.abs(np.diag(MTcov)))
            FKPcorr=FKPcov/(small+np.outer(dF,dF))
            MTcorr=MTcov/(small+np.outer(dM,dM))
            #fullcov = np.tril(np.abs(FKPcorr)) + np.triu(np.abs(MTcorr.T)) - onesk
            fullcov = np.tril(FKPcorr) + np.triu(MTcorr.T) - onesk  
            thistitle = 'Corr(P_' + str(nt) + ',P_' + str(ntp) + ') '
            pl.imshow(fullcov,extent=[0,26,0,26],origin='lower',interpolation='nearest',cmap='jet')
            #pl.title(thistitle)
            #print(indexcov,nameindex)   
            pl.xticks(indexcov,nameindex,size=12,name='monospace')
            pl.yticks(indexcov,nameindex,size=12,name='monospace')
            pl.annotate(r'$\bf{FKP}$',(np.int(len(kph0)//10),np.int(len(kph0)/2.5)),fontsize=28, color='white')
            pl.annotate(r'$\bf{MTOE}$',(np.int(len(kph0)//2),np.int(len(kph0)//10)),fontsize=28,color='white')
            pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=12)
            pl.ylabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=12)
            #pl.xlim([0.,0.4])
            #pl.ylim([0.,0.4])
            pl.colorbar()
            pl.savefig(dir_figs + '/' + 'selected/' + handle_estimates + '_2Dcorr_' + str(nt) + '_' + str(ntp) + 'newformat.pdf')
            pl.close('all')


        
            
# Marker: open circle
mymark=[r'$\circ$']
smoothfilt = np.ones(5)/5.0
kphsmooth = np.convolve(kph,smoothfilt,mode='valid')
    
    
# new plot!!! same as above but for MT/FKP gain
for nt in range(ntracers):
    #tcfkp = np.convolve(Theor_Cov_FKP[nt]/(small+P0_fkp_mean[nt]**2),smoothfilt,mode='valid')
    #pl.plot(kphsmooth,np.sqrt(np.abs(tcfkp)),'r-',linewidth=0.5)
    color1=mycolor[nt]
    fkpcov= np.sqrt(np.abs(np.diagonal(relcov0_FKP[nt,nt])))
    mtcov = np.sqrt(np.abs(np.diagonal(relcov0_MT[nt,nt])))
    pl.plot(kph,(fkpcov/fkpcov)*1.0,color='k',linestyle='--')
    pl.plot(kph,mtcov/fkpcov,marker='o',color=color1,markersize=5,linestyle='-',label='Tracer '+str(nt+1))
ymin, ymax = 0.0,1.3#1.2
#pl.xlim([kph[2],kph[-2]])
pl.xlim([kph[0],0.3])
pl.ylim([ymin,ymax])
#pl.text(0.3,1.07,"LC-selection",fontsize=14)
#pl.text(0.3,1.17,"Mass selection",fontsize=14)
pl.legend(loc="upper right",fontsize=7,frameon=False)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$\sigma [P^{MT}]/[P^{MT}] / \sigma [P^{FKP}]/[P^{FKP}] $',fontsize=14)
#thistitle = 'Gain on variance of tracer ' + str(nt)
#pl.title(thistitle,fontsize=16)
pl.savefig(dir_figs + '/' + 'selected/' + handle_estimates  +'_sigmas_gain.pdf')
pl.close('all')    



# new plot!!! same as above but for MT/FKP gain

for nt in range(ntracers):
    #tcfkp = np.convolve(Theor_Cov_FKP[nt]/(small+P0_fkp_mean[nt]**2),smoothfilt,mode='valid')
    #pl.plot(kphsmooth,np.sqrt(np.abs(tcfkp)),'r-',linewidth=0.5)
    color1=mycolor[nt]
    fkpcov= np.sqrt(np.abs(np.diagonal(relcov0_FKP[nt,nt])))
    mtcov = np.sqrt(np.abs(np.diagonal(relcov0_MT[nt,nt])))
    pl.plot(kph,(fkpcov/fkpcov)*1.0,color='k',linestyle='--')
    pl.plot(kph,mtcov/fkpcov,marker='o',color=color1,markersize=5,linestyle='-',label='Tracer '+str(nt+1))
ymin, ymax = 0.0,1.3#1.2
#pl.xlim([kph[2],kph[-2]])
pl.xlim([kph[0],0.3])
pl.ylim([ymin,ymax])
#pl.text(0.3,1.07,"LC-selection",fontsize=14)
#pl.text(0.3,1.17,"LC-selection",fontsize=14)
pl.legend(loc="upper right",fontsize=8,frameon=False)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$\sigma [P^{MT}]/[P^{MT}] / \sigma [P^{FKP}]/[P^{FKP}] $',fontsize=14)
#thistitle = 'Gain on variance of tracer ' + str(nt)
#pl.title(thistitle,fontsize=16)
pl.savefig(dir_figs + '/' + 'selected/' + handle_estimates + '_sigmas_gain_2.pdf')
pl.close('all')    
    


##color_code = ['blue','cyan','lime','orange','red','maroon','pink','green','black','grey','navy','salmon','purple']

if plot_all_cov:
    ntcount=0
    if ntracers > 1:
        for nt in range(ntracers):
            for ntp in range(nt+1,ntracers):
                #color1=color_code[ntcount]
                fkpcov= np.sqrt(np.diagonal(fraccov0_FKP[ntcount]))
                mtcov = np.sqrt(np.diagonal(fraccov0_MT[ntcount]))
                pl.plot(kph,(fkpcov/fkpcov)*1.0,color='k',linestyle='--')
                pl.plot(kph,mtcov/fkpcov,marker='o',markersize = 5.0,linestyle='-', label='ratio '+str(nt+1)+'-'+str(ntp+1))
                ntcount = ntcount + 1

#pl.legend(loc="upper right",fontsize=6,frameon=False)
ymin, ymax = 0.0,1.5
#pl.xlim([kph[2],kph[-5]])
pl.xlim([kph[0],0.3])
pl.ylim([ymin,ymax])
#pl.text(0.3,1.07,"LC-selection",fontsize=14)
pl.xlabel(r'$k\,[h$Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$\sigma(P_\mu/P_\nu)_{MT}/\sigma(P_\mu/P_\nu)_{FKP}$',fontsize=12)
#thistitle = 'MT/FKP gain on ratios of spectra for ' + str(nt+1) + ' , ' + str(ntp+1)
#pl.title(thistitle,fontsize=16)
pl.savefig(dir_figs + '/' + 'selected/' +handle_estimates + '_frac_cov_gain.pdf')
pl.close('all')
               

  
# Plot estimates along with theory -- deconvolved spectra
# Monopole only
pl.xscale('log')
pl.yscale('log')
#ylow=np.median(P0_mean_dec[0,-10:])*0.1
#yhigh=np.mean(np.abs(P0_mean_dec[:,2:10]))*3.0
ylow=400.
yhigh=120000.


for nt in range(ntracers):
    gp = 1.#+0.02*(nt-1.5)
    gk = 1.#+0.01*(nt-1.5)
    gp2 = 1.#+0.04*(nt-1.5)
    gk2 = 1.#+0.03*(nt-1.5)
    # Monopole
    color1=mycolor[nt]
    p = np.abs(gp*P0_data_dec[0,nt])
    p_mean = np.abs(gp*np.mean(P0_data_dec[:,nt,:],axis=0))
    p2 = gp2*effbias[nt]**2*P0_fkp_dec[0,nt]
    # print(P0_fkp[:,nt].shape)
    p2_mean = gp2*effbias[nt]**2*(np.mean(P0_fkp_dec[:,nt,:],axis=0))
    #p2 = gp*effbias[nt]**2*P0_fkp[0,nt]
    errp = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt].T)))
    errp2 = np.sqrt(np.diag(np.cov(effbias[nt]**2*P0_fkp_dec[1:,nt].T)))
    if nt==0:
        pl.errorbar(gk*kph,p_mean,errp,color=color1,linestyle ='None', marker='^', capsize = 3, markersize = 3,elinewidth=1.2, label='MT (mean of mocks)')
        #pl.errorbar(gk2*kph,p2,errp2,color=color1,linestyle ='None', marker='x',ms=3.,elinewidth=1.2, label = 'FKP')
        pl.errorbar(gk2*kph,p2_mean,errp2,color=color1,linestyle ='None', marker='x',capsize=3, markersize = 4,elinewidth=1.2, label = 'FKP (mean of mocks)')
        pl.plot(kph,gp*P0_model[nt],color=color1,linewidth=1.,label='Theoretical model')
        ##pl.plot(kph,p_mean,color=color1,linewidth=2.,linestyle='--',label='Mean of mocks')
    else:
        pl.errorbar(gk*kph,p_mean,errp,color=color1,linestyle ='None', marker='^', capsize=3, markersize=3,elinewidth=1.2)
        #pl.errorbar(gk2*kph,p2,errp2,color=color1,linestyle ='None', marker='x',ms=3.,elinewidth=1.2)
        pl.errorbar(gk2*kph,p2_mean,errp2,color=color1,linestyle ='None', marker='x',capsize=3,markersize=3,elinewidth=1.2)
        pl.plot(kph,gp*P0_model[nt],color=color1,linewidth=1.)
        ##pl.plot(kph,p_mean,color=color1,linewidth=2.,linestyle='--')
        
#pl.text(0.3,20000,r"$0.6<z<0.75$", fontsize = 12)
#pl.text(0.18,20000,"W4 field", fontsize = 12)
pl.legend(loc="lower left",fontsize=12.,frameon=False)
pl.xlabel(r'$k \, [h \, $Mpc$^{-1}]$',fontsize=14)
pl.ylabel(r'$P^{(0)}(k) \, [h^{-3} \, $Mpc$^3]$',fontsize=14)
#pl.title(r'Deconvolved spectra - $P_i^{(0)} (k)$',fontsize=16)
pl.savefig(dir_figs + '/' + 'selected/' + handle_estimates + '_errbar_P0_deconv_kbar=' + str(kph_central) + '_paper_new.pdf')
pl.close('all')







print('Figures created and saved in /fig .')

print ()
print('----------------------------------')
print ()



######################
# Now writing results to file
print('Writing results to /spectra/' + handle_estimates)

# Full dataset. Stack the spectra, then take covariances
p_stack = np.zeros((n_maps,2*ntracers*num_binsk),dtype="float32")
p_stack_FKP = np.zeros((n_maps,2*ntracers*num_binsk),dtype="float32")
p_theory = np.zeros((num_binsk,2*ntracers),dtype="float32")
p_data = np.zeros((num_binsk,2*ntracers),dtype="float32")

for nt in range(ntracers):
    p_stack[:,2*nt*num_binsk:(2*nt+1)*num_binsk] = P0_data_dec[:,nt]
    p_stack[:,(2*nt+1)*num_binsk:(2*nt+2)*num_binsk] = P2_data_dec[:,nt]
    p_stack_FKP[:,2*nt*num_binsk:(2*nt+1)*num_binsk] = effbias[nt]**2*P0_fkp_dec[:,nt]
    p_stack_FKP[:,(2*nt+1)*num_binsk:(2*nt+2)*num_binsk] = effbias[nt]**2*P2_fkp_dec[:,nt]
    p_theory[:,2*nt] = P0_model[nt]
    p_theory[:,2*nt+1] = P2_model[nt]
    p_data[:,2*nt] = P0_data_dec[0,nt]
    p_data[:,2*nt+1] = P2_data_dec[0,nt]


# Return bias to the computed cross-spectra (done in FKP) as well:
index=0
for i in range(ntracers):
    for j in range(i+1,ntracers):
        Cross0[:,index] = effbias[i]*effbias[j]*Cross0[:,index]
        Cross0_dec[:,index] = effbias[i]*effbias[j]*Cross0_dec[:,index]
        Cross2[:,index] = effbias[i]*effbias[j]*Cross2[:,index]
        Cross2_dec[:,index] = effbias[i]*effbias[j]*Cross2_dec[:,index]
        index += 1

Cross0_mean = np.mean(Cross0[1:],axis=0)
Cross2_mean = np.mean(Cross2[1:],axis=0)


# Export data
np.savetxt(dir_specs + '/' + handle_estimates + '_vec_k.dat',kph,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_P0_P2_theory.dat',p_theory,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_P0_P2_data.dat',p_data,fmt="%6.4f")

np.savetxt(dir_specs + '/' + handle_estimates + '_decP_k_data_MT.dat',p_stack.T,fmt="%6.2f")
np.savetxt(dir_specs + '/' + handle_estimates + '_decP_k_data_FKP.dat',p_stack_FKP.T,fmt="%6.2f")

np.savetxt(dir_specs + '/' + handle_estimates + '_nbar_mean.dat',nbarbar,fmt="%2.6f")
np.savetxt(dir_specs + '/' + handle_estimates + '_bias.dat',gal_bias,fmt="%2.3f")
np.savetxt(dir_specs + '/' + handle_estimates + '_effbias.dat',effbias,fmt="%2.3f")


# Saving cross spectra

np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P0_theory.dat',Cross_P0_theory.T,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P0_model.dat',Cross_P0_model.T,fmt="%6.4f")

np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P2_theory.dat',Cross_P2_theory.T,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P2_model.dat',Cross_P2_model.T,fmt="%6.4f")

Cross0_save=np.reshape(Cross0,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
Cross2_save=np.reshape(Cross2,(n_maps,ntracers*(ntracers-1)//2*pow_bins))

Cross0_dec_save=np.reshape(Cross0_dec,(n_maps,ntracers*(ntracers-1)//2*pow_bins))
Cross2_dec_save=np.reshape(Cross2_dec,(n_maps,ntracers*(ntracers-1)//2*pow_bins))

np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P0_data.dat',Cross0_save.T,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_dec_P0_data.dat',Cross0_dec_save.T,fmt="%6.4f")

np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_P2_data.dat',Cross2_save.T,fmt="%6.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_Cross_dec_P2_data.dat',Cross2_dec_save.T,fmt="%6.4f")


########################################
# Window function
wf = np.zeros((2*ntracers,num_binsk),dtype="float32")
for k in range(pow_bins):
    wf[:,k]  = np.hstack((winfun0[:,k],winfun2[:,k]))

# make sure that window function is not zero
wf[wf==0]=1.0
wf[np.isnan(wf)]=1.0
np.savetxt(dir_specs + '/' + handle_estimates + '_WinFun02_k_data.dat',wf.T,fmt="%3.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_WinFun_Cross0_k_data.dat',wf_c0,fmt="%3.4f")
np.savetxt(dir_specs + '/' + handle_estimates + '_WinFun_Cross2_k_data.dat',wf_c2,fmt="%3.4f")


####################################
# Checking cross-correlations
print()

index=0
for nt in range(ntracers):
    for ntp in range(nt+1,ntracers):
        print("Cross-spec of P^0 of tracers", nt, ntp, " -- mean/model, mean/theory")
        var = np.array([ effbias[nt]*effbias[ntp]*Cross0_mean_dec[index]/Cross_P0_model[index] , effbias[nt]*effbias[ntp]*Cross0_mean_dec[index]/Cross_P0_theory[index] ])
        print(var.T)
        index += 1
        print()

print()
index=0
for nt in range(ntracers):
    for ntp in range(nt+1,ntracers):
        print("Cross-spec of P^0 x Auto-spec of P^0 of tracers", nt, ntp, " -- P_ij^2[fkp]/P_i[MTOE] P_j[MTOE]")
        print(effbias[nt]**2*effbias[ntp]**2*Cross0_mean_dec[index]**2/P0_mean_dec[nt]/P0_mean_dec[ntp])
        print()
        index += 1

print()
index=0
for nt in range(ntracers):
    for ntp in range(nt+1,ntracers):
        print("Cross-spec of P^0 x Auto-spec of P^0 of tracers", nt, ntp, " -- P_ij^2[fkp]/P_i[FKP] P_j[FKP]")
        print(Cross0_mean_dec[index]**2/P0_fkp_mean_dec[nt]/P0_fkp_mean_dec[ntp])
        print()
        index += 1


####################################
# The measured monopoles and the flat-sky/theory monopoles are different; find relative normalization

normmonos_data = np.zeros(ntracers)
normmonos_mocks = np.zeros(ntracers)
chi2_red_data = np.zeros(ntracers)
chi2_red_mocks = np.zeros(ntracers)

for nt in range(ntracers):
    err = np.sqrt(np.diag(np.cov(P0_data_dec[1:,nt,kb_cut_min:kb_cut_max].T)))
    data = [ P0_data_dec[0,nt,kb_cut_min:kb_cut_max]/err , P0_theory[nt,kb_cut_min:kb_cut_max]/err ]
    this_norm, success = leastsq(residuals,1.0,args=data)
#    print weights_mono
    normmonos_data[nt] = np.sqrt(this_norm)
#    norm_monos[nt] = np.mean(np.sqrt(P0_data_dec[0,nt,kb_cut_min:kb_cut_max]/P0_mean_dec[nt,kb_cut_min:kb_cut_max]))
    chi2_red_data[nt] = np.sum(residuals(this_norm,data)**2)/(len(err)-1.)

    mock_data = [ P0_mean_dec[nt,kb_cut_min:kb_cut_max]/err , P0_theory[nt,kb_cut_min:kb_cut_max]/err ]
    this_norm, success = leastsq(residuals,1.0,args=mock_data)
#    print weights_mono
    normmonos_mocks[nt] = np.sqrt(this_norm)
#    norm_monos[nt] = np.mean(np.sqrt(P0_data_dec[0,nt,kb_cut_min:kb_cut_max]/P0_mean_dec[nt,kb_cut_min:kb_cut_max]))
    chi2_red_mocks[nt] = np.sum(residuals(this_norm,mock_data)**2)/(len(err)-1.)

print()
print ("Fiducial biases of the mocks, and best-fit values from theory (for updating values in the input file, or HOD)")
for nt in range(ntracers):
    print ('Tracer ', nt, ': fiducial bias = ', np.around(gal_bias[nt],3), ' ; best fit:', np.around(normmonos_mocks[nt]*gal_bias[nt],3), ' (chi^2 = ', np.around(chi2_red_mocks[nt],3), ')')
print()

try:
    data_bias
    print ("Fiducial biases of the data, and best-fit values from theory (for updating values in the input file, or HOD)")
    for nt in range(ntracers):
        print ('Tracer ', nt, ' -- data: fiducial bias = ', np.around(data_bias[nt],3), ' ; update this to:', np.around(normmonos_data[nt]*data_bias[nt],3), ' (chi^2 = ', np.around(chi2_red_data[nt],3), ')')
except:
    print ("Fiducial biases of the data, and best-fit values from theory (for updating values in the input file, or HOD)")
    for nt in range(ntracers):
        print ('Tracer ', nt, ' -- data: fiducial bias = ', np.around(gal_bias[nt],3), ' ; update this to:', np.around(normmonos_data[nt]*gal_bias[nt],3), ' (chi^2 = ', np.around(chi2_red_data[nt],3), ')')

print()
print ("Quick update bias (from MT estimator):")
for nt in range(ntracers):
    print (np.around(normmonos_mocks[nt]*gal_bias[nt],3))

sys.exit(-1)


