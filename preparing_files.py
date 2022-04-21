import numpy as np

data_xyz = np.loadtxt('data/lightcone/Lightcone_xyz.dat')
data_rdz = np.loadtxt('data/lightcone/Lightcone_RA_DEC_z.dat')

redshift_range = [0, 0.3]
z_red = data_rdz[:, 2]
z_bins = np.linspace(redshift_range[0],redshift_range[1],50)
z_ctrs = 0.5*(z_bins[1:]+z_bins[:-1])

r = np.zeros((3, data_xyz.shape[0]))
r[0,:] = data_xyz[:, 0]
r[1,:] = data_xyz[:, 1]
r[2,:] = data_xyz[:, 2]
rad = np.linalg.norm(r, axis = 0)
rmin, rmax = np.min(rad), np.max(rad)
r_bins = np.linspace(rmin, rmax, 50)
dr = np.diff(r_bins)
r = 0.5*(r_bins[1:] + r_bins[:-1])

rz = np.histogram2d(z_red, rad, bins = (z_bins, r_bins))[0]
z_rad = np.sum(rz.T*z_ctrs,axis=1)/np.sum(rz.T,axis=1)

f_sky = 0.146447  # area in steradians (f_sky=1 is full sky)
volumes = f_sky * (4*np.pi) * r**2 * dr  # volume of shells in light cone

histo_rad = np.histogram(rad, bins = r_bins)[0]

num_dens = np.array([r, z_rad, histo_rad/volumes])

np.savetxt('data/lightcone/r_z_hmf_massbin.dat', num_dens.T)
