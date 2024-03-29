import numpy as np

data = {}
for i in range(1):
    data[i] = np.loadtxt(f"data/lightcone/r_z_hmf_massbin.dat")
    #this file contains the radius, the redshift and the halo mass function for the specified mass bin

aux = np.zeros( (1, data[0].shape[0]) )

for i in range(1):
    aux[i] = data[i][:, 2]

new_data = np.zeros( (2, data[0].shape[0]) )

new_data[0] = data[0][:, 1]
new_data[1] = np.mean(aux, axis = 0)

#This file contains the redshift and the mean between the halo mass function for the different seeds
np.savetxt('data/lightcone/radial_selection_function.dat', new_data.T)

print('Done!')
