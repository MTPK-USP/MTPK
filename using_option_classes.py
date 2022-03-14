#The following two classes have the inttention to substitute the old program of inputs
from cosmo import cosmo #class to cosmological parameter
from code_options import code_parameters #class to code options 

##################################################
# COMOLOGY OPTIONS
##################################################

#This stantiates the class with default parameters
my_cosmology = cosmo() 
print('Default parameters:')
print(my_cosmology)
print()

#To change one or other parameter, do:
my_cosmology = cosmo(h = 0.72, n_s = 0.95)
print('Modified parameters:')
print(my_cosmology)
print()

#This returns a dictionary with all the default parameters
physical_options = my_cosmology.default_params
print('A dictionary with the parameters:')
print(physical_options)
print()

#Using the method cosmo_print
print('Using the method cosmo_print:')
print(my_cosmology.cosmo_print())
print()

#Example of local change
print('Example of local change:')
physical_options['h'] = 0.69
print(physical_options)
print()

#Testing methods of cosmos class
print()
print("Testing f_evolving(z = 1)", my_cosmology.f_evolving(1.0))
print()
print("Testing f_phenomenological", my_cosmology.f_phenomenological() )
print()
print("Testing H(z = 0)", my_cosmology.H(0, True) )
print()
print("Testing H(z = 0)", my_cosmology.H(0, False) )
print()
print("Testing cosmological distance: z = 0:", my_cosmology.comoving(0, True) )
print()
print("Testing cosmological distance: z = 0:", my_cosmology.comoving(0, False) )
print()
print("Testing cosmological distance: z = 1:", my_cosmology.comoving(1, True) )
print()
print("Testing cosmological distance: z = 1:", my_cosmology.comoving(1, False) )
print()
print('chi_h: z = 0:', my_cosmology.chi_h(0.0))
print()
print('chi_h: z = 1:', my_cosmology.chi_h(1.0))
print()

##################################################
# CODE OPTIONS
##################################################

#This stantiates the class with default parameters
print('Default parameters:')
my_code_options = code_parameters()
print(my_code_options)
print()

#To change one or other parameter, do:
my_code_options = code_parameters(cell_size = 42.0)
print('Modified parameters:')
print(my_code_options)
print()

#This returns a dictionary with all the default parameters
parameters_code = my_code_options.default_params
print('A dictionary with the parameters:')
print(parameters_code)
print()

#Using the method cosmo_print
print('Using the method parameters_print:')
print(my_code_options.parameters_print())
print()

#Example of local change
print('Example of local change:')
parameters_code['cell_size'] = 0.69
print(parameters_code)
print()
