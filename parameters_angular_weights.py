########################################
#  Selection function (weights) parameters
#
#  For each tracer you will give a set of parameters that define how the 
#  selection function varies with RA & dec -- i.e., the weight of each position .
#  Here, the weight is defined such that:
#  n_final = weight * n_random
#
#  Parameters for quadratic fit of the weights
#  Remember: all in RA, dec, in units of RADIANS.
# 
#  Format: [ [tracer_1(RA),tracer_1(dec)] , [tracer_2(RA),tracer_2(dec)] , ... ]
#

pivot    = [ [ 2.4  , 0.5 ] , [ 2.6  ,  0.6 ] ]
lin_par  = [ [ 0.15 , 0.0 ] , [ 0.05 , -0.2 ] ]
quad_par = [ [-0.2  , 0.0 ] , [ 0.05 ,  0.1 ] ]
