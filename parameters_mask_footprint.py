########################################
# The fooprint is defined in terms of straight lines in RA, dec (both in RADIANS)
# Each line defines the boundary "above" which the mask is zero (no galaxies), as in:
#
#  l1 = a * RA + b * dec - c > 0   -->   then, mask=1 above the line l1, mask=0 below l1
# 
# NOTE: 0 <= RA <= 360 deg !!!
#
# Notice that the inverse (i.e., "below" the line) would be given by:
#
#  a * RA + b * dec - c < 0   ==  -a * RA - b * dec + c > 0
#
# i.e., just invert the sign of the constants for "below the line" in the sense above.
#
# So, if a point has RA and dec which satisfies this condition, then we are inside the allower region
# You can use any number of lines, and the allowed region will be the intersection of all inequalities, i.e:
# 
#  (l1 >0) & (l2 > 0) & (l3 > 0) & ... must ALL be satisfied. 
#
# Hence, the region must be convex.
########################################
import numpy as np

# 2.2 < RA < 2.8  and  0.3 < dec < 0.8
# 126.05 < RA < 160.4  and  17.19 < dec < 45.84
a_edge = [ 1.0  , -1.0  ,  0.0  ,  0.0  ]
b_edge = [ 0.0  ,  0.0  ,  1.0  , -1.0  ]
c_edge = [ 2.2  , -2.8  ,  0.3  , -0.8  ]

