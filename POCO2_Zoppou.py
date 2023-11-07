# Test script of POCO2 model. 
#
# Ref: Zoppou et al (1997)
#
#

#
# Imports
#

import math
import numpy as np
from scipy import special
import OpSplit as ops
from importlib import reload

ops = reload(ops)

#
# Functions
#

speed = lambda u0, x: u0*x            # Advective speed
diffusivity = lambda D0, x: D0*x**2   # Diffusivity

def ic(c0,x0,x):
    """ Initial condition c(x,0)=0, x>x0
    """

    return np.where(x<=x0,c0,0.0)

#
# Parameters
#

u0 = 1
D0 = 0.02
c0 = 100
x0 = 1

#
# Grid
#

L = 20  # Length 
M = 80 # Number of physical grid cells
h = L / M  # Grid spacing (cell width)
N = M+2 # Number of computational grid cells

# Grid
#  -0.5  0  0.5  1  1.5  2  2.5                                                    20 20.5 21 21.5
#    |---x---|---x---|---x---|---x---|-........................-|---x---|---x---|---x---|---x---|
#  j=0   0   1   1   2   2   3   3   4                                              M  M+1 M+1 M+2



xb = np.linspace(-0.5,21.5,N+1) # Coordinates of grid cell boundaries

xc = 0.5*(xb[1:]+xb[0:-1])

# Coefficients
D = diffusivity(D0,xc)
A = speed(u0,xb)

# Integration parameters

ah=u0*xb.max()/h

K = 1/3.

T=2.  # Integration interval (days)

u_max = u0*L # Maximum advection speed

dt = 0.5*h/u_max  # Time step

J = int(T/round(dt,8)) # Number of time steps

dtout = 10   # Time step to print model status

Jout = math.floor(J/dtout)

# Echo parameters

print(f"Integration time: {T:f}")
print(f"Time step: {dt:f}")
print(f"Number of time steps: {J:d}")

# Initial condition

c_ini = ic(c0,x0,xc)

## Time loop

t = np.linspace(0.0, T, J)

# Solution arrays

c=np.zeros((J,N)) # Full solution
c_out = np.zeros((Jout,N)) # Solution at output time steps
t_out = np.zeros((Jout,)) # Time of output time steps

# Copy initial condition to solution array
c[0,:] = c_ini

n = 0

n_out = -1

for n in range(1, J):

    c[n,:] = ops.step_splitting(c[n-1,:],dt,M,h,D,np.maximum(A,u0),
                                p=[K,
                                   [c0*u0,0.0],
                                   [c0,0.0]])

    if n % dtout == 0:
        n_out += 1
         
        print(f"Max c= {c[n,1:-1].max()}")
        print(f"Min c= {c[n,1:-1].min()}")

        c_out[n_out,:] = c[n,:]
        t_out[n_out] = t[n]
        #printStatus(n,t[n],np.vstack((u[n,:], v[n,:])))
        #print(f"Remineralization rate max {r1.max()} min {r1.min()}")
        #print(f"Sediment oxygen: {e[n]}")

np.savetxt('time.txt',t_out)
np.savetxt('x.txt',xb)       # coordinates of grid cell boundaries
np.savetxt('concentration_Zoppou.txt',c_out)