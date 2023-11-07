#
# POCO2 - Particulate Organic Carbon (POC) and Dissolved Oxygen (O2) in Fjord Basins.
#
#
# 1. Introduction
# ---------------
#
# POCO2 models the coupled dynamics of POC and O2 in the basin of sill fjords. The equations solved are:
#
#         dc_1/dt + d(W.c_1)/dz = d/dz(-D.dc_1/dz) + r_1                                             (1)
#
#         dc_2/dt = d/dz(-D.dc_2/dz) + r_2,                                                          (2)
#
# 
# where c_1 and c_2 are the concentrations of POC and O2, respectively, W is the POC sinking velocity,
# D is the vertical diffusivity and r_1 and r_2 are the reaction terms of POC and O2 respectively. 
#
# 2. Numerics
# -----------
#
# 2.1  Solution method
#
# POCO2 is one-dimensional and models water column processes. The model solves (1) and (2) by the operator
# splitting technique, where each term is solved by a distinct numerical operator in a sequential manner,
# i.e., if c_i(n) is the concentration of species ci (i=1,2) at time step n, then c_i(n+1) is:
#
#        c_i(n+1) = [exp(dt/2.L1).exp(dt/2.L2).exp(L3).exp(dt/2.L2).exp(dt/2.L1)]c_i(n).             (3)
#
# Thus, we apply sequentially three numerical operators to c_i(n) – L1, L2 and L3 –, to find the solution
# at the next time step n+1. L1, L2, and L3 are operators that solve the advection, diffusion and reaction
# terms of (1,2) separately, i.e., the technique solves the following sub-problems in sequence: 
#
#    d(u_i)/dt = L1(u_i) = - d(W.u_i)/dz,     u_i(n) = c_i(n),     t in [n, n+1/2],                  (4)
#    d(v_i)/dt = L2(v_i) = d/dz(-D.dv_i/dz),  v_i(n) = u_i(n+1/2), t in [n, n+1/2],                  (5)
#    d(w_i)/dt = L3(w_i) = r_i,               w_i(n) = v_i(n+1/2), t in [n, n+1],                    (6)
#    d(v_i)/dt = L2(v_i) = d/dz(-D.dv_i/dz),  v_i(n+1/2) = w_i(n+1), t in [n+1/2, n+1],              (7)
#    d(u_i)/dt = L1(u_i) = d/dz(-D.dv_i/dz),  u_i(n+1/2) = v_i(n+1), t in [n+1/2, n+1],              (8)
#
# and c_i(n+1) = u_i(n+1). In the equations above (4-8), u_i, v_i and w_i are intermediate values of c_i
# obtained from the solution of the split equations. Note that for i=2, L1=1, i.e., advection does not  
# change the O2 concentration.
#
# The L1 and L2 operators are defined as follows: 
#
#   L1: 
#
#      Spatial scheme: flux form, third order upwind limited.
#      Temporal scheme: Runge-Kutta 4th order.
#
#   L2: 
#
#      Spatial scheme: 2nd order centred
#      Temporal scheme: implicit trapezoidal
#
# The L3 operator is a reaction scheme: TBA!!!
#
# 2.2 Numerical grid
#
# The computations are performed in a cell-centred grid. The grid setup is as follows, where h is the
# cell length.
#
#
# z=        H_s                                                                        H_b
# x=-h       0  h/2  h 3h/2 2h                                                         M.h
#    |---x---|---x---|---x---|---x---|-........................-|---x---|---x---|---x---|---x---|
#  j=0   0   1   1   2   2   3   3   4                                              M  M+1 M+1 M+2
#
# The water column between the sill depth (H_s) and the bottom (H_b) is divided in M cells of length h.
# The c_i's are stored at the cell centers x_j = (j-1/2)*h, and the fluxes are computed at the cell
# faces x_j=(j-1)*h. The j=0 and j=M+1 cell centers are ghost points used to manipulate boundary conditions.
#
# 3. Contents
# -----------
#
# This file contains the objects necessary to implement the operator splitting techinique as defined in 2.1.
#
#

import numpy as np
from importlib import reload
import L1
import L2 

L1 = reload(L1)
L2 = reload(L2)

# 4th order Finite difference formulas for left boundary

d1left = lambda c, h: (1/h)*(-25/12*c[0]+4*c[1]-3*c[2]+4/3*c[3]-1/4*c[4])
d2left = lambda c, h: (1/h**2)*(15/4*c[0]-77/6*c[1]+107/6*c[2]-13*c[3]+61/12*c[4]-5/6*c[5])
d3left = lambda c, h: (1/h**3)*(-49/8*c[0]+29*c[1]-461/8*c[2]+62*c[3]-307/8*c[4]+13*c[5]-15/8*c[6])
d4left = lambda c, h: (1/h**4)*(28/3*c[0]-111/2*c[1]+142*c[2]-1219/6*c[3]+176*c[4]-185/2*c[5]+82/3*c[6]-7/2*c[7])
		
# 4th order Finite difference formulas for right boundary

d1right = lambda c, h: (1/h)*(25/12*c[-1]-4*c[-2]+3*c[-3]-4/3*c[-4]+1/4*c[-5])
d2right = lambda c, h: (1/h**2)*(15/4*c[-1]-77/6*c[-2]+107/6*c[-3]-13*c[-4]+61/12*c[-5]-5/6*c[-6])
d3right = lambda c, h: (1/h**3)*(49/8*c[-1]-29*c[-2]+461/8*c[-3]-62*c[-4]+307/8*c[-5]-13*c[-6]+15/8*c[-7])
d4right = lambda c, h: (1/h**4)*(28/3*c[-1]-111/2*c[-2]+142*c[-3]-1219/6*c[-4]+176*c[-5]-185/2*c[-6]+82/3*c[-7]-7/2*c[-8])

# Boundary conditions for intermediate steps

def fL1left(f1,c,h,a,dt):
    """ Implements the left first intermediate BC. Khan & Liu (1995), eq. 40.
    """

    a1 = -0.5*dt*a*d1left(c,h)

    a2 = a**2*dt**2/8*d2left(c,h)
    #print(d1left(c,h))
    #print(d2left(c,h))
    #print(a1)
    #print(a2)

    return f1 + a1 + a2

def fL2left(f11, c, D, h, dt):
    """ Implements the left second intermediate BC. Khan & Liu (1995), eq. 41.
    """

    D0 = D[0]
    D1 = d1left(D,h)
    D2 = d2left(D,h)
    D22 = d2left(D**2,h)
    D3 = d3left(D,h)

    c1 = d1left(c,h)
    c2 = d2left(c,h)
    c3 = d3left(c,h)
    c4 = d4left(c,h)

    a1 = 0.5*dt*(D1*c1+D0*c2)

    a2 = dt**2/8*(D1*D2*c1+D0*D3*c1+D0*D2*c2+
                  2*(D1*D1*c2+D0*D2*c2+D0*D1*c3)+
                  D22*c3+D0*c4)
    
    return f11 + a1 + a2

def fL1right(f1,c,h,a,dt):
    """ Implements the right first intermediate BC. Khan & Liu (1995), eq. 40.
    """

    a1 = -0.5*dt*a*d1right(c,h)

    a2 = a**2*dt**2/8*d2right(c,h)

    return f1 + a1 + a2

def fL2right(f11, c, D, h, dt):
    """ Implements the left second intermediate BC. Khan & Liu (1995), eq. 41.
    """

    D0 = D[0]
    D1 = d1right(D,h)
    D2 = d2right(D,h)
    D22 = d2right(D**2,h)
    D3 = d3right(D,h)

    c1 = d1right(c,h)
    c2 = d2right(c,h)
    c3 = d3right(c,h)
    c4 = d4right(c,h)

    a1 = 0.5*dt*(D1*c1+D0*c2)

    a2 = dt**2/8*(D1*D2*c1+D0*D3*c1+D0*D2*c2+
                  2*(D1*D1*c2+D0*D2*c2+D0*D1*c3)+
                  D22*c3+D0*c4)
    
    return f11 + a1 + a2

def step_splitting(c,dt,M,h,D,a,p=None):
    """ Advance a time step with the operator splitting framework.
    """    

    #print(f"  max c= {c.max()}")
    #print(f"  min c= {c.min()}")

    # d(u_i)/dt = L1(u_i) = - d(W.u_i)/dz,     u_i(n) = c_i(n),     t in [n, n+1/2],                  (4)

    u = L1.L1_Advection(c,M,h,a,0.5*dt,full_step=False)
    #print(u.shape)
    #print(f"  u[0]= {u[0]}")
    #print(f"  u[-1]= {u[-1]}")

    u[0]=fL1left(c[0],c,h,a[0],dt)
    #print(f"  u[0]= {u[0]}")

    u[-1]=fL1right(c[-1],c,h,a[-1],dt)
    #print(f"  u[-1]= {u[-1]}")

    #print(f"  max u= {u.max()}")
    #print(f"  min u= {u.min()}")

    # d(v_i)/dt = L2(v_i) = d/dz(-D.dv_i/dz),  v_i(n) = u_i(n+1/2), t in [n, n+1/2],                  (5)

    v = L2.L2_Diffusion(u,h,D,0.5*dt,full_step=False)
    #print(f"  v[0]= {v[0]}")
    #print(f"  v[-1]= {v[-1]}")

    v[0] = fL2left(u[0], u, D, h, dt)

    v[-1] = fL2left(u[-1], u, D, h, dt)
    #print(f"  v[0]= {v[0]}")
    #print(f"  v[-1]= {v[-1]}")

    #print(f"  max v= {v.max()}")
    #print(f"  min v= {v.min()}")

    # d(w_i)/dt = L3(w_i) = r_i,               w_i(n) = v_i(n+1/2), t in [n, n+1],                    (6)

    w = v # L3=1

    w[0] = fL2left(v[0], v, D, h, dt)

    w[-1] = fL2left(v[-1], v, D, h, dt)

    #print(f"  max w= {w.max()}")
    #print(f"  min w= {w.min()}")

    # d(v_i)/dt = L2(v_i) = d/dz(-D.dv_i/dz),  v_i(n+1/2) = w_i(n+1), t in [n+1/2, n+1],              (7)

    v = L2.L2_Diffusion(w,h,D,0.5*dt,full_step=False)
    #print(f"  v[0]= {v[0]}")
    #print(f"  v[-1]= {v[-1]}")

    v[0] = fL2left(w[0], w, D, h, dt)

    v[-1] = fL2left(w[-1], w, D, h, dt)
    #print(f"  v[0]= {v[0]}")
    #print(f"  v[-1]= {v[-1]}")

    #print(f"  max v= {v.max()}")
    #print(f"  min v= {v.min()}")

    # d(u_i)/dt = L1(u_i) = -D.du_i/dz,  u_i(n+1/2) = v_i(n+1), t in [n+1/2, n+1],              (8) 

    tmp1 = L1.L1_Boundary(v, p[0], a[1], a[-2], p[1][0],p[1][1])#aflux_left, aflux_right)
    #print(f"  tmp1[0]= {tmp1[0]}")
    #print(f"  tmp1[-1]= {tmp1[1]}")
    #print(tmp1)

    tmp2 = L2.L2_Boundary(v,D,h, p[2][0], p[2][1])#v_left, dflux_right)
    #print(f"  tmp2[0]= {tmp2[0]}")
    #print(f"  tmp2[1]= {tmp2[1]}")
    #print(tmp2)

    v[0] = 0.5*(tmp1[0]+tmp2[0])
    v[-1] = 0.5*(tmp1[1]+tmp2[1])

    #print(f"  v[0]= {v[0]}")
    #print(f"  v[-1]= {v[-1]}")

    #print(f"  max v= {v.max()}")
    #print(f"  min v= {v.min()}")

    #print(f"  v[0]= {tmp1[0]+tmp2[0]}")
    #print(f"  v[-1]= {v[-1]}")
    #print(v)
    u = L1.L1_Advection(v,M,h,a,0.5*dt,full_step=False)

    #print(f"  max u= {u.max()}")
    #print(f"  min u= {u.min()}")

    return u



    


