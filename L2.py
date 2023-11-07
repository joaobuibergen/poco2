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
# This file contains the objects necessary to implement the L2 operator as defined in 2.1.
#
#

import numpy as np
from scipy.sparse import spdiags
from scipy.sparse.linalg import inv

#+++++++++++++++++++++++++++++++++++++++++++
# FUNCTIONS
#+++++++++++++++++++++++++++++++++++++++++++

#
# Conservative central scheme for diffusion
#

def implicit_trapezoidal(w,d,h,dt,gamma0,gamma1):
    """ Implicit trapezoidal rule for the diffusion equation with Dirichlet BC at j=0 and Neumann BC at j=M.

        Implemented according to Hundsdorfer & Verwer (2003) p. 143 eq 1.11.

    """
    
    h2 = 1/h**2

    b = 0.5*h2*dt*d[1:-1]

    A_main = 1+2*b
    A_lower = -b
    A_upper = A_lower.copy()

    # Apply BC
    A_main[0] = 1+3*b[0]
    A_main[-1] = 1+b[-1]
    #A_lower[0]=0
    #A_upper[-1]=0

    A = spdiags(np.array([A_main,
                                A_lower,
                                A_upper]),np.array([0, -1, 1]),
                                m=A_main.size,n=A_main.size,
                                format="csc")
    #print(A)

    B_main = 1-2*b
    B_lower = b
    B_upper = B_lower.copy()

    # Apply BC
    B_main[0] = 1-3*b[0]
    B_main[-1] = 1-b[-1]
    #B_lower[0]=0
    #B_upper[-1]=0

    B = spdiags(np.array([B_main,
                                B_lower,
                                B_upper]),np.array([0, -1, 1]),
                                m=A_main.size,n=A_main.size,
                                format="csc")


    # Forcing vector
    G = dt*np.zeros(b.shape)
    G[0] += 2*b[1]*(gamma0[0]+gamma0[1])
    G[-1] += h*b[-2]*(gamma1[0]+gamma1[1])

    #print(G)

    A1 = inv(A)
    #print(A1)

    C = np.matmul(A1.toarray(),B.toarray())

    D = np.dot(A1.toarray(),G)

    #print(C)

    u = np.dot(C,w[1:-1])+D

    return u

def bc_left_diffusion(w1, D, h, gamma0):
    """ bc_left_diffusion implements the gamma_0 flux value at j=1/2, by modifying the value of w at j=0. 
    """

    return w1-h*gamma0/D

def bc_right_diffusion(wM, D, h, gammaM):
    """ bc_left_diffusion implements the gamma_M flux value at j=M+1/2, by modifying the value of w at j=M+1. 
    """

    return h*gammaM/D+wM

# Boundary condition diffusive flux 
# Hundsdorder & Verwer (2003)
#def gamma(d,x,t):
#   return -d*np.pi/2*np.exp(-0.25*np.pi**2*t)*np.sin(0.5*np.pi*x)

def L2_Boundary(u,D,h,u_left,flux_right):
    """ Computes boundary conditions at left and right for the diffusion problem.
    """
    #u[0] = 2*u_left-u[1]
    u0 = 2*u_left-u[1]
    #print(f" u_left:  {u_left}")
    #print(f" u[1]:  {u[1]}")
    #u[-1] = bc_right_diffusion(u[-2], D[-1], h, flux_right)
    u1 = bc_right_diffusion(u[-2], D[-1], h, flux_right)
        #print(f" u[-1]:  {u[-1]}")

    return (u0,u1)


def L2_Diffusion(u,h,D,dt,full_step=False,P=None):
    """ Diffusion operator.
    """

    if P is None:
        P = [np.empty((2,)),np.empty((2,))]

    #g0=np.array([gamma0[n-1],gamma0[n]])
    #g1=np.array([gamma1[n-1],gamma1[n]])
    #print(P[0])
    #print(P[1])

    u[1:-1] = implicit_trapezoidal(u,D,h,dt,P[0],P[1])

    #print(f"Max u: {u.max()}")
    #print(f"Min u: {u.min()}")
    #print(f" g0[n]:  {P[0][1]}")
    #print(f" u[1]:  {u[1]}")

    if full_step: # If this is not an intermediate step, we compute here the boundary values at j=0, M+1.

        u[0] = 2*P[0][1]-u[1]
        #print(f" u[0]:  {u[0]}")
        u[-1] = bc_right_diffusion(u[-2], D[-1], h, P[1][1])
        #print(f" u[-1]:  {u[-1]}")

    return u

