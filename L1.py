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
# This file contains the objects necessary to implement the L1 operator as defined in 2.1.
#
#

import numpy as np

#+++++++++++++++++++++++++++++++++++++++++++
# Third order limited upwind scheme for advection
#+++++++++++++++++++++++++++++++++++++++++++

# Limiter function (K-scheme, Hundsdorfer & Verwer 2007, p. 219)

def psi(r,K): 
    return np.maximum(0,np.minimum(1,np.minimum(0.25*(1+K)+0.25*(1-K)*r,r)))

# Slope

def theta(w):#,s,m):

    #for j in range(1,m-1):
    #    s[j]=(f[j]-f[j-1]+10**-10)/(f[j+1]-f[j]+10**-10)

    s = (w[1:-1]-w[:-2]+10**-10)/(w[2:]-w[1:-1]+10**-10)

    return s 

# Third-order upwind limited scheme

def upwind3_limited(w,a,h):
    """ RHS of advection equation.
        Set K=1/3 to get 3rd order upwind-biased.
    """

    h1=1/h

    # Compute slope (Note: THETA has indices 0:N-1)
    # I have to pad the slope vector because of limiter PSI1
    THETA = np.pad(theta(w),(1,1),'edge') 

    # Compute limiter
    PSI = psi(THETA,1/3)
    PSI1 = psi(1/THETA,1/3) # PSI1 is PSI(1/THETA)

    # Compute values at cell boundaries 
    WR = w[0:-1]+PSI[0:-1]*(w[1:]-w[0:-1])

    WL = w[1:]+PSI1[1:]*(w[0:-1]-w[1:])

    # Velocity at interior cell boundaries
    A = a[1:-1]#0.5*(a[0:-1]+a[1:])

    # Compute fluxes

    flux_right = np.maximum(A[1:],0)*WR[1:]+np.minimum(A[1:],0)*WL[1:] # Flux at right cell boundary j+1/2

    flux_left = np.maximum(A[0:-1],0)*WR[0:-1]+np.minimum(A[0:-1],0)*WL[0:-1] # Flux at left cell boundary j-1/2

    # Compute advection term

    w_prime = h1*(flux_left-flux_right)

    return w_prime 

# Runge-Kutta method

def rk4(u,dt,f,h=1,W=1):
    """
    """

    # Fourth-order RK scheme coefficients
    a21 = 1/2
    a32 = 1/2
    a43 = 1
    b1=1/6
    b2=1/3
    b3=1/3
    b4=1/6
    # !!!!! I'M LEAVING W AS CONSTANT FOR NOW, UNTIL I FIND A WAY TO MAKE INTO A TIME DEPENDENT VARIABLE.
    # THEN THESE COEFFICIENTS WILL BE USED IN THE CALCULATION OF F2, F3 AND F4.
    #c1=0
    #c2=1/2
    #c3=1/2
    #c4=1

    # Stage 1
    u1=u
    u2=u1.copy()
    u3=u1.copy()
    u4=u1.copy()

    # Stage 2
    F2 = f(u1,W,h)
    #print("Stage 2:")
    #print(f"  max F: {F2.max()}")
    #print(f"  min F: {F2.min()}")
    u2[1:-1] = u[1:-1] + dt * a21 * F2

    # Stage 3
    F3 = f(u2,W,h)
    #print("Stage 3:")
    #print(f"  max F: {F3.max()}")
    #print(f"  min F: {F3.min()}")
    u3[1:-1] = u[1:-1] + dt * a32 * F3

    # Stage 4
    F4 = f(u3,W,h)#f(t[n-1] + c4*dt, u3, r3, ah, m)
    #print("Stage 4:")
    #print(f"  max F: {F4.max()}")
    #print(f"  min F: {F4.min()}")
    u4[1:-1] = u[1:-1] + dt * a43 * F4

    F1 = f(u1,W,h)
    F2 = f(u2,W,h)
    F3 = f(u3,W,h)
    F4 = f(u4,W,h)

    u[1:-1] = u[1:-1] + dt * (b1 * F1 + 
                  b2 * F2 +
                  b3 * F3 + 
                  b4 * F4 ) 
    
    return u

# Boundary condition at the left boundary
def bc_left(w, a, psi, psi1, gamma_0):
    """ bc_left implements the gamma_0 flux value at j=1/2, by modifying the value of w at j=0. The modification depends on the
        sign of the advection speed a_(1/2). 
    """

    p = gamma_0/a

    if psi==0 or psi1==0:
        w0=p
    else:
        if a>=0:
            if psi==1:
                w0 = 3*w[1]-3*w[2]+w[3]
            else:
                w0 = (p - psi*w[1])/(1-psi)
        else:
            w0 = (p-(1-psi1)*w[1])/psi1

    return w0

# Boundary condition at the right boundary
def bc_right(wM, a, psi, psi1, gamma_M):
    """ bc_right implements the gamma_M flux value at j=M+1/2, by modifying the value of w at j=M+1. The modification depends on the
        sign of the advection speed a_(M+1/2). 
    """
    p = gamma_M/a

    if psi==0 or psi1==0:
        w1=p
    else:
        if a>=0:
            w1 = (p-(1-psi)*wM)/psi
        else:
            w1 = (p-psi1*wM)/(1-psi1)

    return w1

def gamma(a0,x):
    return a0*x

def L1_Boundary(u, K, w_left, w_right, flux_left, flux_right):
    """ Computes boundary conditions at left and right for the advection problem.
    """
    # Compute limiter
    
    PSI = psi(theta(u),K)
    PSI1 = psi(1/theta(u),K) # PSI1 is PSI(1/THETA)

    #print("Boundary values:")
    
    #u[0]=bc_left(u, w_left, PSI[1], PSI1[1], flux_left)
    u0=bc_left(u, w_left, PSI[1], PSI1[1], flux_left)
    #print(f" u[0]:  {u[0]}")

    # Set right boundary consistent with flux

    #u[-1]=bc_right(u[-2], w_right, PSI[-2], PSI1[-2], flux_right)
    u1=bc_right(u[-2], w_right, PSI[-2], PSI1[-2], flux_right)
    #print(f" u[-1]:  {u[-1]}")

    return (u0,u1)


def L1_Advection(u,M,h,W,dt,full_step=False,P=None):
    """ Advection operator.
    """

    if P is None:
        P = []

    K=1/3

    u = rk4(u,dt,upwind3_limited,h,W)

    #print(f"Max u: {u.max()}")
    #print(f"Min u: {u.min()}")

    if full_step: # If this is not an intermediate step, we compute here the boundary values at j=0, M+1.

        w_left = W[1]
        w_right = W[-2]

        flux_left = gamma(P[0],h)*u[1]
        flux_right = gamma(P[0],M*h)*u[-2]

        u = L1_Boundary(u, K, w_left, w_right, flux_left, flux_right)

    return u






