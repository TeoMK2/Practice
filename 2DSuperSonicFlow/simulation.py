from errno import ETIME

import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.optimize import fsolve   #Nonlinear Equation Solver
from scipy.sparse.csgraph import depth_first_tree


# this program is aim to calculate Prandtl-Meyer flow with numerical simulation.
#
# Finite Different Method
#
# Conservative form:    d(U)/dt       +         d(E)/dx                +              d(F)/dy               = 0
#
# Continuity:           d(ρ)/dt       +        d(ρu)/dx                +             d(ρv)/dy               = 0
#
# Momentum:             d(ρu)/dt      +    d(ρu^2+p-τxx)/dx            +           d(ρuv-τyx)/dy             = 0
#                       d(ρv)/dt      +      d(ρuv-τxy)/dx             +          d(ρv^2+p-τyy)/dy           = 0
#
# Energy:               d(Et)/dt + d((Et+p)*u + qx - u*τxx - v*τxy)/dx + d((Et+p)*v + qy - u*τyx - v*τyy)/dy = 0
#
# there Et = ρ(e+V^2*0.5)  τxy = τyx = μ(dv/dx + du/dy)     p = ρTR   k = μ*cp/Pr
#       qx = -kdT/dx     τxx = lmd*(div(V)) + 2μ(du/dx)   e = ev*T
#       qy = -kdT/dy     τyy = lmd*(div(V)) + 2μ(dv/dy)   μ = μ0*(T/Tr)^(3*0.5)*(T+110)/(T+110)   (Sutherland’s law)

class parameter:

    def __init__(self):
        # Geometry
        self.Lx = 0.00001   #m

        # Physical parameters
        self.MaIn = 4      #case: Ma=25, Re=1000*25/4, Nx,Ny=41*(25/4)**(9/4)
        self.Prn = 0.71
        self.R = 287    #J/kg/K
        self.gamma = 1.4
        self.cv = self.R/(self.gamma-1)
        self.cp = self.gamma*self.cv
        self.Tr = 288.16        # reference temperature
        self.Tw = 288.16        # equal to Tin (K)
        self.mu0 = 1.789*10**-5 # reference dynamic viscosity   (Pa*s)

        # Calculative parameters
        self.K = 0.6
        self.Cy = 0.6
        self.Nx = 71
        self.Ny = 71
        self.tn = 10000  # Maximum iteration times

        # Inlet condition
        self.pin = 101325.0 # N/m^2
        self.Tin = 288.16   # equal to Tw (K)
        self.Uin = self.MaIn*np.sqrt(self.gamma*self.Tin*self.R)   # m/s
        self.rhoin = self.pin/self.R/self.Tin
        self.Ren = self.rhoin*self.Uin*self.Lx/self.mu0    #1000

        self.Ly = 5*self.delta()

    def dx(self):
        return self.Lx/self.Nx

    def dy(self):
        return self.Ly/self.Ny

    def delta(self):
        return 5*self.Lx/np.sqrt(self.Ren)

def muCalc(ppt, T):
    return ppt.mu0*(T/ppt.Tr)**(3*0.5)*(ppt.Tr+110)/(T+110)
def kappaCalc(ppt, mu):
    return mu*ppt.cp/ppt.Prn

def converge(ppt, rho, newRho, dt, iterNum):
    def resiCalc(ppt, rho, newRho, dt):
        return np.max(newRho-rho)/dt/ppt.rhoin*(ppt.Lx/ppt.Uin)
    flag = 0
    residual = resiCalc(ppt, rho, newRho, dt)

    if iterNum%10 == 0:
        print("iteration times: ", iterNum)
        print("residual: ", residual)
    if residual <= 1e-8:
        flag = 1
    if iterNum >= ppt.tn:
        flag = 2

    return flag

def BCSet(ppt, U):
    rho, u1, u2, e = conserve2Origin(U)
    T = e/ppt.cv
    p = rho*ppt.R*T

    u1[0 , :] , u2[0 , :] , p[0 , :] ,T[0 , :] = ppt.Uin            , 0                  , ppt.pin          , ppt.Tin              # inlet
    u1[-1, :] , u2[-1, :] , p[-1, :] ,T[-1, :] = 2*u1[-2,:]-u1[-3,:], 2*u2[-2,:]-u2[-3,:], 2*p[-2,:]-p[-3,:], 2*T[-2,:]-T[-3,:]    # outlet
    u1[: ,-1] , u2[: ,-1] , p[: ,-1] ,T[: ,-1] = ppt.Uin            , 0                  , ppt.pin          , ppt.Tin              # upper boundary
    u1[: , 0] , u2[: , 0] , p[: , 0] ,T[: , 0] = 0                  , 0                  , 2*p[: ,1]-p[: ,2], ppt.Tw               # wall
    u1[0 , 0] , u2[0 , 0] , p[0 , 0] ,T[0 , 0] = 0                  , 0                  , ppt.pin          , ppt.Tin              #Leading edge point

    rho = p/T/ppt.R
    U1 = rho
    U2 = rho*u1
    U3 = rho*u2
    U4 = rho*(T*ppt.cv + (u1**2 + u2**2)/2)

    return [U1, U2, U3, U4]

def conserve2Origin(U):

    rho = U[0]
    u1 = U[1]/U[0]
    u2 = U[2]/U[0]
    e = U[3]/U[0] - (u1**2 + u2**2)/2

    return rho, u1, u2, e

def origin2Conserve(ppt, rho, u1, u2, e, mu, kappa, direction):

    def midVariants1(u1, u2, T, mu, kappa, dx, dy, direction):
        nx = len(u1[:,0])
        ny = len(u1[0,:])
        tauxx = np.zeros_like(u1)
        tauxy = np.zeros_like(u1)
        qx = np.zeros_like(u1)

        if direction == 1:
            for j in range(ny):
                for i in range(nx):
                    if i < nx-1:
                        if j == 0:                      # i=[0,-2], j=0
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j+1]-u2[i,j])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i+1,j]-u2[i,j])/dx + (u1[i,j+1]-u1[i,j])/dy )
                        elif j == ny-1:    # i=[0,-2], j=-1
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j]-u2[i,j-1])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i+1,j]-u2[i,j])/dx + (u1[i,j]-u1[i,j-1])/dy )
                        else:                           # i=[0,-2], j=[1,-2] normal case
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j+1]-u2[i,j-1])/dy*0.5 )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i+1,j]-u2[i,j])/dx + (u1[i,j+1]-u1[i,j-1])/dy*0.5 )
                        qx[i,j] = -kappa[i,j]*(T[i+1,j] - T[i,j])/dx
                    elif i == nx-1:
                        if j == 0:                      # i=-1, j=0
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i,j]-u1[i-1,j])/dx - (u2[i,j+1]-u2[i,j])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i,j]-u2[i-1,j])/dx + (u1[i,j+1]-u1[i,j])/dy )
                        elif j == ny-1:    # i=-1, j=-1
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i,j]-u1[i-1,j])/dx - (u2[i,j]-u2[i,j-1])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i,j]-u2[i-1,j])/dx + (u1[i,j]-u1[i,j-1])/dy )
                        else:                           # i=-1, j=[1,-2]
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i,j]-u1[i-1,j])/dx - (u2[i,j+1]-u2[i,j-1])/dy*0.5 )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i,j]-u2[i-1,j])/dx + (u1[i,j+1]-u1[i,j-1])/dy*0.5 )
                        qx[i,j] = qx[-2,j]#-kappa[i,j]*(T[i,j] - T[i-1,j])/dx

        elif direction == -1:
            for j in range(ny):
                for i in range(nx):
                    if i > 0:
                        if j == 0:                      # i=[1,-1], j=0
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i,j]-u1[i-1,j])/dx - (u2[i,j+1]-u2[i,j])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i,j]-u2[i-1,j])/dx + (u1[i,j+1]-u1[i,j])/dy )
                        elif j == ny-1:    # i=[1,-1], j=-1
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i,j]-u1[i-1,j])/dx - (u2[i,j]-u2[i,j-1])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i,j]-u2[i-1,j])/dx + (u1[i,j]-u1[i,j-1])/dy )
                        else:                           # i=[1,-1], j=[1,-2] normal case
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i,j]-u1[i-1,j])/dx - (u2[i,j+1]-u2[i,j-1])/dy*0.5 )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i,j]-u2[i-1,j])/dx + (u1[i,j+1]-u1[i,j-1])/dy*0.5 )
                        qx[i,j] = -kappa[i,j]*(T[i,j] - T[i-1,j])/dx
                    elif i == 0:
                        if j == 0:                      # i=0, j=0
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j+1]-u2[i,j])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i+1,j]-u2[i,j])/dx + (u1[i,j+1]-u1[i,j])/dy )
                        elif j == ny-1:    # i=0, j=-1
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j]-u2[i,j-1])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i+1,j]-u2[i,j])/dx + (u1[i,j]-u1[i,j-1])/dy )
                        else:                           # i=0, j=[1,-2]
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j+1]-u2[i,j-1])/dy*0.5 )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i+1,j]-u2[i,j])/dx + (u1[i,j+1]-u1[i,j-1])/dy*0.5 )
                        qx[i,j] = qx[1,j]#-kappa[i,j]*(T[i+1,j] - T[i,j])/dx
        else:
            print("error direction vector detected.")
            sys.exit()

        return tauxx, tauxy, qx

    def midVariants2(u1, u2, T, mu, kappa, dx, dy, direction):
        nx = len(u2[:,0])
        ny = len(u2[0,:])
        tauyy = np.zeros_like(u2)
        tauyx = np.zeros_like(u2)
        qy = np.zeros_like(u2)

        if direction == 1:
            for i in range(nx):
                for j in range(ny):
                    if j < ny-1:
                        if i == 0:                      # i=0, j=[0,-2]
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i+1,j]-u1[i,j])/dx )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i+1,j]-u2[i,j])/dx )
                        elif i == nx-1:    # i=-1,j=[0,-2]
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i,j]-u1[i-1,j])/dx )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i,j]-u2[i-1,j])/dx )
                        else:                           # i=[1,-2], j=[0,-2] normal case
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i+1,j]-u2[i-1,j])/dx*0.5 )
                        qy[i,j] = -kappa[i,j]*(T[i,j+1] - T[i,j])/dy
                    elif j == ny-1:
                        if i == 0:                      # i=0, j=-1
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i+1,j]-u1[i,j])/dx )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i+1,j]-u2[i,j])/dx )
                        elif i == nx-1:    # i=-1, j=-1
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i,j]-u1[i-1,j])/dx )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i,j]-u2[i-1,j])/dx )
                        else:                           # i=[1,-2], j=-1
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i+1,j]-u2[i-1,j])/dx*0.5 )
                        qy[i,j] = qy[i,-2]#-kappa[i,j]*(T[i,j] - T[i,j-1])/dy

        elif direction == -1:
            for i in range(nx):
                for j in range(ny):
                    if j > 0:
                        if i == 0:                      # i=0, j=[1,-1]
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i+1,j]-u1[i,j])/dx )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i+1,j]-u2[i,j])/dx )
                        elif i == nx-1:    # i=-1,j=[1,-1]
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i,j]-u1[i-1,j])/dx )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i,j]-u2[i-1,j])/dx )
                        else:                           # i=[1,-2], j=[1,-1] normal case
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i+1,j]-u2[i-1,j])/dx*0.5 )
                        qy[i,j] = -kappa[i,j]*(T[i,j] - T[i,j-1])/dy
                    elif j == 0:
                        if i == 0:                      # i=0, j=0
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i+1,j]-u1[i,j])/dx )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i+1,j]-u2[i,j])/dx )
                        elif i == nx-1:    # i=-1, j=0
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i,j]-u1[i-1,j])/dx )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i,j]-u2[i-1,j])/dx )
                        else:                           # i=[1,-2], j=0
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                            tauyx[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i+1,j]-u2[i-1,j])/dx*0.5 )
                        qy[i,j] = qy[i,1]#-kappa[i,j]*(T[i,j+1] - T[i,j])/dy
        else:
            print("error direction vector detected.")
            sys.exit()

        return tauyy, tauxy, qy

    T = e/ppt.cv
    Et = rho*(e + (u1**2 + u2**2)/2)
    p = rho*ppt.R*T

    U1 = rho
    U2 = rho*u1
    U3 = rho*u2
    U4 = Et
    U = [U1, U2, U3, U4]

    # difference at contrary direction with the difference direction of MacCormack step
    # this processing aim to maintain second-order spatial accuracy in temporal advancement
    tauxx, tauxy, qx = midVariants1(u1, u2, T, mu, kappa, ppt.dx(), ppt.dy(), direction)
    E1 = rho*u1
    E2 = rho*u1**2 + p - tauxx
    E3 = rho*u1*u2 - tauxy
    E4 = (Et + p)*u1 - u1*tauxx - u2*tauxy + qx
    E = [E1, E2, E3, E4]

    tauyy, tauyx, qy = midVariants2(u1, u2, T, mu, kappa, ppt.dx(), ppt.dy(), direction)
    F1 = rho*u2
    F2 = rho*u1*u2 - tauyx
    F3 = rho*u2**2 + p - tauyy
    F4 = (Et + p)*u2 - u1*tauyx - u2*tauyy + qy
    F = [F1, F2, F3, F4]

    return U, E, F

def init():

    def oriVarAllocate():
        x = np.linspace(0, ppt.Lx, ppt.Nx)
        y = np.linspace(0, ppt.Ly, ppt.Ny)

        # Properties
        p = np.zeros((ppt.Nx,ppt.Ny),dtype=float)
        u1 = np.zeros((ppt.Nx,ppt.Ny),dtype=float)
        u2 = np.zeros((ppt.Nx,ppt.Ny),dtype=float)
        T = np.zeros((ppt.Nx,ppt.Ny),dtype=float)

        return x, y, p, u1, u2, T

    def flowFieldInit(ppt, p, u1, u2, T):

        p[:,:] = ppt.pin
        u1[:,:] = ppt.Uin
        u2[:,:] = 0
        T[:,:] = ppt.Tin

        # boundary condition
        # upper boundary
        u1[:,-1] = ppt.Uin
        u2[:,-1] = 0
        p[:,-1] = ppt.pin
        T[:,-1] = ppt.Tin

        # wall boundary
        u1[:,0] = 0
        u2[:,0] = 0
        p[:,0] = 2*p[:,1]-p[:,2]
        T[:,0] = ppt.Tw

        rho = p/T/ppt.R
        e = T*ppt.cv
        return rho, u1, u2, p, e

    ppt = parameter()
    x, y, p, u1, u2, T = oriVarAllocate()
    rho, u1, u2, p, e = flowFieldInit(ppt, p, u1, u2, T)

    return ppt, x, y, rho, u1, u2, p, e

def tscheme(ppt, rho, u1, u2, e):

    # MacCormack scheme
    def dtCalc(mu, rho, T, gamma, R, Prn, dx, dy, K):
        nu = np.max(4/3*(gamma*mu/Prn)/rho)
        a = np.sqrt(gamma*R*T)
        dt = 1/(np.abs(u1)/dx + np.abs(u2)/dy + a*np.sqrt(1/dx**2 + 1/dy**2) + 2*nu*(1/dx**2 + 1/dy**2))
        return np.min(K*dt)

    def artiVisc(p, U, Cy): #artificial viscosity
        S = np.zeros_like(p)
        for i in range(1,len(p[:,0])-1):
            for j in range(1,len(p[0,:])-1):
                S[i,j] = Cy*( np.abs(p[i+1,j] - 2*p[i,j] + p[i-1,j])/(p[i+1,j] + 2*p[i,j] + p[i-1,j])*(U[i+1,j] - 2*U[i,j] + U[i-1,j]) +
                              np.abs(p[i,j+1] - 2*p[i,j] + p[i,j-1])/(p[i,j+1] + 2*p[i,j] + p[i,j-1])*(U[i,j+1] - 2*U[i,j] + U[i,j-1]) )
        return S

    def preStep(E, F, dx, dy):
        predU = []
        for n in range(len(E[:])):
            nU, nE, nF = np.zeros_like(E[n]), E[n], F[n]
            for i in range(1,len(nU[:,0])-1):
                for j in range(1,len(nU[0,:])-1):
                    nU[i,j] = -(nE[i+1,j] - nE[i,j])/dx - (nF[i,j+1] - nF[i,j])/dy
            predU.append(nU)
        return predU

    def corrStep(preE, preF, dx, dy):
        corrdU = []
        for n in range(len(preE[:])):
            nU, nE, nF = np.zeros_like(preE[n]), preE[n], preF[n]
            for i in range(1,len(nU[:,0])-1):
                for j in range(1,len(nU[0,:])-1):
                    nU[i,j] = -(nE[i,j] - nE[i-1,j])/dx - (nF[i,j] - nF[i,j-1])/dy
            corrdU.append(nU)
        return corrdU

    mu = muCalc(ppt, e/ppt.cv)
    kappa = kappaCalc(ppt, mu)
    # Calculate dt
    dt = dtCalc(mu, rho, e/ppt.cv, ppt.gamma, ppt.R, ppt.Prn, ppt.dx(), ppt.dy(), ppt.K)
    U, E, F = origin2Conserve(ppt, rho, u1, u2, e, mu, kappa,-1)  #prepare for MacCormack pre-step

    # pre-step
    predU = preStep(E, F, ppt.dx(), ppt.dy())
    preU = []
    for n in range(len(U)):
        preU.append(U[n] + dt*predU[n])# + artiVisc(p, U[n], ppt.Cy))
    preU = BCSet(ppt, preU)

    preRho, preu1, preu2, pree = conserve2Origin(preU)

    preMu = muCalc(ppt, pree/ppt.cv)
    preKappa = kappaCalc(ppt, preMu)
    __, preE, preF = origin2Conserve(ppt, preRho, preu1, preu2, pree, preMu, preKappa,1)

    # correct-step
    corrdU = corrStep(preE, preF, ppt.dx(), ppt.dy())
    newU = []
    for n in range(len(U)):
        newU.append(U[n] + dt*(predU[n] + corrdU[n])*0.5)# + artiVisc(preP, preU[n], ppt.Cy))
    newU = BCSet(ppt, newU)

    return newU, dt

def MDOT(rho, u1, dy):

    inletSum = 0
    outletSum = 0

    for i in range(len(u1[0,:])):
        inletSum += rho[0,i]*u1[0,i]*dy*1
        outletSum += rho[-1,i]*u1[-1,i]*dy*1

    dif = np.abs(inletSum - outletSum)/(inletSum+outletSum)*2
    if dif <= 0.01:
        print("Mass flux meets the requirements: ", dif)
    else:
        print("Mass flux not meets the requirements: ", dif)

    return 0

def postProgress(ppt, x, y, rho, u1, u2, e):

    Ma = np.sqrt(u1**2 + u2**2)/np.sqrt(ppt.gamma*ppt.R*e/ppt.cv)
    p = rho*ppt.R*e/ppt.cv

    def drawData(x, y, data, name, start, end):
        plt.figure()
        # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
        pic = plt.contourf(x,y,data,alpha=0.8,cmap='jet')#,levels=np.linspace(start,end,50)
        # plt.plot(x, Ma, '-o', linewidth=1.0, color='black', markersize=1)
        plt.colorbar(pic)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Evolution of ' + name)
        # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
        plt.show()
        return

    # drawData(x, y, u1.T, 'u1', -0.1*ppt.Uin, 5*ppt.Uin)
    # drawData(x, y, u2.T, 'u2', -0.1*ppt.Uin, 5*ppt.Uin)
    drawData(x, y, p.T, 'Pressure', np.min(p), np.max(p))
    drawData(x, y, e.T/ppt.cv, 'Temperature', np.min(e.T/ppt.cv), np.max(e.T/ppt.cv))
    drawData(x, y, rho.T, 'rho', np.min(rho), np.max(rho))
    drawData(x, y, Ma.T, 'Mach number', np.min(Ma), np.max(Ma))

    return 0

def main():

    ppt, x, y, rho, u1, u2, p, e = init()
    iterNumb = 0
    t = 0

    while(1):
        newU, dt = tscheme(ppt, rho, u1, u2, e)

        t += dt
        newRho, newu1, newu2, newe = conserve2Origin(newU)

        flag = converge(ppt, rho, newRho, dt, iterNumb)

        if flag == 1:
            print("Calculation converged.")
            break
        elif flag == 2:
            print("Reach maximum iteration times")
            break

        iterNumb += 1
        rho, u1, u2, e = newRho, newu1, newu2, newe

    MDOT(rho, u1, ppt.dy())
    postProgress(ppt, x, y, rho, u1, u2, e)

    return 0

if __name__ == "__main__":
    main()