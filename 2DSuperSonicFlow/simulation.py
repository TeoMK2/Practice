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
        self.MaInit = 4
        self.Prn = 0.71
        self.R = 284.453
        self.gamma = 1.4
        self.cv = self.R/(self.gamma-1)
        self.cp = self.gamma*self.cv
        self.Tr = 288.16        # reference temperature
        self.Tw = 288.16        # equal to Tin (K)
        self.mu0 = 1.789*10**-5 # reference dynamic viscosity   (Pa*s)

        # Calculative parameters
        self.CourantNum = 0.6
        self.Cy = 0.6
        self.Nx = 71
        self.Ny = 71
        self.tn = 100

        # Inlet condition
        self.Uin = 340.28   # equal to speed of sound (m/s)
        self.pin = 101325.0 # N/m^2
        self.Tin = 288.16   # equal to Tw (K)
        self.rhoin = self.pin/self.R/self.Tin
        self.Ren =  self.rhoin*self.Uin*self.Lx/self.mu0#1000

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
def eCalc(ppt, T):
    return ppt.cv*T
def rhoCalc(p, T, R):
    return p/T/R

def BCSet(ppt, U):
    rho, u1, u2, e = conserve2Origin(U)
    T = e/ppt.cv
    p = rho*ppt.R*T

    u1[0,:] , u2[0,:] , p[0,:] ,T[0,:]  = ppt.Uin            , 0                  , ppt.pin          , ppt.Tin      # inlet
    u1[-1,:], u2[-1,:], p[-1,:],T[-1,:] = 2*u1[-2,:]-u1[-3,:], 2*u2[-2,:]-u2[-3,:], 2*p[-2,:]-p[-3,:], 2*T[-2,:]-T[-3,:]   # outlet
    u1[:,-1], u2[:,-1], p[:,-1],T[:,-1] = ppt.Uin            , 0                  , ppt.pin          , ppt.Tin     # upper boundary
    u1[:,0] , u2[:,0] , p[:,0] ,T[:,0]  = 0                  , 0                  , 2*p[:,-1]-p[:,-2], ppt.Tw   # wall

    rho = rhoCalc(p, T, ppt.R)
    U1 = rho
    U2 = rho*u1
    U3 = rho*u2
    U4 = rho*(T*ppt.cv+np.sqrt(u1**2+u2**2)/2)

    return [U1, U2, U3, U4]

def conserve2Origin(U):

    rho = U[0]
    u1 = U[1]/rho
    u2 = U[2]/rho
    e = U[3]/rho-np.sqrt(u1**2 + u2**2)/2

    return rho, u1, u2, e

def origin2Conserve(ppt, rho, u1, u2, e, mu, kappa, direction):

    def midVariants1(u1, u2, T, mu, kappa, dx, dy, direction):
        tauxx = np.zeros_like(u1)
        tauxy = np.zeros_like(u1)
        qx = np.zeros_like(u1)

        if direction == 1:
            for j in range(len(tauxx[0,:])):
                for i in range(len(tauxx[:,0])):
                    if i < len(tauxx[:,0])-1:
                        if j == 0:                      # i=[0,-2], j=0
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j+1]-u2[i,j])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i+1,j]-u2[i,j])/dx + (u1[i,j+1]-u1[i,j])/dy )
                        elif j == len(tauxx[0,:])-1:    # i=[0,-2], j=-1
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j]-u2[i,j-1])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i+1,j]-u2[i,j])/dx + (u1[i,j]-u1[i,j-1])/dy )
                        else:                           # i=[0,-2], j=[1,-2] normal case
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j+1]-u2[i,j-1])/dy*0.5 )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i+1,j]-u2[i,j])/dx + (u1[i,j+1]-u1[i,j-1])/dy*0.5 )
                        qx[i,j] = -kappa[i,j]*(T[i+1,j] - T[i,j])/dx
                    elif i == len(tauxx[:,0])-1:
                        if j == 0:                      # i=-1, j=0
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i,j]-u1[i-1,j])/dx - (u2[i,j+1]-u2[i,j])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i,j]-u2[i-1,j])/dx + (u1[i,j+1]-u1[i,j])/dy )
                        elif j == len(tauxx[0,:])-1:    # i=-1, j=-1
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i,j]-u1[i-1,j])/dx - (u2[i,j]-u2[i,j-1])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i,j]-u2[i-1,j])/dx + (u1[i,j]-u1[i,j-1])/dy )
                        else:                           # i=-1, j=[1,-2]
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i,j]-u1[i-1,j])/dx - (u2[i,j+1]-u2[i,j-1])/dy*0.5 )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i,j]-u2[i-1,j])/dx + (u1[i,j+1]-u1[i,j-1])/dy*0.5 )
                        qx[i,j] = -kappa[i,j]*(T[i,j] - T[i-1,j])/dx

        elif direction == -1:
            for j in range(len(tauxx[0,:])):
                for i in range(len(tauxx[:,0])):
                    if i > 0:
                        if j == 0:                      # i=[1,-1], j=0
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i,j]-u1[i-1,j])/dx - (u2[i,j+1]-u2[i,j])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i,j]-u2[i-1,j])/dx + (u1[i,j+1]-u1[i,j])/dy )
                        elif j == len(tauxx[0,:])-1:    # i=[1,-1], j=-1
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
                        elif j == len(tauxx[0,:])-1:    # i=0, j=-1
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j]-u2[i,j-1])/dy )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i+1,j]-u2[i,j])/dx + (u1[i,j]-u1[i,j-1])/dy )
                        else:                           # i=0, j=[1,-2]
                            tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j+1]-u2[i,j-1])/dy*0.5 )
                            tauxy[i,j] =     mu[i,j]*(   (u2[i+1,j]-u2[i,j])/dx + (u1[i,j+1]-u1[i,j-1])/dy*0.5 )
                        qx[i,j] = -kappa[i,j]*(T[i+1,j] - T[i,j])/dx
        else:
            print("error direction vector detected.")
            sys.exit()

        return tauxx, tauxy, qx

    def midVariants2(u1, u2, T, mu, kappa, dx, dy, direction):
        tauyy = np.zeros_like(u2)
        tauxy = np.zeros_like(u2)
        qy = np.zeros_like(u2)

        if direction == 1:
            for i in range(len(tauyy[:,0])):
                for j in range(len(tauyy[0,:])):
                    if j < len(tauyy[0,:])-1:
                        if i == 0:                      # i=0, j=[0,-2]
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i+1,j]-u1[i,j])/dx )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i+1,j]-u2[i,j])/dx )
                        elif i == len(tauyy[:,0])-1:    # i=-1,j=[0,-2]
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i,j]-u1[i-1,j])/dx )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i,j]-u2[i-1,j])/dx )
                        else:                           # i=[1,-2], j=[0,-2] normal case
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i+1,j]-u2[i-1,j])/dx*0.5 )
                        qy[i,j] = -kappa[i,j]*(T[i,j+1] - T[i,j])/dy
                    elif j == len(tauyy[0,:])-1:
                        if i == 0:                      # i=0, j=-1
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i+1,j]-u1[i,j])/dx )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i+1,j]-u2[i,j])/dx )
                        elif i == len(tauyy[:,0])-1:    # i=-1, j=-1
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i,j]-u1[i-1,j])/dx )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i,j]-u2[i-1,j])/dx )
                        else:                           # i=[1,-2], j=-1
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i+1,j]-u2[i-1,j])/dx*0.5 )
                        qy[i,j] = -kappa[i,j]*(T[i,j] - T[i,j-1])/dy

        elif direction == -1:
            for i in range(len(tauyy[:,0])):
                for j in range(len(tauyy[0,:])):
                    if j > 0:
                        if i == 0:                      # i=0, j=[1,-1]
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i+1,j]-u1[i,j])/dx )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i+1,j]-u2[i,j])/dx )
                        elif i == len(tauyy[:,0])-1:    # i=-1,j=[1,-1]
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i,j]-u1[i-1,j])/dx )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i,j]-u2[i-1,j])/dx )
                        else:                           # i=[1,-2], j=[1,-1] normal case
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i,j-1])/dy + (u2[i+1,j]-u2[i-1,j])/dx*0.5 )
                        qy[i,j] = -kappa[i,j]*(T[i,j] - T[i,j-1])/dy
                    elif j == 0:
                        if i == 0:                      # i=0, j=0
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i+1,j]-u1[i,j])/dx )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i+1,j]-u2[i,j])/dx )
                        elif i == len(tauyy[:,0])-1:    # i=-1, j=0
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i,j]-u1[i-1,j])/dx )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i,j]-u2[i-1,j])/dx )
                        else:                           # i=[1,-2], j=0
                            tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                            tauxy[i,j] =     mu[i,j]*(   (u1[i,j+1]-u1[i,j])/dy + (u2[i+1,j]-u2[i-1,j])/dx*0.5 )
                        qy[i,j] = -kappa[i,j]*(T[i,j+1] - T[i,j])/dy
        else:
            print("error direction vector detected.")
            sys.exit()

        return tauyy, tauxy, qy

    T = e/ppt.cv
    Et = rho*(e+np.sqrt(u1**2+u2**2)/2)
    p = rho*ppt.R*T

    U1 = rho
    U2 = rho*u1
    U3 = rho*u2
    U4 = Et
    U = [U1, U2, U3, U4]

    # difference at contrary direction with the difference direction of MacCormack step
    # this processing aim to maintain second-order spatial accuracy in temporal advancement
    tauxx, tauxy, qx = midVariants1(u1, u2, T, mu, kappa, ppt.dx(), ppt.dy(), direction)
    E1 = U2
    E2 = rho*u1**2 + p - tauxx
    E3 = rho*u1*u2 - tauxy
    E4 = (Et + p)*u1 - u1*tauxx - u2*tauxy + qx
    E = [E1, E2, E3, E4]

    tauyy, tauxy, qy = midVariants2(u1, u2, T, mu, kappa, ppt.dx(), ppt.dy(), direction)
    F1 = U3
    F2 = rho*u1*u2 - tauxy
    F3 = rho*u2**2 + p - tauxy
    F4 = (Et + p)*u2 - u1*tauxy - u2*tauyy + qy
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
        p[:,0] = 2*p[:,-1]-p[:,-2]
        T[:,0] = ppt.Tw

        rho = rhoCalc(p, T, ppt.R)
        e = eCalc(ppt, T)
        mu = muCalc(ppt, T)
        kappa = kappaCalc(ppt, mu)
        return rho, u1, u2, T, p, e, mu, kappa

    ppt = parameter()
    x, y, p, u1, u2, T = oriVarAllocate()
    rho, u1, u2, T, p, e, mu, kappa = flowFieldInit(ppt, p, u1, u2, T)

    return ppt, rho, u1, u2, T, p, e, mu, kappa

def tscheme(ppt, rho, u1, u2, e, mu, kappa):

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
            for i in range(len(nU[:,0])-1):
                for j in range(len(nU[0,:])-1):
                    nU[i,j] = -((nE[i+1,j] - nE[i,j])/dx + (nF[i,j+1]-nF[i,j])/dy)
            predU.append(nU)
        return predU

    def corrStep(preE, preF, dx, dy):
        corrdU = []
        for n in range(len(preE[:])):
            nU, nE, nF = np.zeros_like(E[n]), E[n], F[n]
            for i in range(1,len(nU[:,0])):
                for j in range(1,len(nU[0,:])):
                    nU[i,j] = -((nE[i,j] - nE[i-1,j])/dx + (nF[i,j-1]-nF[i,j])/dy)
            corrdU.append(nU)
        return corrdU

    # Calculate dt
    dt = dtCalc(mu, rho, e/ppt.cv, ppt.gamma, ppt.R, ppt.Prn, ppt.dx(), ppt.dy(), ppt.CourantNum)
    U, E, F = origin2Conserve(ppt, rho, u1, u2, e, mu, kappa, -1)  #prepare for MacCormack pre-step

    # pre-step
    predU = preStep(E, F, ppt.dx(), ppt.dy())
    preU = []
    # p = rho*ppt.R*e/ppt.cv
    for n in range(len(U)):
        preU.append(U[n] + dt*predU[n])# + artiVisc(p, U[n], ppt.Cy))

    preRho, preu1, preu2, pree = conserve2Origin(preU)

    mu = muCalc(ppt, pree/ppt.cv)
    kappa = kappaCalc(ppt, mu)
    __, preE, preF = origin2Conserve(ppt, preRho, preu1, preu2, pree, mu, kappa, 1)

    # correct-step
    corrdU = corrStep(preE, preF, ppt.dx(), ppt.dy())
    newU = []
    # preP = preRho*ppt.R*pree/ppt.cv
    for n in range(len(U)):
        newU.append(U[n] + dt*(predU[n] + corrdU[n])*0.5)# + artiVisc(preP, preU[n], ppt.Cy))

    newU = BCSet(ppt, newU)
    return newU, dt

# def postProgress(xSet, E, eta, F1, F2, F3, F4, gamma, R):
#
#     rho, u1, u2, p, T, Ma = conserve2Origin(F1, F2, F3, F4, gamma, R)
#
#     def printData():
#         print("------------solve complete.------------")
#         # print("iteration or temporal advancement times:", timeStepNumb)
#         # print("total physical space:", xSet)
#
#         print("---------------------------------------")
#         # print("residual:", residual)
#         # print("ρ:", rho)
#         # print("u1:", u1)
#         # print("T:", T)
#         # print("p", p)
#         # print("Ma", Ma)
#         # print("F1:", F1)
#         return
#
#     def drawData(x, y, data, name):
#         plt.figure()
#         # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
#         pic = plt.contourf(x,y,data,alpha=0.8,cmap='jet')
#         # plt.plot(x, Ma, '-o', linewidth=1.0, color='black', markersize=1)
#         plt.colorbar(pic)
#         plt.xlabel('x (m)')
#         plt.ylabel('y (m)')
#         plt.title('Evolution of ' + name)
#         # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
#         plt.show()
#         return
#
#     # printData()
#
#     drawData(x, y, Ma, 'Ma')
#     drawData(x, y, rho, 'rho')
#     drawData(x, y, u1, 'u1')
#     drawData(x, y, u2, 'u2')
#     drawData(x, y, T, 'T')
#     drawData(x, y, p, 'p')
#
#     return 0

def main():

    ppt, rho, u1, u2, T, p, e, mu, kappa = init()
    iterNumb = 0
    t = 0

    while(t < ppt.tn):
        newU, dt = tscheme(ppt, rho, u1, u2, e, mu, kappa)

        t += dt
        rho, u1, u2, e = conserve2Origin(newU)
        mu = muCalc(ppt, e/ppt.cv)
        kappa = kappaCalc(ppt, mu)
        if (iterNumb == 10):
            print("1")
        if (iterNumb >= 1000): #defensive design
            break
        iterNumb += 1

# postProgress(xSet, E, eta, F1, F2, F3, F4, gamma, R)

    return 0

if __name__ == "__main__":
    main()