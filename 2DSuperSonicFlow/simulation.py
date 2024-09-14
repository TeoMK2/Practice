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

def muCalc(T, Tr, mu0):
    return mu0*(T/Tr)**(3*0.5)*(Tr+110)/(T+110)
def kappaCalc(mu, cp, Prn):
    return mu*cp/Prn
def eCalc(T, cv):
    return cv*T
def rhoCalc(p, T, R):
    return p/T/R

def BCupdate(data):
    data[0,:] = data[1,:]
    data[-1,:] = data[-2,:]
    data[:,0] = 0
    data[:,-1] = data[:,-2]
    return 0

def BCSet(ppt, U):
    rho, u1, u2, e = conserve2Origin(U)
    T = e/ppt.cv
    p = rho*ppt.R*T

    u1[0,:] , u2[0,:] , p[0,:] ,T[0,:]  = ppt.Uin            ,                   0,           ppt.pin, ppt.Tin      # inlet
    u1[:,-1], u2[:,-1], p[:,-1],T[:,-1] = ppt.Uin            ,                   0,           ppt.pin, ppt.Tin     # upper boundary
    u1[:,0] , u2[:,0] , p[:,0] ,T[:,0]  = 0                  ,                   0, 2*p[:,-1]-p[:,-2], ppt.Tw   # wall
    u1[-1,:], u2[-1,:], p[-1,:],T[-1,:] = 2*u1[-2,:]-u1[-3,:], 2*u2[-2,:]-u2[-3,:], 2*p[:,-2]-p[:,-3], 2*T[-2,:]-T[-3,:]   # outlet

    U1 = rho
    U2 = rho*u1
    U3 = rho*u2
    U4 = rho*(e/ppt.cv+np.sqrt(u1**2+u2**2)/2)

    return np.hstack([U1, U2, U3, U4])

def conserve2Origin(U):

    rho = U[:,0]
    u1 = U[:,1]/rho
    u2 = U[:,2]/rho
    e = U[:,3]/rho-np.sqrt(u1**2 + u2**2)/2

    return rho, u1, u2, e

def origin2Conserve(ppt, rho, u1, u2, e, p, mu, kappa, direction):

    def midVariants1(u1, u2, T, mu, kappa, dx, dy, direction):
        tauxx = np.zeros_like(u1)
        tauxy = np.zeros_like(u1)
        qx = np.zeros_like(u1)
        for i in range(1,len(tauxx[:,0])-1):
            for j in range(1,len(tauxx[0,:])-1):
                if direction == 1:
                    tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i,j])/dx - (u2[i,j+1]-u2[i,j-1])/dy*0.5 )
                    tauxy[i,j] =     mu[i,j]*(   (u1[i+1,j]-u1[i,j])/dx + (u2[i,j+1]-u2[i,j-1])/dy*0.5 )
                    qx[i,j] = -kappa[i,j]*(T[i+1,j] - T[i,j])/dx
                elif direction == -1:
                    tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i,j]-u1[i-1,j])/dx - (u2[i,j+1]-u2[i,j-1])/dy*0.5 )
                    tauxy[i,j] =     mu[i,j]*(   (u1[i,j]-u1[i-1,j])/dx + (u2[i,j+1]-u2[i,j-1])/dy*0.5 )
                    qx[i,j] = -kappa[i,j]*(T[i,j] - T[i-1,j])/dx
                else:
                    print("error direction vector detected.")
                    sys.exit()
        for property in ([tauxx, tauxy, qx]):
            BCupdate(property)
        return tauxx, tauxy, qx

    def midVariants2(u1, u2, T, mu, kappa, dx, dy, direction):
        tauyy = np.zeros_like(u2)
        tauxy = np.zeros_like(u2)
        qy = np.zeros_like(u2)
        for i in range(1,len(tauyy[:,0])-1):
            for j in range(1,len(tauyy[0,:])-1):
                if direction == 1:
                    tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j])/dy - (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                    tauxy[i,j] =     mu[i,j]*(   (u2[i,j+1]-u2[i,j])/dy + (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                    qy[i,j] = -kappa[i,j]*(T[i,j+1] - T[i,j])/dy
                elif direction == -1:
                    tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j]-u2[i,j-1])/dy - (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                    tauxy[i,j] =     mu[i,j]*(   (u2[i,j]-u2[i,j-1])/dy + (u1[i+1,j]-u1[i-1,j])/dx*0.5 )
                    qy[i,j] = -kappa[i,j]*(T[i,j] - T[i,j-1])/dy
                else:
                    print("error direction vector detected.")
                    sys.exit()
        for property in ([tauyy, tauxy, qy]):
            BCupdate(property)
        return tauyy, tauxy, qy

    T = e/ppt.cv
    Et = rho*(e+np.sqrt(u1**2+u2**2)/2)

    U1 = rho
    U2 = rho*u1
    U3 = rho*u2
    U4 = Et
    U = np.hstack([U1, U2, U3, U4])

    # difference at contrary direction with the difference direction of MacCormack step
    # this processing aim to maintain second-order spatial accuracy in temporal advancement
    tauxx, tauxy, qx = midVariants1(u1, u2, T, mu, kappa, ppt.dx(), ppt.dy(), direction)
    E1 = U2
    E2 = rho*u1**2 + p - tauxx
    E3 = rho*u1*u2 - tauxy
    E4 = (Et + p)*u1 - u1*tauxx - u2*tauxy + qx
    E = np.hstack([E1, E2, E3, E4])

    tauyy, tauxy, qy = midVariants2(u1, u2, T, mu, kappa, ppt.dx(), ppt.dy(), direction)
    F1 = U3
    F2 = rho*u1*u2 - tauxy
    F3 = rho*u2**2 + p - tauxy
    F4 = (Et + p)*u2 - u1*tauxy - u2*tauyy + qy
    F = np.hstack([F1, F2, F3, F4])

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

        p[:,:] = 0
        u1[:,:] = 0
        u2[:,:] = 0
        T[:,:] = 0

        # boundary condition
        u1[0,:], u2[0,:], p[0,:], T[0,:] = ppt.Uin, 0, ppt.pin, ppt.Tin # inlet
        u1[:,-1], u2[:,-1], p[:,-1], T[:,-1] = ppt.Uin, 0, ppt.pin, ppt.Tin # upper boundary
        T[:,0], u1[:,0], u2[:,0] = ppt.Tw, 0, 0                         # wall

        rho = rhoCalc(p, T, ppt.R)
        e = eCalc(T, ppt.cv)
        mu = muCalc(T, ppt.Tr, ppt.mu0)
        kappa = kappaCalc(mu, ppt.cp, ppt.Prn)
        return rho, u1, u2, T, p, e, mu, kappa

    ppt = parameter()
    x, y, p, u1, u2, T = oriVarAllocate()
    rho, u1, u2, T, p, e, mu, kappa = flowFieldInit(ppt, p, u1, u2, T)

    return ppt, rho, u1, u2, T, p, e, mu, kappa

def tscheme(ppt, rho, u1, u2, T, p, e, mu, kappa):

    # MacCormack scheme
    def dtCalc(mu, rho, T, gamma, Prn, dx, dy, K):
        nu = np.max(4/3*(gamma*mu/Prn)/rho)
        dt = 1/(np.abs(u1)/dx + np.abs(u2)/dy + np.sqrt(T)*np.sqrt(1/dx**2 + 1/dy**2) + 2*nu*(1/dx**2 + 1/dy**2))
        return np.min(K*dt)

    def artiVisc(p, U, Cy): #artificial viscosity
        S = np.zeros_like(p)
        for i in range(1,len(p[:,0])-1):
            for j in range(1,len(p[0,:])-1):
                S[i,j] = Cy*( np.abs(p[i+1,j] - 2*p[i,j] + p[i-1,j])/(p[i+1,j] + 2*p[i,j] + p[i-1,j])*(U[i+1,j] - 2*U[i,j] + U[i-1,j]) +
                              np.abs(p[i,j+1] - 2*p[i,j] + p[i,j-1])/(p[i,j+1] + 2*p[i,j] + p[i,j-1])*(U[i,j+1] - 2*U[i,j] + U[i,j-1]) )
        return S

    def preStep(E, F, dx, dy):
        predU = np.zeros_like(E)
        for n in range(E[0,:]):
            nE, nF = E[:,n], F[:,n]
            for i in range(len(predU[:,0])-1):
                for j in range(len(predU[0,:])-1):
                    predU[:,n][i,j] = -((nE[i+1,j] - nE[i,j])/dx + (nF[i,j+1]-nF[i,j])/dy)
        return predU

    def corrStep(preE, preF, dx, dy):
        corrdU = np.zeros_like(preE)
        for n in range(preE[0,:]):
            nE, nF = preE[:,n], preF[:,n]
            for i in range(1,len(corrdU[:,0])):
                for j in range(1,len(corrdU[0,:])):
                    corrdU[:,n][i,j] = -((nE[i,j] - nE[i-1,j])/dx + (nF[i,j-1]-nF[i,j])/dy)
        return corrdU


    # Calculate dt
    dt = dtCalc(mu, rho, T, ppt.gamma, ppt.Prn, ppt.dx(), ppt.dy(), ppt.CourantNum)
    U, E, F = origin2Conserve(ppt, rho, u1, u2, e, p, mu, kappa,-1)  #prepare for MacCormack pre-step
    #pre-step

    predU = preStep(E, F, ppt.dx(), ppt.dy())
    preU = np.hstack([U[:,0] + dt*predU[:,0] + artiVisc(p, U[:,0], ppt.Cy),
                      U[:,1] + dt*predU[:,1] + artiVisc(p, U[:,1], ppt.Cy),
                      U[:,2] + dt*predU[:,2] + artiVisc(p, U[:,2], ppt.Cy),
                      U[:,3] + dt*predU[:,3] + artiVisc(p, U[:,3], ppt.Cy)])
    preRho, preu1, preu2, pree = conserve2Origin(preU)
    preP = preRho*ppt.R*pree/ppt.cv
    mu = muCalc(pree/ppt.cv, ppt.Tr, ppt.mu0)
    kappa = kappaCalc(mu, ppt.cp, ppt.Prn)
    __, preE, preF = origin2Conserve(ppt, preRho, preu1, preu2, pree, preP, mu, kappa,1)

    # #correct-step
    corrdU = corrStep(preE, preF, ppt.dx(), ppt.dy())
    newU = np.hstack([U[:,0] + dt*(predU[:,0] + corrdU[:,0])*0.5 + artiVisc(preP, preU[:,0], ppt.Cy),
                      U[:,1] + dt*(predU[:,1] + corrdU[:,1])*0.5 + artiVisc(preP, preU[:,1], ppt.Cy),
                      U[:,2] + dt*(predU[:,2] + corrdU[:,2])*0.5 + artiVisc(preP, preU[:,2], ppt.Cy),
                      U[:,3] + dt*(predU[:,3] + corrdU[:,3])*0.5 + artiVisc(preP, preU[:,3], ppt.Cy)])

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
    xSet = np.zeros(1,dtype=float)
    iterNumb = 0
    t = 0

    while(t < ppt.tn):
        newU, dt = tscheme(ppt, rho, u1, u2, T, p, e, mu, kappa)

        t += dt
        rho, u1, u2, e = conserve2Origin(newU)
        T = e/ppt.cv
        p = rho*ppt.R*e
        mu = muCalc(e/ppt.cv, ppt.Tr, ppt.mu0)
        kappa = kappaCalc(mu, ppt.cp, ppt.Prn)

        if (iterNumb >= 1000): #defensive design
            break

# postProgress(xSet, E, eta, F1, F2, F3, F4, gamma, R)

    return 0

if __name__ == "__main__":
    main()