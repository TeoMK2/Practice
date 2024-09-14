from errno import ETIME

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve   #Nonlinear Equation Solver

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
# there Et = ρ(e+V^2/2)  τxy = τyx = μ(dv/dx + du/dy)     p = ρTR   k = μ*cp/Pr
#       qx = -kdT/dx     τxx = lmd*(div(V)) + 2μ(du/dx)   e = ev*T
#       qy = -kdT/dy     τyy = lmd*(div(V)) + 2μ(dv/dy)   μ = μ0*(T/T0)^(3/2)*(T+110)/(T+110)   (Sutherland’s law)

class parameter:

    def __init__(self):
        self.Lx = 0
        self.Ly = 0
        self.Ren = 1000
        self.MaInit = 4
        self.Prn = 1

        self.R = 0
        self.cv = 0
        self.cp = 0
        self.T0 = 0 # reference temperature
        self.mu0 = 0 # reference dynamic viscosity

        self.CourantNum = 0.5
        self.Cy = 0.6
        self.Nx = 40
        self.Ny = 401
        self.tn = 100

    def dx(self):
        return (self.Lx)/self.Nx

    def dy(self):
        return (self.Ly)/self.Ny

def muCalc(T, T0, mu0):
    return mu0*(T/T0)**(3/2)*(T0+110)/(T+110)

def kappaCalc(mu, cp, Prn):
    return mu*cp/Prn

def eCalc(T, cv):
    return cv*T

def pCalc(rho, T, R):
    return rho*T*R

def BCupdate(data):
    data[0,:] = data[1,:]
    data[-1,:] = data[-2,:]
    data[:,0] = 0
    data[:,-1] = data[:,-2]
    return 0

# TODO: boundray condition
# def BCSet(U):
#
#     return F1, F2, F3, F4

def conserve2Origin(U):

    rho = U[:,0]
    u1 = U[:,1]/rho
    u2 = U[:,2]/rho
    Et = U[:,3]
    e = Et/rho-np.sqrt(u1**2 + u2**2)

    return rho, u1, u2, e

def origin2Conserve(ppt, rho, u1, u2, e, p, mu, kappa):

    def midVariants(u1, u2, T, mu, kappa, dx, dy):

        tauxx = np.zeros_like(u1)
        tauxy = np.zeros_like(u1)
        tauyy = np.zeros_like(u1)
        qx = np.zeros_like(u1)
        qy = np.zeros_like(u1)

        for i in range(1,len(tauxx[:,0])-1):
            for j in range(1,len(tauxx[0,:])-1):
                tauxx[i,j] = 2/3*mu[i,j]*( 2*(u1[i+1,j]-u1[i-1,j])/dx - (u2[i,j+1]-u2[i,j-1])/dy )/2
                tauxy[i,j] =     mu[i,j]*(   (u1[i+1,j]-u1[i-1,j])/dx + (u2[i,j+1]-u2[i,j-1])/dy )/2
                tauyy[i,j] = 2/3*mu[i,j]*( 2*(u2[i,j+1]-u2[i,j-1])/dy - (u1[i+1,j]-u1[i-1,j])/dx )/2
                qx[i,j] = -kappa[i,j]*(T[i+1,j] - T[i-1,j])/dx/2
                qy[i,j] = -kappa[i,j]*(T[i,j+1] - T[i,j-1])/dy/2
        for ppt in ([tauxx, tauxy, tauyy, qx, qy]):
            BCupdate(ppt)

        return tauxx, tauxy, tauyy, qx, qy

    T = e/ppt.cv
    Et = rho*(e+np.sqrt(u1**2+u2**2))

    tauxx, tauxy, tauyy, qx, qy = midVariants(u1, u2, T, mu, kappa, ppt.dx(), ppt.dy())

    U1 = rho
    U2 = rho*u1
    U3 = rho*u2
    U4 = Et
    U = np.hstack([U1, U2, U3, U4])

    E1 = U2
    E2 = rho*u1**2 + p - tauxx
    E3 = rho*u1*u2 - tauxy
    E4 = (Et + p)*u1 - u1*tauxx - u2*tauxy + qx
    E = np.hstack([E1, E2, E3, E4])

    F1 = U3
    F2 = rho*u1*u2 - tauxy
    F3 = rho*u2**2 + p - tauxy
    F4 = (Et + p)*u2 - u1*tauxy - u2*tauyy + qy
    F = np.hstack([F1, F2, F3, F4])

    return U, E, F

def init():

    def parameterInput():

        ppt = parameter()

        return ppt

    def oriVarAllocate():

        x = np.linspace(0, ppt.Lx, ppt.Nx)
        y = np.linspace(0, ppt.Ly, ppt.Ny)

        # Properties
        rho = np.zeros((ppt.Nx,ppt.Ny),dtype=float)
        u1 = np.zeros((ppt.Nx,ppt.Ny),dtype=float)
        u2 = np.zeros((ppt.Nx,ppt.Ny),dtype=float)
        T = np.zeros((ppt.Nx,ppt.Ny),dtype=float)

        return x, y, rho, u1, u2, T

    def flowFieldInit(ppt, rho, u1, u2, T):

        rho[:,:] = 0
        u1[:,:] = 0
        u2[:,:] = 0
        T[:,:] = 0

        p = pCalc(rho, T, ppt.R)
        e = eCalc(T, ppt.cv)
        mu = muCalc(T, ppt.T0, ppt.mu0)
        kappa = kappaCalc(mu, ppt.cp, ppt.Prn)
        return rho, u1, u2, T, p, e, mu, kappa

    ppt = parameterInput()
    x, y, rho, u1, u2, T = oriVarAllocate()
    rho, u1, u2, T, p, e, mu, kappa = flowFieldInit(ppt, rho, u1, u2, T)

    U, E, F = origin2Conserve(ppt, rho, u1, u2, e, p, mu, kappa)
    return ppt, U, E, F

def tscheme(ppt, U, E, F):

    # MacCormack scheme

    def artiVisc(p, U, Cy): #artificial viscosity
        # TODO

        S = np.zeros_like(U)
        for j in range(1,len(S)-1):
            S[j] = Cy * np.abs(p[j+1] - 2*p[j] + p[j-1]) / (p[j+1] + 2*p[j] + p[j-1]) * (U[j+1] - 2*U[j] + U[j-1])

        return S

    def preStep(E, F, dx, dy):

        predU = np.zeros_like(E)
        for n in range(E[0,:]):
            nE, nF = E[:,n], F[:,n]
            for i in range(len(predU[:,0])-1):
                for j in range(len(predU[0,:])-1):
                    predU[:,n][i,j] = (nE[i+1,j] - nE[i,j])/dx + (nF[i,j+1]-nF[i,j])/dy

        return predU

    def corrStep(preE, preF, dx, dy):

        corrdU = np.zeros_like(preE)
        for n in range(preE[0,:]):
            nE, nF = preE[:,n], preF[:,n]
            for i in range(1,len(corrdU[:,0])):
                for j in range(1,len(corrdU[0,:])):
                    corrdU[:,n][i,j] = (nE[i,j] - nE[i-1,j])/dx + (nF[i,j-1]-nF[i,j])/dy

        return corrdU


    # Calculate p
    rho, u1, u2, e = conserve2Origin(U)
    p = pCalc(rho, e/ppt.cv, ppt.R)

    # Calculate dt
    dc = np.append(ppt.dy/(np.sqrt(u1**2+u2**2) + np.sqrt(e/ppt.cv)).reshape(-1,1), ppt.dx/(np.sqrt(u1**2+u2**2) + np.sqrt(e/ppt.cv)).reshape(-1,1), axis=1)
    dt = ppt.CourantNum*np.min(dc)

    #pre-step
    predU = preStep(E, F, ppt.dx, ppt.dy)
    preU = np.hstack([U[:,0] + dt*predU[:,0] + artiVisc(p, U[:,0], ppt.Cy),
                      U[:,1] + dt*predU[:,1] + artiVisc(p, U[:,1], ppt.Cy),
                      U[:,2] + dt*predU[:,2] + artiVisc(p, U[:,2], ppt.Cy),
                      U[:,3] + dt*predU[:,3] + artiVisc(p, U[:,3], ppt.Cy)])

    preRho, preu1, preu2, pree = conserve2Origin(preU)
    preP = pCalc(preRho, pree/ppt.cv, ppt.R)
    mu = muCalc(pree/ppt.cv, ppt.T0, ppt.mu0)
    kappa = kappaCalc(mu, ppt.cp, ppt.Prn)
    __, preE, preF = origin2Conserve(ppt, preRho, preu1, preu2, pree, preP, mu, kappa)

    # #correct-step
    corrdU = corrStep(preE, preF, ppt.dx, ppt.dy)
    newU = np.hstack([U[:,0] + dt*(predU[:,0] + corrdU[:,0])/2 + artiVisc(preP, preU[:,0], ppt.Cy),
                       U[:,1] + dt*(predU[:,1] + corrdU[:,1])/2 + artiVisc(preP, preU[:,1], ppt.Cy),
                       U[:,2] + dt*(predU[:,2] + corrdU[:,2])/2 + artiVisc(preP, preU[:,2], ppt.Cy),
                       U[:,3] + dt*(predU[:,3] + corrdU[:,3])/2 + artiVisc(preP, preU[:,3], ppt.Cy)])

    # TODO: boundray condition
    # newU = BCSet(newU)

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

    ppt, U, E, F = init()
    xSet = np.zeros(1,dtype=float)
    iterNumb = 0
    t = 0

    while(t < ppt.tn):
        newU, dt = tscheme(ppt, U, E, E)

    # F1 = np.append(F1, newF1[np.newaxis,:], axis=0)
    # F2 = np.append(F2, newF2[np.newaxis,:], axis=0)
    # F3 = np.append(F3, newF3[np.newaxis,:], axis=0)
    # F4 = np.append(F4, newF4[np.newaxis,:], axis=0)


        t += dt
        rho, u1, u2, e = conserve2Origin(newU)
        p = pCalc(rho,e/ppt.cv,ppt.R)
        mu = muCalc(e/ppt.cv, ppt.T0, ppt.mu0)
        kappa = kappaCalc(mu, ppt.cp, ppt.Prn)

        if (iterNumb >= 1000): #defensive design
            break

        U, E, E = origin2Conserve(ppt, rho, u1, u2, e, p, mu, kappa)

# postProgress(xSet, E, eta, F1, F2, F3, F4, gamma, R)

    return 0

if __name__ == "__main__":
    main()