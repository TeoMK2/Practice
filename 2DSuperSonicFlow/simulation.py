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

def BCSet(F1, F2, F3, F4, gamma, R, theta):

    return F1, F2, F3, F4

def conserve2Origin(F1, F2, F3, F4, gamma, R):

    return rho, u1, u2, p, T, Ma

def origin2Conserve(rho, u1, u2, Et, p, tauxx, tauxy, tauyy, qx, qy):

    U1 = rho
    U2 = rho*u1
    U3 = rho*u2
    U4 = Et

    E1 = U2
    E2 = rho*u1**2 + p - tauxx
    E3 = rho*u1*u2 - tauxy
    E4 = (Et + p)*u1 - u1*tauxx - u2*tauxy + qx

    F1 = U3
    F2 = rho*u1*u2 - tauxy
    F3 = rho*u2**2 + p - tauxy
    F4 = (Et + p)*u2 - u1*tauxy - u2*tauyy + qy

    return U, E, F

def fluxVariablesSet(F1, F2, F3, F4, rho, gamma):

    return G1, G2, G3, G4

def init():

    def parameterInput():

        return gamma, Courant, theta, E, MaInit, Cy

    def originalVariablesInit():

        # Grid parameter
        Lx = 65
        Ly = 40
        Ny = 401
        eta = np.linspace(0, 1, Ny)
        # Properties
        rho = np.zeros((1,Ny),dtype=float)
        u1 = np.zeros((1,Ny),dtype=float)
        u2 = np.zeros((1,Ny),dtype=float)
        # u3 = np.zeros((len(x),len(y)),dtype=float)
        T = np.zeros((1,Ny),dtype=float)
        p = np.zeros((1,Ny),dtype=float)

        return Lx, Ly, eta, rho, u1, u2, T, p

    def flowFieldInit(MaInit, gamma, u1, u2, p, rho, T):

        return p, rho, T, u1, u2, R

    gamma, Courant, theta, E, MaInit, Cy = parameterInput()
    Lx, Ly, eta, rho, u1, u2, T, p = originalVariablesInit()
    p, rho, T, u1, u2, R = flowFieldInit(MaInit, gamma, u1, u2, p, rho, T)

    F1, F2, F3, F4 = origin2Conserve(rho, u1, u2, p, gamma)

    return gamma, R, theta, Courant, Cy, Lx, E, Ly, eta, rho, u1, u2, T, p, F1, F2, F3, F4

def tscheme(theta, h, detadx, eta, F1, F2, F3, F4, Courant, gamma, R, Cy):
    # MacCormack scheme

    def artiVisc(p, U, Cy): #artificial viscosity
        S = np.zeros_like(U)
        # for j in range(1,len(S)-1):
        #     S[j] = Cy * np.abs(p[j+1] - 2*p[j] + p[j-1]) / (p[j+1] + 2*p[j] + p[j-1]) * (U[j+1] - 2*U[j] + U[j-1])

        for j in range(len(S)):
            if 0 < j < len(S)-1:
                S[j] = Cy * np.abs(p[j+1] - 2*p[j] + p[j-1]) / (p[j+1] + 2*p[j] + p[j-1]) * (U[j+1] - 2*U[j] + U[j-1])
            elif j == 0:
                S[j] = Cy * np.abs(p[j+2] - 2*p[j+1] + p[j]) / (p[j+2] + 2*p[j+1] + p[j]) * (U[j+2] - 2*U[j+1] + U[j])
            else:
                S[j] = 0

        return S

    def preStep(F1, F2, F3, F4, rho, h, detadx, deta):

        G1, G2, G3, G4 = fluxVariablesSet(F1, F2, F3, F4, rho, gamma)

        predF1 = np.zeros_like(F1)
        predF2 = np.zeros_like(F2)
        predF3 = np.zeros_like(F3)
        predF4 = np.zeros_like(F4)

        for j in range(len(predF1)):
            if 0 <= j < len(predF1)-1:
                predF1[j] = -(detadx[j]*(F1[j+1] - F1[j]) + 1/h*(G1[j+1] - G1[j]))/deta
                predF2[j] = -(detadx[j]*(F2[j+1] - F2[j]) + 1/h*(G2[j+1] - G2[j]))/deta
                predF3[j] = -(detadx[j]*(F3[j+1] - F3[j]) + 1/h*(G3[j+1] - G3[j]))/deta
                predF4[j] = -(detadx[j]*(F4[j+1] - F4[j]) + 1/h*(G4[j+1] - G4[j]))/deta
            if j >= len(predF1)-1:
                predF1[j] = predF1[j-1]#-(detadx[j]*(F1[j] - F1[j-1]) + 1/h*(G1[j] - G1[j-1]))/deta
                predF2[j] = predF2[j-1]#-(detadx[j]*(F2[j] - F2[j-1]) + 1/h*(G2[j] - G2[j-1]))/deta
                predF3[j] = predF3[j-1]#-(detadx[j]*(F3[j] - F3[j-1]) + 1/h*(G3[j] - G3[j-1]))/deta
                predF4[j] = predF4[j-1]#-(detadx[j]*(F4[j] - F4[j-1]) + 1/h*(G4[j] - G4[j-1]))/deta

        return predF1, predF2, predF3, predF4

    def correctionStep(preF1, preF2, preF3, preF4, preRho, h, detadx, deta):

        preG1, preG2, preG3, preG4 = fluxVariablesSet(preF1, preF2, preF3, preF4, preRho, gamma)

        cordF1 = np.zeros_like(preF1)
        cordF2 = np.zeros_like(preF2)
        cordF3 = np.zeros_like(preF3)
        cordF4 = np.zeros_like(preF3)

        for j in range(len(cordF1)):
            if 0 < j <= len(cordF1)-1:
                cordF1[j] = -(detadx[j]*(preF1[j] - preF1[j-1]) + 1/h*(preG1[j] - preG1[j-1]))/deta
                cordF2[j] = -(detadx[j]*(preF2[j] - preF2[j-1]) + 1/h*(preG2[j] - preG2[j-1]))/deta
                cordF3[j] = -(detadx[j]*(preF3[j] - preF3[j-1]) + 1/h*(preG3[j] - preG3[j-1]))/deta
                cordF4[j] = -(detadx[j]*(preF4[j] - preF4[j-1]) + 1/h*(preG4[j] - preG4[j-1]))/deta
            if j <= 0:
                cordF1[j] = -(detadx[j]*(preF1[j+1] - preF1[j]) + 1/h*(preG1[j+1] - preG1[j]))/deta
                cordF2[j] = -(detadx[j]*(preF2[j+1] - preF2[j]) + 1/h*(preG2[j+1] - preG2[j]))/deta
                cordF3[j] = -(detadx[j]*(preF3[j+1] - preF3[j]) + 1/h*(preG3[j+1] - preG3[j]))/deta
                cordF4[j] = -(detadx[j]*(preF4[j+1] - preF4[j]) + 1/h*(preG4[j+1] - preG4[j]))/deta

        return cordF1, cordF2, cordF3, cordF4




    #STEP1: calculate ρ or p
    #STEP2: calculate u1, u2
    #STEP3: calculate T
    # LOOP

    rho, u1, u2, p, T, Ma = conserve2Origin(F1, F2, F3, F4, gamma, R)
    deta = eta[1]-eta[0]
    dksi = caldksi(Courant, deta, h, Ma)

    #pre-step
    predF1, predF2, predF3, predF4 = preStep(F1, F2, F3, F4, rho, h, detadx, deta)
    preF1, preF2, preF3, preF4 = (
        F1 + dksi*predF1 + artiVisc(p, F1, Cy),
        F2 + dksi*predF2 + artiVisc(p, F2, Cy),
        F3 + dksi*predF3 + artiVisc(p, F3, Cy),
        F4 + dksi*predF4 + artiVisc(p, F4, Cy))

    preRho, preu1, preu2, preP, preT, preMa = conserve2Origin(preF1, preF2, preF3, preF4, gamma, R)

    # #correct-step
    cordF1, cordF2, cordF3, cordF4 = correctionStep(preF1, preF2, preF3, preF4, preRho, h, detadx, deta)
    newF1, newF2, newF3, newF4 = (
        F1 + (predF1 + cordF1)*0.5*dksi+ artiVisc(preP, preF1, Cy),
        F2 + (predF2 + cordF2)*0.5*dksi+ artiVisc(preP, preF2, Cy),
        F3 + (predF3 + cordF3)*0.5*dksi+ artiVisc(preP, preF3, Cy),
        F4 + (predF4 + cordF4)*0.5*dksi+ artiVisc(preP, preF4, Cy))

    newF1, newF2, newF3, newF4 = BCSet(newF1, newF2, newF3, newF4, gamma, R, theta)

    return newF1, newF2, newF3, newF4, dksi

def postProgress(xSet, E, eta, F1, F2, F3, F4, gamma, R):

    rho, u1, u2, p, T, Ma = conserve2Origin(F1, F2, F3, F4, gamma, R)

    def printData():
        print("------------solve complete.------------")
        # print("iteration or temporal advancement times:", timeStepNumb)
        # print("total physical space:", xSet)

        print("---------------------------------------")
        # print("residual:", residual)
        # print("ρ:", rho)
        # print("u1:", u1)
        # print("T:", T)
        # print("p", p)
        # print("Ma", Ma)
        # print("F1:", F1)
        return

    def drawData(x, y, data, name):
        plt.figure()
        # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
        pic = plt.contourf(x,y,data,alpha=0.8,cmap='jet')
        # plt.plot(x, Ma, '-o', linewidth=1.0, color='black', markersize=1)
        plt.colorbar(pic)
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Evolution of ' + name)
        # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
        plt.show()
        return

    # printData()

    drawData(x, y, Ma, 'Ma')
    drawData(x, y, rho, 'rho')
    drawData(x, y, u1, 'u1')
    drawData(x, y, u2, 'u2')
    drawData(x, y, T, 'T')
    drawData(x, y, p, 'p')

    return 0

def main():

     = init()
    xSet = np.zeros(1,dtype=float)
    iterNumb = 0

    newF1, newF2, newF3, newF4, dksi = tscheme(theta, h, detadx, eta, F1[len(F1)-1,:], F2[len(F2)-1,:], F3[len(F3)-1,:],F4[len(F4)-1,:], Courant, gamma, R, Cy)

    F1 = np.append(F1, newF1[np.newaxis,:], axis=0)
    F2 = np.append(F2, newF2[np.newaxis,:], axis=0)
    F3 = np.append(F3, newF3[np.newaxis,:], axis=0)
    F4 = np.append(F4, newF4[np.newaxis,:], axis=0)

    xSet = np.append(xSet, xSet[-1] + dksi)
    iterNumb += 1

    if (iterNumb >= 1000): #defensive design
        break

    postProgress(xSet, E, eta, F1, F2, F3, F4, gamma, R)

    return 0

if __name__ == "__main__":
    main()