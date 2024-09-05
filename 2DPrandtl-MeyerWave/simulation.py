import numpy as np
import matplotlib.pyplot as plt

# this program is aim to calculate Prandtl-Meyer flow with numerical simulation.
#
# Finite Different Method
#
# Conservative form:    d(F)/dx              = -d(G)/dy              + J
#
# Coordinate transformation:    d(F)/dzeta   = -d(F)/deta*(deta/dx)  - 1/h*d(G)/deta
#
# Continuity:           d(ρu)/dt             = -d(ρu)/dx             + 0
#
# Momentum:             d(ρu^2+p)/dt         = -d(ρuv)/dx            + 0
#                       d(ρuv)/dt            = -d(ρv^2+p)/dx         + 0
#
# Energy:               d(ρu(e+V^2/2)+pu)/dt = -d(ρv(e+V^2/2)+pv)/dx + 0

def BCSet(F1, F2, F3, F4, rho, gamma, R):

    #TODO:边界条件设置

    rho, u1, u2, p, T, Ma = originalVariables(F1, F2, F3, F4, gamma, R)

    phi1 = np.arctan(u2/u1)
    Ma1cal = np.sqrt(np.square(u1)+np.square(u2))/a1cal
    fcal = np.sqrt((gamma+1)/(gamma-1))*np.arctan((gamma-1)/(gamma+1)*(np.square(Ma1cal)-1)) - np.arctan(np.square(Ma1cal)-1)
    PMfactor = fcal + phi1


    # #upper BC
    # F1[len(F1) - 1] = F1[len(F1) - 2]
    # F2[len(F2) - 1] = F2[len(F2) - 2]
    # F3[len(F3) - 1] = F3[len(F3) - 2]
    # F4[len(F4) - 1] = F4[len(F4) - 2]
    #
    # #lower BC
    # G1, G2, G3, G4 = fluxVariablesSet(F1, F2, F3, F4, rho, gamma)
    # predF = np.zeros(4)
    # i = 0
    # for preFG in ([[F1,G1], [F2,G2], [F3,G3], [F4,G4]]):
    #     Fc = preFG[0][0]
    #     Fs = preFG[0][1]
    #     Gc = preFG[1][0]
    #     Gc = preFG[0][1]
    #     predF[i] = preStep(FG[[0],[0]], FG[[0],[0]], h, detadx, deta)
    #     i += 1
    #
    # for corFG in ([[F1,G1], [F2,G2], [F3,G3], [F4,G4]]):
    #     Fc = corFG[0][0]
    #     Fs = corFG[0][1]
    #     Gc = corFG[1][0]
    #     Gc = corFG[0][1]
    #     predF[i] = preStep(FG[[0],[0]], FG[[0],[0]], h, detadx, deta)
    #     i += 1
    #
    # def preStep(Fc, FS, Gc, Gs, h, detadx, deta):
    #
    #     predF = -(detadx*(Fc - FS) + 1/h*(Gc - Gs))/deta
    #
    #     return predF
    #
    # def corStep(preFc, preFs, preGc, preGs, h, detadx, deta):
    #
    #     cordF = -(detadx*(preFc - preFs) + 1/h*(preGc - preGs))/deta
    #
    #     return cordF
    #
    #
    #
    # def preStep(F1, F2, F3, F4, rho, h, detadx, deta):
    #
    #     G1, G2, G3, G4 = fluxVariablesSet(F1, F2, F3, F4, rho, gamma)
    #
    #     predF1 = np.zeros_like(F1)
    #     predF2 = np.zeros_like(F2)
    #     predF3 = np.zeros_like(F3)
    #     predF4 = np.zeros_like(F4)
    #
    #     for j in range(1, len(predF1) - 1):
    #         predF1[j] = -(detadx*(F1[j] - F1[j-1]) + 1/h[j]*(G1[j] - G1[j-1]))/deta
    #         predF2[j] = -(detadx*(F2[j] - F2[j-1]) + 1/h[j]*(G2[j] - G2[j-1]))/deta
    #         predF3[j] = -(detadx*(F3[j] - F3[j-1]) + 1/h[j]*(G3[j] - G3[j-1]))/deta
    #         predF4[j] = -(detadx*(F4[j] - F4[j-1]) + 1/h[j]*(G4[j] - G4[j-1]))/deta
    #
    #     return predF1, predF2, predF3, predF4
    #
    # def correctionStep(preF1, preF2, preF3, preF4, preRho, h, detadx, deta):
    #
    #     preG1, preG2, preG3, preG4 = fluxVariablesSet(preF1, preF2, preF3, preF4, preRho, gamma)
    #
    #     cordF1 = np.zeros_like(preF1)
    #     cordF2 = np.zeros_like(preF2)
    #     cordF3 = np.zeros_like(preF3)
    #     cordF4 = np.zeros_like(preF3)
    #
    #     for j in range(1, len(cordF1[:,0]) - 1):
    #         cordF1[j] = -(detadx*(preF1[j+1] - preF1[j]) + 1/h[j]*(preG1[j+1] - preG1[j]))/deta
    #         cordF2[j] = -(detadx*(preF2[j+1] - preF2[j]) + 1/h[j]*(preG2[j+1] - preG2[j]))/deta
    #         cordF3[j] = -(detadx*(preF3[j+1] - preF3[j]) + 1/h[j]*(preG3[j+1] - preG3[j]))/deta
    #         cordF4[j] = -(detadx*(preF4[j+1] - preF4[j]) + 1/h[j]*(preG4[j+1] - preG4[j]))/deta
    #
    #     return cordF1, cordF2, cordF3, cordF4

    return F1, F2, F3, F4

def fluxVariablesSet(F1, F2, F3, F4, rho, gamma):

    G1 = rho*F3/F1
    G2 = F3
    G3 = rho*np.square(F3/F1) + F2 - np.square(F1)/rho
    G4 = gamma/(gamma-1)*(F2 - np.square(F1)/rho)*F3/F1 + rho/2*F3/F1*(np.square(F1/rho)+np.square(F3/rho))

    return G1, G2, G3, G4

def originalVariables(F1, F2, F3, F4, gamma, R):

    A = F3**2/(2*F1) - F4
    B = gamma/(gamma-1)*F1*F2
    C = -(gamma+1)/(2-gamma-1)*F1**3
    rho = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
    u1 = F1/rho
    u2 = F3/F1
    p = F2 - F1*u1
    T = p/(rho*R)
    #TODO: sound speed
    Ma = np.sqrt(np.square(u1) + np.square(u2))/339 # a0 = 339

    return rho, u1, u2, p, T, Ma

def init():

    def parameterInput():

        gamma = 1.4
        theta = 0.0934  #5.352degree
        Courant = 0.5
        # tStepNumb = 0#1600 #time step number and residual collection
        E = 10 #m
        R = 1.295 #g/L
        MaInit = 2

        return gamma, Courant, theta, E, R, MaInit

    def originalVariablesInit():

        # Grid parameter
        Lx = 65
        y = np.linspace(0, 40, 41)  #m

        # Properties
        rho = np.zeros((1,len(y)),dtype=float)
        u1 = np.zeros((1,len(y)),dtype=float)
        u2 = np.zeros((1,len(y)),dtype=float)
        # u3 = np.zeros((len(x),len(y)),dtype=float)
        T = np.zeros((1,len(y)),dtype=float)
        p = np.zeros((1,len(y)),dtype=float)

        return Lx, y, rho, u1, u2, T, p

    def conservVariablesInit(rho, u1, u2, p, gamma):

        F1 = rho*u1
        F2 = rho*np.square(u1)+p
        F3 = rho*u1*u2
        F4 = gamma/(gamma-1)*p*u1 + rho*u1*(np.square(u1)+np.square(u2))/2

        return F1, F2, F3, F4

    def flowFieldInit(Ma, u1, u2, p, rho, T):

        p[0,:] = 1.01 * 10**5 # N/m^2
        rho[0,:] = 1.23       # kg/m^3
        T[0,:] = 286.1        # K
        u1[0,:] = Ma * 339 #sound speed (m/s)
        u2[0,:] = 0

        return p, rho, T, u1, u2

    def gridInit(E, theta, x, y, ys, h):

        # physical space
        for i in range(len(ys)):
            if x[i] <= E:
                ys[i] = 0
                h[i] = y[len(y)-1]
            else:
                ys[i] = -(x[i] - E) * np.tan(theta)
                h[i] = y[len(y)-1] + (x[i] - E) * np.tan(theta) # H = y[len(y)-1]

        detadx = (h - y[len(y)-1] + ys)/h**2

        return h, detadx

    gamma, Courant, theta, E, R, MaInit = parameterInput()
    Lx, y, rho, u1, u2, T, p = originalVariablesInit()
    # h, detadx = gridInit(E, theta, x, y, ys, h)
    p, rho, T, u1, u2 = flowFieldInit(MaInit, u1, u2, p, rho, T)

    F1, F2, F3, F4 = conservVariablesInit(rho, u1, u2, p, gamma)

    return gamma, R, theta, Courant, Lx, E, y, rho, u1, u2, T, p, F1, F2, F3, F4

def tscheme(currLx, theta, E, y, F1, F2, F3, F4, Courant, gamma, R):
    # MacCormack scheme
    rho, u1, u2, p, T, Ma = originalVariables(F1, F2, F3, F4, gamma, R)

    def calh(currLx, y, E, theta):
        if currLx <= E:
            h = y[len(y)-1]
        else:
            h = y[len(y)-1] + (currLx - E) * np.tan(theta)
        return h

    def caldetadx(currLx, E, h, theta, eta):
        detadx = np.zeros_like(y)
        for j in range(len(detadx)):
            if currLx <= E:
                detadx[j] = 0
            else:
                detadx[j] = (1 - eta[j])*np.tan(theta)/h
        return detadx

    def caldzeta(y, u1, u2, Ma):
        theta = np.arctan(u2/u1)
        niu = np.arcsin(1/Ma)
        dzeta = Courant*(y[1]-y[0])/np.max([np.abs(np.tan(theta[:]+niu[:])),np.abs(np.tan(theta[:]-niu[:]))])
        return dzeta

    #artificial viscosity
    def artiVisc(p, U):
        Cx = 0.2
        S = np.zeros_like(U)
        for j in range(1,len(S)-1):
            S[j] = Cx * np.abs(p[j+1] - 2*p[j] + p[j-1]) / (p[j+1] + 2*p[j] + p[j-1]) * (U[j+1] - 2*U[j] + U[j-1])
        return S

    def preStep(F1, F2, F3, F4, rho, h, detadx, eta):

        G1, G2, G3, G4 = fluxVariablesSet(F1, F2, F3, F4, rho, gamma)

        predF1 = np.zeros_like(F1)
        predF2 = np.zeros_like(F2)
        predF3 = np.zeros_like(F3)
        predF4 = np.zeros_like(F4)

        for j in range(0, len(predF1) - 1):
            if 0 < j <= len(predF1)-1:
                predF1[j] = -(detadx[j]*(F1[j] - F1[j-1]) + 1/h*(G1[j] - G1[j-1]))/(eta[j] - eta[j-1])
                predF2[j] = -(detadx[j]*(F2[j] - F2[j-1]) + 1/h*(G2[j] - G2[j-1]))/(eta[j] - eta[j-1])
                predF3[j] = -(detadx[j]*(F3[j] - F3[j-1]) + 1/h*(G3[j] - G3[j-1]))/(eta[j] - eta[j-1])
                predF4[j] = -(detadx[j]*(F4[j] - F4[j-1]) + 1/h*(G4[j] - G4[j-1]))/(eta[j] - eta[j-1])
            if j <= 0:
                predF1[j] = -(detadx[j]*(F1[j+1] - F1[j]) + 1/h*(G1[j+1] - G1[j]))/(eta[j+1] - eta[j])
                predF2[j] = -(detadx[j]*(F2[j+1] - F2[j]) + 1/h*(G2[j+1] - G2[j]))/(eta[j+1] - eta[j])
                predF3[j] = -(detadx[j]*(F3[j+1] - F3[j]) + 1/h*(G3[j+1] - G3[j]))/(eta[j+1] - eta[j])
                predF4[j] = -(detadx[j]*(F4[j+1] - F4[j]) + 1/h*(G4[j+1] - G4[j]))/(eta[j+1] - eta[j])

        return predF1, predF2, predF3, predF4

    def correctionStep(preF1, preF2, preF3, preF4, preRho, h, detadx, deta):

        preG1, preG2, preG3, preG4 = fluxVariablesSet(preF1, preF2, preF3, preF4, preRho, gamma)

        cordF1 = np.zeros_like(preF1)
        cordF2 = np.zeros_like(preF2)
        cordF3 = np.zeros_like(preF3)
        cordF4 = np.zeros_like(preF3)

        for j in range(1, len(cordF1) - 1):
            if 0 <= j < len(cordF1)-1:
                cordF1[j] = -(detadx[j]*(preF1[j+1] - preF1[j]) + 1/h*(preG1[j+1] - preG1[j]))/(eta[j+1] - eta[j])
                cordF2[j] = -(detadx[j]*(preF2[j+1] - preF2[j]) + 1/h*(preG2[j+1] - preG2[j]))/(eta[j+1] - eta[j])
                cordF3[j] = -(detadx[j]*(preF3[j+1] - preF3[j]) + 1/h*(preG3[j+1] - preG3[j]))/(eta[j+1] - eta[j])
                cordF4[j] = -(detadx[j]*(preF4[j+1] - preF4[j]) + 1/h*(preG4[j+1] - preG4[j]))/(eta[j+1] - eta[j])
            if j >= len(cordF1)-1:
                cordF1[j] = -(detadx[j]*(preF1[j] - preF1[j-1]) + 1/h*(preG1[j] - preG1[j-1]))/(eta[j] - eta[j-1])
                cordF2[j] = -(detadx[j]*(preF2[j] - preF2[j-1]) + 1/h*(preG2[j] - preG2[j-1]))/(eta[j] - eta[j-1])
                cordF3[j] = -(detadx[j]*(preF3[j] - preF3[j-1]) + 1/h*(preG3[j] - preG3[j-1]))/(eta[j] - eta[j-1])
                cordF4[j] = -(detadx[j]*(preF4[j] - preF4[j-1]) + 1/h*(preG4[j] - preG4[j-1]))/(eta[j] - eta[j-1])

        return cordF1, cordF2, cordF3, cordF4

    h = calh(currLx, y, E, theta)
    dzeta = caldzeta(y, u1, u2, Ma)
    eta = (y + (currLx - E)*np.tan(theta))/h
    detadx = caldetadx(currLx, E, h, theta, eta)

    #pre-step
    predF1, predF2, predF3, predF4 = preStep(F1, F2, F3, F4, rho, h, detadx, eta)
    preF1, preF2, preF3, preF4 = (
        F1 + dzeta*predF1 + artiVisc(p, F1),
        F2 + dzeta*predF2 + artiVisc(p, F2),
        F3 + dzeta*predF3 + artiVisc(p, F3),
        F4 + dzeta*predF4 + artiVisc(p, F4))

    # preF1, preF2, preF3, preF4 = BCSet(preF1, preF2, preF3, preF4, rho, gamma, R)

    preRho, preu1, preu2, preP, preT, preMa = originalVariables(preF1, preF2, preF3, preF4, gamma, R)

    # #correct-step
    cordF1, cordF2, cordF3, cordF4 = correctionStep(preF1, preF2, preF3, preF4, preRho, h, detadx, eta)
    newF1, newF2, newF3, newF4 = (
        F1 + (predF1 + cordF1)*0.5*dzeta+ artiVisc(preP, preF1),
        F2 + (predF2 + cordF2)*0.5*dzeta+ artiVisc(preP, preF2),
        F3 + (predF3 + cordF3)*0.5*dzeta+ artiVisc(preP, preF3),
        F4 + (predF4 + cordF4)*0.5*dzeta+ artiVisc(preP, preF4))

    #BCSet(U1 + (predU1 + cordU1)*0.5*dt + corS1, U2 + (predU2 + cordU2)*0.5*dt + corS2, U3 + (predU3 + cordU3)*0.5*dt + corS3, A, gamma)

    # newF1, newF2, newF3, newF4 = BCSet(newF1, newF2, newF3, newF4, rho, gamma)

    return newF1, newF2, newF3, newF4, dzeta

def postProgress(F1, F2, F3, F4, totalx):

    # rho, u1, T = originalVariables(F1, F2, F3, F4, A, gamma)
    # p = rho * T
    # Ma = u1 / np.sqrt(T)
    #
    # m = rho * u1 * A

    def printData():
        print("------------solve complete.------------")
        # print("iteration or temporal advancement times:", timeStepNumb)
        print("total physical space:", totalx)

        print("---------------------------------------")
        # print("residual:", residual)
        # print("ρ:", rho)
        # print("u1:", u1)
        # print("T:", T)
        # print("p", p)
        # print("Ma", Ma)
        print("F1:", F1)
        return

    def drawData():
        plt.figure()
        # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
        plt.plot(x, Ma, '-o', linewidth=1.0, color='black', markersize=1)
        # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
        plt.show()
        return

    printData()

    # drawData()

    # print("residual:", residual)
    return 0

def main():

    gamma, R, theta, Courant, Lx, E, y, rho, u1, u2, T, p, F1, F2, F3, F4 = init()
    # totalt = 0
    # residual = np.array([],dtype=float)
    totalx = 0
    while(totalx < Lx):
        newF1, newF2, newF3, newF4, dzeta = tscheme(totalx, theta, E, y, F1[len(F1)-1,:], F2[len(F2)-1,:], F3[len(F3)-1,:],F4[len(F4)-1,:], Courant, gamma, R)
        if (totalx>=12.900):
            newF1, newF2, newF3, newF4, dzeta = tscheme(totalx, theta, E, y, F1[len(F1)-1,:], F2[len(F2)-1,:], F3[len(F3)-1,:],F4[len(F4)-1,:], Courant, gamma, R)
        # R = np.max([np.max(newU1 - U1), np.max(newU2 - U2), np.max(newU3 - U3)]) / dt
        # residual = np.append(R,residual)

        F1 = np.append(F1, newF1[np.newaxis,:], axis=0)
        F2 = np.append(F2, newF2[np.newaxis,:], axis=0)
        F3 = np.append(F3, newF3[np.newaxis,:], axis=0)
        F4 = np.append(F4, newF4[np.newaxis,:], axis=0)

        totalx += dzeta
        # totalt += dt

    postProgress(F1, F2, F3, F4, totalx)

    return 0

if __name__ == "__main__":
    main()