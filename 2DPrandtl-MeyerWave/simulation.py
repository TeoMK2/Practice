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




def BCSet(U1, U2, U3, A,gamma):

    #TODO:边界条件设置

    # # inlet
    # U1[0] = A[0]   #fix rho
    # U2[0] = 2*U2[1] - U2[2] #interplot
    # U3[0] = U1[0]*(1/(gamma-1)+gamma/2*np.square(U2[0]/U1[0]))  #T=1
    # # outlet
    # U1[len(U1) - 1] = 2 * U1[len(U1) - 2] - U1[len(U1) - 3]
    # U2[len(U2) - 1] = 2 * U2[len(U2) - 2] - U2[len(U2) - 3]
    # # U3[len(U3) - 1] = 2 * U3[len(U3) - 2] - U3[len(U3) - 3]
    # U3[len(U3) - 1] = 0.6784*A[len(U3) - 1]/(gamma-1) + gamma/2*np.square(U2[len(U3) - 1])/U1[len(U3) - 1]     # p=0.6784 is constant
    return U1, U2, U3

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

    return rho, u1, u2, p, T

def init():

    def parameterInput():

        gamma = 0#1.4
        theta = 5.352   #degree
        Courant = 0#0.5
        tStepNumb = 0#1600 #time step number and residual collection
        E = 3

        return gamma, Courant, tStepNumb, E, theta

    def originalVariablesInit():

        # Grid parameter
        x = np.linspace(0,65,66)    #m
        y = np.linspace(0, 40, 41)  #m
        h = np.zeros_like(x)
        ys = np.zeros_like(x)

        # Properties
        rho = np.zeros((len(x),len(y)),dtype=float)
        u1 = np.zeros((len(x),len(y)),dtype=float)
        u2 = np.zeros((len(x),len(y)),dtype=float)
        # u3 = np.zeros((len(x),len(y)),dtype=float)
        T = np.zeros((len(x),len(y)),dtype=float)
        p = np.zeros((len(x),len(y)),dtype=float)

        return x, y, ys, h, rho, u1, u2, T, p

    def conservVariablesInit(rho, u1, u2, p, gamma):

        F1 = rho*u1
        F2 = rho*np.square(u1)+p
        F3 = rho*u1*u2
        F4 = gamma/(gamma-1)*p*u1 + rho*u1*(np.square(u1)+np.square(u2))/2

        return F1, F2, F3, F4

    def flowFieldInit(p, rho, T):

        p[0,:] = 1.01 * 10**5 # N/m^2
        rho[0,:] = 1.23       # kg/m^3
        T[0,:] = 286.1        # K

        return p, rho, T

    def gridInit(E, theta, x, y, ys, h):

        # physical space
        for i in range(len(ys)):
            if x[i] >= E:
                ys[i] = 0
                h[i] = y[len(y)-1]
            else:
                ys[i] = -(x[i] - E) * np.tan(theta)
                h[i] = y[len(y)-1] + (x[i] - E) * np.tan(theta) # H = y[len(y)-1]

        dzetadx = np.zeros_like(x)
        dzetady = np.zeros_like(y)

        dzetadx[:] = 1
        dzetady[:] = 0
        detadx = (h - y - ys)/h**2
        detady = 1/h

        return dzetadx, detadx, detady, dzetady

    gamma, Courant, tStepNumb, E, theta = parameterInput()
    x, y, ys, h, rho, u1, u2, T, p = originalVariablesInit()
    dzetadx, detadx, detady, dzetady = gridInit(theta, E, x, y, ys, h)
    p, rho, T = flowFieldInit(p, rho, T)

    F1, F2, F3, F4 = conservVariablesInit(rho, u1, u2, p, gamma)

    return gamma, Courant, tStepNumb, x, h, rho, u1, u2, T, p, F1, F2, F3, F4, dzetadx, detadx, detady, dzetady

def tscheme(dzetadx, detadx, detady, dzetady, F1, F2, F3 ,F4, Courant, gamma, R):
    # MacCormack scheme
    rho, u1, u2, p, T = originalVariables(F1, F2, F3, F4, gamma, R)
    # dt = np.min(Courant * (x[1] - x[0]) / (u1 + np.sqrt(T)))#0.0267#
    # dx = 1/(x[1] - x[0])
    # dt = Courant * (x[1] - x[0]) / (u1 + np.sqrt(T))
    # dtdx = dt / (x[1] - x[0])
    deta = 0

    #artificial viscosity
    def artiVisc(p, U):
        Cx = 0.2
        S = np.zeros_like(U)
        for i in range(1,len(S)-1):
            S[i] = Cx * np.abs(p[i+1] - 2*p[i] + p[i-1]) / (p[i+1] + 2*p[i] + p[i-1]) * (U[i+1] - 2*U[i] + U[i-1])

        return S

    def preStep(F1, F2, F3, F4, rho, h, detadx, deta):

        G1, G2, G3, G4 = fluxVariablesSet(F1, F2, F3, F4, rho, gamma)

        predF1 = np.zeros_like(F1)
        predF2 = np.zeros_like(F2)
        predF3 = np.zeros_like(F3)
        predF4 = np.zeros_like(F4)

        for i in range(1, len(predF1) - 1):
            predF1[i] = (detadx*(F1[i] - F1[i-1]) + 1/h[i]*(G1[i] - G1[i-1]))/deta
            predF2[i] = (detadx*(F2[i] - F2[i-1]) + 1/h[i]*(G2[i] - G2[i-1]))/deta
            predF3[i] = (detadx*(F3[i] - F3[i-1]) + 1/h[i]*(G3[i] - G3[i-1]))/deta
            predF4[i] = (detadx*(F4[i] - F4[i-1]) + 1/h[i]*(G4[i] - G4[i-1]))/deta

        return predF1, predF2, predF3, predF4

    def correctionStep(preF1, preF2, preF3, preF4, preRho, h, detadx, deta):

        preG1, preG2, preG3, preG4 = fluxVariablesSet(preF1, preF2, preF3, preF4, preRho, gamma)

        cordF1 = np.zeros_like(preF1)
        cordF2 = np.zeros_like(preF2)
        cordF3 = np.zeros_like(preF3)
        cordF4 = np.zeros_like(preF3)

        for i in range(1, len(cordF1) - 1):
            cordF1[i] = (detadx*(preF1[i+1] - preF1[i]) + 1/h[i]*(preG1[i+1] - preG1[i]))/deta
            cordF2[i] = (detadx*(preF2[i+1] - preF2[i]) + 1/h[i]*(preG2[i+1] - preG2[i]))/deta
            cordF3[i] = (detadx*(preF3[i+1] - preF3[i]) + 1/h[i]*(preG3[i+1] - preG3[i]))/deta
            cordF4[i] = (detadx*(preF4[i+1] - preF4[i]) + 1/h[i]*(preG4[i+1] - preG4[i]))/deta

        return cordF1, cordF2, cordF3, cordF4

    #pre-step
    predF1, predF2, predF3, predF4 = preStep(F1, F2, F3, F4, detadx, deta, rho, h)
    preF1, preF2, preF3, preF4 = (
        F1 + dzeta*predF1 + artiVisc(p, F1),
        F2 + dzeta*predF2 + artiVisc(p, F2),
        F3 + dzeta*predF3 + artiVisc(p, F3),
        F4 + dzeta*predF4 + artiVisc(p, F4))

    # preU1, preU2, preU3 = BCSet(U1 + predU1*dt + preS1, U2 + predU2*dt + preS2, U3 + predU3*dt + preS3, A, gamma)

    preRho, preu1, preu2, preP, preT = originalVariables(preF1, preF2, preF3, preF4, gamma, R)

    #correct-step
    cordF1, cordF2, cordF3, cordF4 = correctionStep(preF1, preF2, preF3, preF4, detadx, deta, preRho, h)
    newF1, newF2, newF3, newF4 = (
        F1 + (predF1 + cordF1)*0.5*dzeta+ artiVisc(preP, preF1),
        F2 + (predF2 + cordF2)*0.5*dzeta+ artiVisc(preP, preF2),
        F3 + (predF3 + cordF3)*0.5*dzeta+ artiVisc(preP, preF3),
        F4 + (predF4 + cordF4)*0.5*dzeta+ artiVisc(preP, preF4))

    #BCSet(U1 + (predU1 + cordU1)*0.5*dt + corS1, U2 + (predU2 + cordU2)*0.5*dt + corS2, U3 + (predU3 + cordU3)*0.5*dt + corS3, A, gamma)

    return newF1, newF2, newF3, newF4

# def postProgress(U1, U2, U3, x, A, gamma, timeStepNumb, totalt, residual):
#
#     #TODO: post-progress
#
#     rho, u1, T = originalVariables(F1, F2, F3, F4, A, gamma)
#     p = rho * T
#     Ma = u1 / np.sqrt(T)
#
#     m = rho * u1 * A
#
#     def printData():
#         print("------------solve complete.------------")
#         print("iteration or temporal advancement times:", timeStepNumb)
#         print("total physical time:", totalt)
#
#         print("---------------------------------------")
#         print("residual:", residual)
#         print("ρ:", rho)
#         print("u1:", u1)
#         print("T:", T)
#         print("p", p)
#         print("Ma", Ma)
#         return
#
#     def drawData():
#         plt.figure()
#         # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
#         plt.plot(x, Ma, '-o', linewidth=1.0, color='black', markersize=1)
#         # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
#         plt.show()
#         return
#
#     printData()
#
#     drawData()
#
#     # print("residual:", residual)
#     return 0

def main():

    gamma, Courant, tStepNumb, x, h, rho, u1, u2, T, p, F1, F2, F3, F4, dzetadx, detadx, detady, dzetady = init()
    # totalt = 0
    # residual = np.array([],dtype=float)
    for t in range(tStepNumb):
        newF1, newF2, newF3, newF4, dt = tscheme(dzetadx, detadx, detady, dzetady, F1, F2, F3 ,F4, Courant, gamma)

        # R = np.max([np.max(newU1 - U1), np.max(newU2 - U2), np.max(newU3 - U3)]) / dt
        # residual = np.append(R,residual)

        U1, U2, U3 = newU1, newU2, newU3
        # totalt += dt
        # if t == tStepNumb-1:
        #     postProgress(U1, U2, U3, x, A, gamma, tStepNumb, totalt, residual)

    return 0

if __name__ == "__main__":
    main()