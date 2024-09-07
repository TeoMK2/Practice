import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve   #Nonlinear Equation Solver

# this program is aim to calculate Prandtl-Meyer flow with numerical simulation.
#
# Finite Different Method
#
# Conservative form:    d(F)/dx              = -d(G)/dy              + J
#
# Coordinate transformation:    d(F)/dksi   = -d(F)/deta*(deta/dx)  - 1/h*d(G)/deta
#
# Continuity:           d(ρu)/dt             = -d(ρu)/dx             + 0
#
# Momentum:             d(ρu^2+p)/dt         = -d(ρuv)/dx            + 0
#                       d(ρuv)/dt            = -d(ρv^2+p)/dx         + 0
#
# Energy:               d(ρu(e+V^2/2)+pu)/dt = -d(ρv(e+V^2/2)+pv)/dx + 0

def geometry(currLx, y, eta, E, theta0):
    if currLx <= E:
        theta = 0
        h = y[len(y)-1]
        detadx = np.zeros(shape=eta.shape)
    else:
        theta = theta0
        h = y[len(y)-1] + (currLx - E)*np.tan(theta)
        detadx = (1-eta)*np.tan(theta)/h

    return theta, h, detadx

def BCSet(F1b, F2b, F3b, F4b, gamma, R, theta):

    #TODO:边界条件设置
    def calMaAct(f, gamma, Ma_guess):

        # Define the equation for Ma based on the provided formula
        def equation(Ma):
            return np.sqrt((gamma + 1) / (gamma - 1)) * np.arctan(np.sqrt((gamma - 1) / (gamma + 1) * (Ma**2 - 1))) - np.arctan(np.sqrt(Ma**2 - 1)) - f

        # Solve for Ma using fsolve
        Ma_solution = fsolve(equation, Ma_guess)

        return  Ma_solution[0]

    def calMaAct2(f, gamma):
        # 二分法求马赫数
        xm = np.array([1,2.5,5])
        eps = 1e-5
        n = 0
        while abs(xm[0]-xm[2])>= eps:
            fm = np.sqrt((gamma+1)/(gamma-1))*np.arctan(np.sqrt((gamma-1)/(gamma+1)*(xm**2-1)))-np.arctan(np.sqrt(xm**2-1))-f
            if fm[1]*fm[2]<0:
                xm = np.array([xm[1],(xm[1]+xm[2])/2,xm[2]])
            else:
                if fm[1]*fm[2]>0:
                    xm = np.array([xm[0],(xm[0]+xm[1])/2,xm[1]])
                else:
                    xm = np.array([xm[1],xm[1],xm[1]])
            n=n+1
            if n>1000:
                print('Warning ! Max iteration for Ma2')
                break
        return  xm[0]

    rhoCal, u1Cal, u2Cal, pCal, TCal, MaCal = conserve2Origin(F1b, F2b, F3b, F4b, gamma, R)

    phi1 = theta + np.arctan(np.abs(u2Cal/u1Cal))

    fCal = np.sqrt( (gamma + 1)/(gamma - 1) ) * np.arctan(np.sqrt( (gamma - 1)/(gamma + 1) * (MaCal**2-1))) - np.arctan(np.sqrt(MaCal**2-1))
    fAct = fCal + phi1

    MaAct = calMaAct2(fAct, gamma)#calMaAct(fAct, gamma, MaCal)
    pAct = pCal * ( ((1+(gamma-1)/2)*MaCal**2) / ((1+(gamma-1)/2)*MaAct**2) )**(gamma/(gamma-1))
    TAct = TCal *   ((1+(gamma-1)/2)*MaCal**2) / ((1+(gamma-1)/2)*MaAct**2)
    rhoAct = pAct/R/TAct
            # Ma2*np.sqrt(gamma*R*T2/(1+np.tan(theta)**2))
    u1Act = MaAct * np.sqrt(gamma*R*TAct/(1+np.square(np.tan(theta))))
    u2Act = -u1Act*np.tan(theta)
    F1b, F2b, F3b, F4b = origin2Conserve(rhoAct, u1Act, u2Act, pAct, gamma)

    return F1b, F2b, F3b, F4b

def conserve2Origin(F1, F2, F3, F4, gamma, R):

    A = F3**2/2/F1 - F4
    B = gamma/(gamma-1)*F1*F2
    C = -(gamma+1)/2/(gamma-1)*F1**3
    rho = (-B + np.sqrt(B**2 - 4*A*C))/2/A
    u1 = F1/rho
    u2 = F3/F1
    p = F2 - F1*u1
    T = p/rho/R
    Ma = np.sqrt(np.square(u1) + np.square(u2))/np.sqrt(gamma*T*R)

    return rho, u1, u2, p, T, Ma

def origin2Conserve(rho, u1, u2, p, gamma):

    F1 = rho*u1
    F2 = rho*np.square(u1)+p
    F3 = rho*u1*u2
    F4 = gamma/(gamma-1)*p*u1 + rho*u1*(np.square(u1)+np.square(u2))/2

    return F1, F2, F3, F4

def fluxVariablesSet(F1, F2, F3, F4, rho, gamma):

    G1 = rho*F3/F1
    G2 = F3
    G3 = rho*np.square(F3/F1) + F2 - np.square(F1)/rho
    G4 = gamma/(gamma-1)*(F2 - np.square(F1)/rho)*F3/F1 + rho/2*F3/F1*(np.square(F1/rho)+np.square(F3/rho))

    return G1, G2, G3, G4

def init():

    def parameterInput():

        gamma = 1.4
        theta = 5.352*np.pi/180
        Courant = 0.5
        Cy = 0.5    # factor of artificial viscosity
        # tStepNumb = 0#1600 #time step number and residual collection
        E = 10 #m
        # R = 1.295 #g/L
        MaInit = 2

        return gamma, Courant, theta, E, MaInit, Cy

    def originalVariablesInit():

        # Grid parameter
        Lx = 65
        y = np.linspace(0, 40, 41)  #m
        eta = np.linspace(0, 1, len(y))
        # Properties
        rho = np.zeros((1,len(y)),dtype=float)
        u1 = np.zeros((1,len(y)),dtype=float)
        u2 = np.zeros((1,len(y)),dtype=float)
        # u3 = np.zeros((len(x),len(y)),dtype=float)
        T = np.zeros((1,len(y)),dtype=float)
        p = np.zeros((1,len(y)),dtype=float)

        return Lx, y, eta, rho, u1, u2, T, p

    def flowFieldInit(MaInit, gamma, u1, u2, p, rho, T):

        p[0,:] = 1.01 * 10**5 # N/m^2
        rho[0,:] = 1.23       # kg/m^3
        T[0,:] = 286.1        # K
        R = p[0,0]/rho[0,0]/T[0,0]
        u1[0,:] = MaInit * np.sqrt(gamma*T[0,:]*R)
        u2[0,:] = 0

        return p, rho, T, u1, u2, R

    gamma, Courant, theta, E, MaInit, Cy = parameterInput()
    Lx, y, eta, rho, u1, u2, T, p = originalVariablesInit()
    p, rho, T, u1, u2, R = flowFieldInit(MaInit, gamma, u1, u2, p, rho, T)

    F1, F2, F3, F4 = origin2Conserve(rho, u1, u2, p, gamma)

    return gamma, R, theta, Courant, Cy, Lx, E, y, eta,rho, u1, u2, T, p, F1, F2, F3, F4

def tscheme(theta, h, detadx, y, F1, F2, F3, F4, Courant, gamma, R, Cy):
    # MacCormack scheme

    def caldksi(y, u1, u2, Ma):
        theta = np.arctan(u2/u1)
        niu = np.arcsin(1/Ma)
        dksi = Courant*(y[1]-y[0])/max(max(np.abs(np.tan(theta+niu))),max(np.abs(np.tan(theta-niu))))
        return dksi

    def artiVisc(p, U, Cy): #artificial viscosity
        S = np.zeros_like(U)
        for j in range(1,len(S)-1):
            S[j] = Cy * np.abs(p[j+1] - 2*p[j] + p[j-1]) / (p[j+1] + 2*p[j] + p[j-1]) * (U[j+1] - 2*U[j] + U[j-1])
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
                predF1[j] = -(detadx[j]*(F1[j] - F1[j-1]) + 1/h*(G1[j] - G1[j-1]))/deta
                predF2[j] = -(detadx[j]*(F2[j] - F2[j-1]) + 1/h*(G2[j] - G2[j-1]))/deta
                predF3[j] = -(detadx[j]*(F3[j] - F3[j-1]) + 1/h*(G3[j] - G3[j-1]))/deta
                predF4[j] = -(detadx[j]*(F4[j] - F4[j-1]) + 1/h*(G4[j] - G4[j-1]))/deta

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

    rho, u1, u2, p, T, Ma = conserve2Origin(F1, F2, F3, F4, gamma, R)
    dksi = caldksi(y, u1, u2, Ma)
    deta = (y[1]-y[0])/y[len(y)-1]

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

    newF1[0] = 2*newF1[1] - newF1[2]
    newF2[0] = 2*newF2[1] - newF2[2]
    newF3[0] = 2*newF3[1] - newF3[2]
    newF4[0] = 2*newF4[1] - newF4[2]

    newF1[0], newF2[0], newF3[0], newF4[0] = BCSet(newF1[0], newF2[0], newF3[0], newF4[0], gamma, R, theta)

    # newF1, newF2, newF3, newF4 = BCSet(newF1, newF2, newF3, newF4, rho, gamma)

    return newF1, newF2, newF3, newF4, dksi

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

    # def drawData():
    #     plt.figure()
    #     # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
    #     plt.plot(x, Ma, '-o', linewidth=1.0, color='black', markersize=1)
    #     # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
    #     plt.show()
    #     return

    printData()

    # drawData()
    return 0

def main():

    gamma, R, theta0, Courant, Cy, Lx, E, y, eta, rho, u1, u2, T, p, F1, F2, F3, F4 = init()
    totalx = 0
    iterNumb = 0
    while(totalx < Lx):
        theta, h, detadx = geometry(totalx, y, eta, E, theta0)  #calculate current geometry parameter
        if (totalx >= 10.0):
            newF1, newF2, newF3, newF4, dksi = tscheme(theta, h, detadx, y, F1[len(F1)-1,:], F2[len(F2)-1,:], F3[len(F3)-1,:],F4[len(F4)-1,:], Courant, gamma, R, Cy)
        else:
            newF1, newF2, newF3, newF4, dksi = tscheme(theta, h, detadx, y, F1[len(F1)-1,:], F2[len(F2)-1,:], F3[len(F3)-1,:],F4[len(F4)-1,:], Courant, gamma, R, Cy)

        F1 = np.append(F1, newF1[np.newaxis,:], axis=0)
        F2 = np.append(F2, newF2[np.newaxis,:], axis=0)
        F3 = np.append(F3, newF3[np.newaxis,:], axis=0)
        F4 = np.append(F4, newF4[np.newaxis,:], axis=0)

        totalx += dksi
        iterNumb += 1

        if (iterNumb >= 1000): #defensive design
            break

    # postProgress(F1, F2, F3, F4, totalx)

    return 0

if __name__ == "__main__":
    main()