import numpy as np
import matplotlib.pyplot as plt

# Sub-supersonic isentropic flow

# Non-dimensionalized equation sets to solve:
# Conservative form:    d(U)/dt                  = -d(F)/dx                     + J
#
# Continuity:           d(ρA)/dt                 = -d(ρAu)/dx
#
# Momentum:             d(ρAu)/dt                = -d(ρAu^2+pA/γ)/dx
#
# Energy:               d(ρA(e/(γ-1)+γ/2*(u)^2))/dt = -d(ρAu(e/γ-1+γu^2/2)+pAu)/dx + p/γ*dA/dx
#

def BCSet(U1, U2, U3, A,gamma):
    # inlet
    U1[0] = A[0]   #fix rho
    U2[0] = 2*U2[1] - U2[2] #interplot
    U3[0] = U1[0]*(1/(gamma-1)+gamma/2*np.square(U2[0]/U1[0]))  #T=1
    # outlet
    U1[len(U1) - 1] = 2 * U1[len(U1) - 2] - U1[len(U1) - 3]
    U2[len(U2) - 1] = 2 * U2[len(U2) - 2] - U2[len(U2) - 3]
    # U3[len(U3) - 1] = 2 * U3[len(U3) - 2] - U3[len(U3) - 3]
    U3[len(U3) - 1] = 0.6784*A[len(U3) - 1]/(gamma-1) + gamma/2*np.square(U2[len(U3) - 1])/U1[len(U3) - 1]     # p=0.6784 is constant
    return U1, U2, U3

def fluxVariablesSet(U1, U2, U3, gamma):

    F1 = U2
    F2 = np.square(U2)/U1 + (gamma-1)/gamma*(U3 - gamma/2*np.square(U2)/U1)
    F3 = gamma*U2*U3/U1 - gamma*(gamma-1)/2*(U2**3)/np.square(U1)

    #tmp = gamma*round(U2[14],3)*round(U3[14],3)/round(U1[14],3) - gamma*(gamma-1)/2*(round(U2[14],3)**3)/np.square(round(U1[14],3))

    return F1, F2, F3

def originalVariables(U1, U2, U3, A, gamma):

    rho = U1/A
    u1 = U2/U1
    T = (gamma-1)*(U3/U1 - gamma/2*np.square(u1))

    return rho, u1, T

def init():

    def parameterInput():
        gamma = 1.4
        Courant = 0.5
        tStepNumb = 1600 #time step number and residual collection

        return gamma, Courant, tStepNumb

    def originalVariablesInit():
        x = np.linspace(0,3,61)
        A = 1 + 2.2 * np.square(x - 1.5)
        rho = np.zeros_like(x)
        # u1 = np.zeros_like(x) #the inital u1 is given by mass flux which is constant
        # u2 = np.zeros_like(x)
        # u3 = np.zeros_like(x)
        T = np.zeros_like(x)

        return x, A, rho, T

    def conservVariablesInit(rho, A, u1, T, gamma):

        U1 = rho*A
        U2 = rho*A*u1
        U3 = rho*A*(T/(gamma-1)+gamma/2*np.square(u1))
        return U1, U2, U3

    def flowFieldInit(x, rho, T):

        for i in range(len(x)):
            if 0<=x[i]<=0.5:
                rho[i] = 1.0
                T[i] = 1.0
            if 0.5<=x[i]<=1.5:
                rho[i] = 1.0 - 0.366*(x[i]-0.5)
                T[i] = 1.0 - 0.167*(x[i]-0.5)
            # if 1.5<=x[i]<=3.0:
            #     rho[i] = 0.634 - 0.3879*(x[i]-1.5)
            #     T[i] = 0.833 - 0.3507*(x[i]-1.5)
            if 1.5<=x[i]<=2.1:
                rho[i] = 0.634 - 0.702*(x[i]-1.5)
                T[i] = 0.833 - 0.4908*(x[i]-1.5)
            if 2.1<=x[i]<=3.0:
                rho[i] = 0.5892 - 0.10228*(x[i]-2.1)
                T[i] = 0.93968 - 0.0622*(x[i]-2.1)

        u1 = 0.59/(rho*A)

        return rho, u1, T

    gamma, Courant, tStepNumb = parameterInput()
    x, A, rho, T = originalVariablesInit()
    rho, u1, T = flowFieldInit(x, rho, T)

    U1, U2, U3 = conservVariablesInit(rho, A, u1, T, gamma)

    return x, A, U1, U2, U3, gamma, Courant, tStepNumb

def tscheme(x, A, U1, U2 ,U3, Courant, gamma):
    # MacCormack scheme
    rho, u1, T = originalVariables(U1, U2, U3, A, gamma)
    dt = np.min(Courant * (x[1] - x[0]) / (u1 + np.sqrt(T)))#0.0267#
    dx = 1/(x[1] - x[0])
    # dt = Courant * (x[1] - x[0]) / (u1 + np.sqrt(T))
    # dtdx = dt / (x[1] - x[0])

    def artiVisc(p, U):
        Cx = 0.2
        S = np.zeros_like(U)
        for i in range(1,len(S)-1):
            S[i] = Cx * np.abs(p[i+1] - 2*p[i] + p[i-1]) / (p[i+1] + 2*p[i] + p[i-1]) * (U[i+1] - 2*U[i] + U[i-1])

        return S

    def preStep(U1, U2, U3, dx, rho, T):

        F1, F2, F3 = fluxVariablesSet(U1, U2, U3, gamma)

        predU1 = np.zeros_like(U1)
        predU2 = np.zeros_like(U2)
        predU3 = np.zeros_like(U3)
        tmpJ2 = np.zeros_like(U2)
        tmpJ22 = np.zeros_like(U2)

        for i in range(1, len(predU1) - 1):
            # original form
            tmpJ2[i] = 1/gamma * rho[i] * T[i] * (A[i+1] - A[i])
            # conservative form
            tmpJ22[i] = (gamma-1)/gamma * (U3[i] - gamma/2 * np.square(U2[i])/U1[i]) * (np.log(A[i+1]) - np.log(A[i]))

            predU1[i] = dx * (-(F1[i + 1] - F1[i])           )# + artiVisc(rho*T, U1, i)
            predU2[i] = dx * (-(F2[i + 1] - F2[i]) + tmpJ2[i])# + artiVisc(rho*T, U2, i)
            predU3[i] = dx * (-(F3[i + 1] - F3[i])           )# + artiVisc(rho*T, U3, i)

        # test for pre-step
        # print((preCtn[15] - rho[15])/dt)
        # print((preMmtU1[15] - u[15])/dt)
        # print((preEng[15] - T[15])/dt)

        return predU1, predU2, predU3

    def correctionStep(preU1, preU2, preU3, dx, rho, T):

        F1, F2, F3 = fluxVariablesSet(preU1, preU2, preU3, gamma)

        cordU1 = np.zeros_like(preU1)
        cordU2 = np.zeros_like(preU2)
        cordU3 = np.zeros_like(preU3)
        tmpJ2  = np.zeros_like(preU2)
        tmpJ22 = np.zeros_like(preU2)

        for i in range(1, len(cordU1) - 1):
            # original form
            tmpJ2[i] = 1/gamma * rho[i] * T[i] * (A[i] - A[i-1])
            # conservative form
            tmpJ22[i] = (gamma-1)/gamma*(U3[i] - gamma/2 * np.square(U2[i])/U1[i]) * (np.log(A[i]) - np.log(A[i-1]))

            cordU1[i] = dx * (-(F1[i] - F1[i - 1])           )# + artiVisc(rho*T, U1, i)
            cordU2[i] = dx * (-(F2[i] - F2[i - 1]) + tmpJ2[i])# + artiVisc(rho*T, U2, i)
            cordU3[i] = dx * (-(F3[i] - F3[i - 1])           )# + artiVisc(rho*T, U3, i)

        # test for pre-step
        # print((preCtn[15] - rho[15])/dt)
        # print((preMmtU1[15] - u[15])/dt)
        # print((preEng[15] - T[15])/dt)

        return cordU1, cordU2, cordU3

    #pre-step
    predU1, predU2, predU3 = preStep(U1, U2, U3, dx, rho, T)
    preS1, preS2, preS3 = artiVisc(rho * T, U1), artiVisc(rho * T, U2), artiVisc(rho * T, U3)
    preU1, preU2, preU3 = BCSet(U1 + predU1*dt + preS1, U2 + predU2*dt + preS2, U3 + predU3*dt + preS3, A, gamma)

    preRho, preu1, preT = originalVariables(preU1, preU2, preU3, A, gamma)

    #correct-step
    cordU1, cordU2, cordU3 = correctionStep(preU1, preU2, preU3, dx, preRho, preT)
    corS1, corS2, corS3 = artiVisc(preRho * preT, preU1), artiVisc(preRho * preT, preU2), artiVisc(preRho * preT, preU3)
    newU1, newU2, newU3 = BCSet(U1 + (predU1 + cordU1)*0.5*dt + corS1, U2 + (predU2 + cordU2)*0.5*dt + corS2, U3 + (predU3 + cordU3)*0.5*dt + corS3, A, gamma)

    return newU1, newU2, newU3, dt

def postProgress(U1, U2, U3, x, A, gamma, timeStepNumb, totalt, residual):

    rho, u1, T = originalVariables(U1, U2, U3, A, gamma)
    p = rho * T
    Ma = u1 / np.sqrt(T)

    m = rho * u1 * A

    def printData():
        print("------------solve complete.------------")
        print("iteration or temporal advancement times:", timeStepNumb)
        print("total physical time:", totalt)

        print("---------------------------------------")
        print("residual:", residual)
        print("ρ:", rho)
        print("u1:", u1)
        print("T:", T)
        print("p", p)
        print("Ma", Ma)
        return

    def drawData():
        plt.figure()
        # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
        plt.plot(x, Ma, '-o', linewidth=1.0, color='black', markersize=1)
        # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
        plt.show()
        return

    printData()

    drawData()

    # print("residual:", residual)
    return 0

def main():

    x, A, U1, U2, U3, gamma, Courant, tStepNumb = init()
    totalt = 0
    residual = np.array([],dtype=float)
    for t in range(tStepNumb):
        newU1, newU2, newU3, dt = tscheme(x, A, U1, U2 ,U3, Courant, gamma)

        R = np.max([np.max(newU1 - U1), np.max(newU2 - U2), np.max(newU3 - U3)]) / dt
        residual = np.append(R,residual)

        U1, U2, U3 = newU1, newU2, newU3
        totalt += dt
        if t == tStepNumb-1:
            postProgress(U1, U2, U3, x, A, gamma, tStepNumb, totalt, residual)

    return 0

if __name__ == "__main__":
    main()