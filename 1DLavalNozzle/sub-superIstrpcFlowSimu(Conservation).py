import numpy as np

# Sub-supersonic isentropic flow

# Non-dimensionalized equation sets to solve:
# Conservative form:    d(U)/dt                  = -d(F)/dx                     + J
#
# Continuity:           d(ρA)/dt                 = -d(ρAu)/dx
#
# Momentum:             d(ρAu)/dt                = -d(ρAu^2+pA/γ)/dx
#
# Energy:               d(ρ(e/γ-1)+γ/2*(u)^2)/dt = -d(ρuA(e/γ-1+γu^2/2)+pAu)/dx + p/γ*dA/dx
#

def BCSet(U1, U2, U3, A,gamma):
    # inlet
    U1[0] = A[0]   #fix rho
    U2[0] = 2*U2[1] - U2[2] #interplot
    U3[0] = U1[0]*(1/(gamma-1)+gamma/2*np.square(U2[0]/U1[0]))  #T=1
    # outlet
    U1[len(U1) - 1] = 2 * U1[len(U1) - 2] - U1[len(U1) - 3]
    U2[len(U2) - 1] = 2 * U2[len(U2) - 2] - U2[len(U2) - 3]
    U3[len(U3) - 1] = 2 * U3[len(U3) - 2] - U3[len(U3) - 3]
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
        tStepNumb = 1400 #time step number and residual collection

        return gamma, Courant, tStepNumb

    def originalVariablesInit():
        x = np.linspace(0,3,31)
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
        for i in range(0,len(x)//6):
            rho[i] = 1
            T[i] = 1
        for i in range(len(x)//6,len(x)//2):
            rho[i] = 1 - 0.366*(x[i]-0.5)
            T[i] = 1 - 0.167*(x[i]-0.5)
        for i in range(len(x)//2,len(x)):
            rho[i] = 0.634 - 0.3879*(x[i]-1.5)
            T[i] = 0.833 - 0.3507*(x[i]-1.5)
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
    dtdx = dt /(x[1] - x[0])
    # dt = Courant * (x[1] - x[0]) / (u1 + np.sqrt(T))
    # dtdx = dt / (x[1] - x[0])

    def preStep(U1, U2, U3, dtdx, rho, T):

        F1, F2, F3 = fluxVariablesSet(U1, U2, U3, gamma)

        preU1 = np.zeros_like(U1)
        preU2 = np.zeros_like(U2)
        preU3 = np.zeros_like(U3)
        tmpJ2 = np.zeros_like(U2)
        tmpJ22 = np.zeros_like(U2)

        for i in range(1, len(preU1) - 1):
            preU1[i] = U1[i] + dtdx * -(F1[i + 1] - F1[i])

            # original form
            tmpJ2[i] = 1/gamma * rho[i] * T[i] * (A[i+1] - A[i])
            # conservative form
            tmpJ22[i] = (gamma-1)/gamma * (U3[i] - gamma/2 * np.square(U2[i])/U1[i]) * (np.log(A[i+1]) - np.log(A[i]))

            preU2[i] = U2[i] + dtdx * (-(F2[i + 1] - F2[i]) + tmpJ2[i])
            preU3[i] = U3[i] + dtdx * (-(F3[i + 1] - F3[i]))

        preU1, preU2, preU3 = BCSet(preU1, preU2, preU3, A, gamma)

        # test for pre-step
        # print((preCtn[15] - rho[15])/dt)
        # print((preMmtU1[15] - u[15])/dt)
        # print((preEng[15] - T[15])/dt)

        return preU1, preU2, preU3

    def correctionStep(U1, U2, U3, dtdx, rho, T):

        F1, F2, F3 = fluxVariablesSet(U1, U2, U3, gamma)

        corrU1 = np.zeros_like(U1)
        corrU2 = np.zeros_like(U2)
        corrU3 = np.zeros_like(U3)
        tmpJ2  = np.zeros_like(U2)
        tmpJ22 = np.zeros_like(U2)

        for i in range(1, len(corrU1) - 1):
            corrU1[i] = U1[i] + dtdx * (-(F1[i] - F1[i - 1]))

            # original form
            tmpJ2[i] = 1/gamma * rho[i] * T[i] * (A[i] - A[i-1])
            # conservative form
            tmpJ22[i] = (gamma-1)/gamma*(U3[i] - gamma/2 * np.square(U2[i])/U1[i]) * (np.log(A[i]) - np.log(A[i-1]))

            corrU2[i] = U2[i] + dtdx * (-(F2[i] - F2[i - 1]) + tmpJ2[i])
            corrU3[i] = U3[i] + dtdx * (-(F3[i] - F3[i - 1]))

        corrU1, corrU2, corrU3 = BCSet(corrU1, corrU2, corrU3, A, gamma)

        # test for pre-step
        # print((preCtn[15] - rho[15])/dt)
        # print((preMmtU1[15] - u[15])/dt)
        # print((preEng[15] - T[15])/dt)

        return corrU1, corrU2, corrU3

    preU1, preU2, preU3 = preStep(U1, U2, U3, dtdx, rho, T)

    preRho, preu1, preT = originalVariables(preU1, preU2, preU3, A, gamma)
    corrU1, corrU2, corrU3 = correctionStep(preU1, preU2, preU3, dtdx, preRho, preT)

    U1 = (corrU1 + U1)*0.5
    U2 = (corrU2 + U2)*0.5
    U3 = (corrU3 + U3)*0.5

    #BC still error
    U1, U2, U3 = BCSet(U1, U2, U3, A,gamma)

    return U1, U2, U3, dt

def postProgress(U1, U2, U3, A, gamma, timeStepNumb, totalt):

    def printData():
        rho, u1, T = originalVariables(U1, U2, U3, A, gamma)
        p = rho*T
        Ma = u1/np.sqrt(T)

        print("------------solve complete.------------")
        print("iteration or temporal advancement times:", timeStepNumb)
        print("total physical time:", totalt)
        print("---------------------------------------")
        print("ρ:", rho)
        print("u1:", u1)
        print("T:", T)
        print("p", p)
        print("Ma", Ma)
        return

    printData()

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
            postProgress(U1, U2, U3, A, gamma, tStepNumb, totalt)

    return 0

if __name__ == "__main__":
    main()