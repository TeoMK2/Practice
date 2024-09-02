import numpy as np

# Sub-supersonic isentropic flow

# Equation sets to solve:
# Continuity:   dρ/dt = -u*dρ/dx - ρ*du/dx - ρu*d(lnA)/dx
# Momentum:     du/dt = -u*du/dx - 1/γ*(dT/dx + T/ρ*dp/dx)
# Energy:       dT/dt = -u*dT/dx - (γ-1)*T*(du/dx + u*d(lnA)/dx)

def BCSet(rho, u1, T):
    u1[0] = 2 * u1[1] - u1[2]
    rho[0] = 1
    T[0] = 1

    p = 0.93
    u1[len(u1) - 1] = 2 * u1[len(u1) - 2] - u1[len(u1) - 3]
    rho[len(rho) - 1] = 2 * rho[len(rho) - 2] - rho[len(rho) - 3]
    T[len(T) - 1] = p / rho[len(rho) - 1]
    return rho, u1, T

def init():

    def interimInit():
        gamma = 1.4
        Courant = 0.5
        timeStepNumber = 5000
        residual = np.zeros(timeStepNumber, dtype=float) #time step number and residual collection
        return gamma, Courant, residual

    def gridInit():
        x = np.linspace(0,3,31)
        A = np.zeros_like(x)
        for i in range(0,len(x)//2):
            A[i] = 1 + 2.2*np.square(x[i]-1.5)
        for i in range(len(x)//2,len(x)):
            A[i] = 1 + 0.2223*np.square(x[i]-1.5)
        rho = np.zeros_like(x)
        u1 = np.zeros_like(x)
        # u2 = np.zeros_like(x)
        # u3 = np.zeros_like(x)
        T = np.zeros_like(x)
        return x, A, rho, u1, T

    def flowFieldInit(x, rho, u1, T):
        for i in range(len(x)):
            rho[i] = 1 - 0.023*x[i]
            T[i] = 1 - 0.00933*x[i]
            u1[i] = 0.05 + 0.11*x[i]
        return rho, u1, T

    gamma, Courant, residual = interimInit()
    x, A, rho, u1, T = gridInit()
    rho, u1, T = flowFieldInit(x, rho, u1, T)
    rho, u1, T = BCSet(rho, u1, T)

    #--------test for inital condition-----------
    # print(np.round(A, 3))
    # print(np.round(rho, 3))
    # print(np.round(u1, 3))
    # print(np.round(T, 3))

    return x, A, rho, u1, T, gamma, Courant, residual

def tscheme(x, A, rho, u1, T, Courant, gamma, totalt):
    # MacCormack scheme
    dt = np.min(Courant * (x[1] - x[0]) / (u1 + np.sqrt(T)))
    dtdx = dt /(x[1] - x[0])
    totalt += dt

    def preStep(rho,u,A,T,gamma,dtdx):

        def continuityCal(rho,u,A,dtdx):
            newRho = np.zeros_like(rho)
            for i in range(1,len(newRho)-1):
                newRho[i] = rho[i] + dtdx * (-u[i]*(rho[i+1] - rho[i]) - rho[i]*(u[i+1] - u[i]) - rho[i]*u[i]*(np.log(A[i+1])-np.log(A[i])))
            return newRho
        def momentumCal(gamma,u,T,rho,dtdx):
            newU = np.zeros_like(u)
            for i in range(1,len(newU)-1):
                newU[i] = u[i] + dtdx * (-u[i]*(u[i+1] - u[i]) - 1/gamma*(T[i+1]-T[i] + T[i]/rho[i]*(rho[i+1] - rho[i])))
            return newU
        def energyCal(gamma,T,u,A,dtdx):
            newT = np.zeros_like(T)
            for i in range(1,len(newT)-1):
                newT[i] = T[i] + dtdx * (-u[i]*(T[i+1] - T[i]) - (gamma-1)*T[i]*(u[i+1]-u[i] + u[i]*(np.log(A[i+1]) - np.log(A[i]))))
            return newT

        preCtn = continuityCal(rho,u,A,dtdx)
        preMmtU1 = momentumCal(gamma,u,T,rho,dtdx)
        # preMmtq2 = momentumCal(gamma,u,T,rho,dtdx)
        # preMmtq3 = momentumCal(gamma,u,T,rho,dtdx)
        preEng = energyCal(gamma,T,u,A,dtdx)


        preCtn, preMmtU1, preEng = BCSet(preCtn, preMmtU1, preEng)

        # test for pre-step
        # print((preCtn[15] - rho[15])/dt)
        # print((preMmtU1[15] - u[15])/dt)
        # print((preEng[15] - T[15])/dt)

        return preCtn, preMmtU1, preEng

    def correctionStep(rho,u,A,T,gamma,dtdx):

        def continuityCal(rho,u,A,dtdx):
            newRho = np.zeros_like(rho)
            for i in range(1,len(newRho)-1):
                newRho[i] = rho[i] + dtdx * (-u[i]*(rho[i] - rho[i-1]) - rho[i]*(u[i] - u[i-1]) - rho[i]*u[i]*(np.log(A[i])-np.log(A[i-1])))
            return newRho
        def momentumCal(gamma,u,T,rho,dtdx):
            newU = np.zeros_like(u)
            for i in range(1,len(newU)-1):
                newU[i] = u[i] + dtdx * (-u[i]*(u[i] - u[i-1]) - 1/gamma*(T[i]-T[i-1] + T[i]/rho[i]*(rho[i] - rho[i-1])))
            return newU
        def energyCal(gamma,T,u,A,dtdx):
            newT = np.zeros_like(T)
            for i in range(1,len(newT)-1):
                newT[i] = T[i] + dtdx * (-u[i]*(T[i] - T[i-1]) - (gamma-1)*T[i]*(u[i]-u[i-1] + u[i]*(np.log(A[i]) - np.log(A[i-1]))))
            return newT

        corrCtn = continuityCal(rho,u,A,dtdx)
        corrMmtU1 = momentumCal(gamma,u,T,rho,dtdx)
        # corrMmtq2 = momentumCal(gamma,u,T,rho,dt,dx)
        # corrMmtq3 = momentumCal(gamma,u,T,rho,dt,dx)
        corrEng = energyCal(gamma,T,u,A,dtdx)

        # test for correct-step
        # print((corrCtn[15] - rho[15])/dt)
        # print((corrMmtU1[15] - u[15])/dt)
        # print((corrEng[15] - T[15])/dt)

        return corrCtn, corrMmtU1, corrEng

    preCtn, preMmtU1, preEng = preStep(rho,u1,A,T,gamma,dtdx)
    corrCtn, corrMmtU1, corrEng = correctionStep(preCtn,preMmtU1,A,preEng,gamma,dtdx)

    newRho = (corrCtn + rho)*0.5
    newU1 = (corrMmtU1 + u1)*0.5
    newT = (corrEng + T)*0.5

    residual = np.max([np.max(newRho - rho),np.max(newU1 - u1),np.max(newT - T)])/dt

    rho = newRho
    u1 = newU1
    T = newT

    #BC still error
    rho, u1, T = BCSet(rho, u1, T)

    return rho, u1, T, residual, totalt

def postProgress(rho, u1, T, residual):

    def printData():
        p = rho*T
        Ma = u1/np.sqrt(T)

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

    x, A, rho, u1, T, gamma, Courant, residual = init()
    totalt = 0
    for t in range(len(residual)):
        rho, u1, T, R, totalt = tscheme(x, A, rho, u1, T, Courant, gamma, totalt)
        residual[t] = R

    postProgress(rho, u1, T, residual)

    return 0

if __name__ == "__main__":
    main()