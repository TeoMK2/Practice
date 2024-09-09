import numpy as np
import matplotlib.pyplot as plt
import sys

# 1D incompressive Couette flow
#
# Non-dimensionalized equation sets to solve:
#
# Continuity:
#
# Momentum:             d(u)/dt                = 1/Re*d2(u)/d2(y)
#

def BCSet(u1):
    # inlet
    u1[0] = 0
    # outlet
    u1[-1] = 1
    return u1

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
        E = 0.5
        tStepNumb = 1600 #time step number and residual collection
        Ren = 10
        return E, Ren, tStepNumb

    def originalVariablesInit():
        y = np.linspace(0,3,21)
        u1 = np.zeros_like(y) #the inital u1 is given by mass flux which is constant
        return y, u1

    def flowFieldInit(u1):
        u1[:-1] = 0
        u1[-1] = 1
        return u1

    E, Ren, tStepNumb = parameterInput()
    y, u1 = originalVariablesInit()
    u1 = flowFieldInit(u1)

    return y, u1, E, Ren, tStepNumb

def tscheme(y, u1, E, Ren):
    # Crank-Nicolson scheme

    def combMatrix(coefF, coefC, coefB, size):
        upperDiag = np.ones(size-1)*coefF
        mainDiag = np.ones(size)*coefC
        lowerDiag = np.ones(size-1)*coefB
        Matrix = np.diag(mainDiag) + np.diag(lowerDiag, -1) + np.diag(upperDiag, 1)
        return Matrix

    def solveTridiaMatrix(TDM, RHS):

        if len(TDM[0,:]) != len(RHS) or len(TDM[:,0]) != len(RHS):
            print("error: Matrix's size is not match with solution vector's size")
            sys.exit()

        for i in range(1,len(TDM[0,:])):
            if TDM[i,i-1] != 0:
                TDM[i,:] -= TDM[i-1,:]*(TDM[i,i-1]/TDM[i-1,i-1])
                RHS[i] -= RHS[i-1]*(TDM[i,i-1]/TDM[i-1,i-1])

        for i in range(len(TDM[0,:])-2,-1,-1):
            if TDM[i,i+1] != 0:
                TDM[i,:] -= TDM[i+1,:]*(TDM[i,i+1]/TDM[i+1,i+1])
                RHS[i] -= RHS[i+1]*(TDM[i,i+1]/TDM[i+1,i+1])

        for i in range(len(RHS)):
            RHS[i] /= TDM[i,i]

        return RHS

    # calculate sum of right hand side terms
    dy = y[1] - y[0]
    dt = E*Ren*dy**2
    coef = E/2
    RHS = np.zeros(len(y)-2)
    for i in range(len(RHS)):
        # where i in RHS correspond to i+1 in u1
        RHS[i] = (1 - 2*coef)*u1[i+1] + coef*(u1[i+2] + u1[i])

    # combine a tridiagonal matrix
    triDiagMatrix = combMatrix(-coef, 1 + coef, -coef, len(y)-2)

    # solve tridiagonal matrix with Thomas' method
    newu1 = np.pad(solveTridiaMatrix(triDiagMatrix, RHS), pad_width=1, mode='constant', constant_values=0)    # append head and tail
    BCSet(newu1)

    return newu1, dt

def postProgress(u1, y, tStepNumb, totalt, residual):

    # rho, u1, T = originalVariables(U1, U2, U3, A, gamma)
    # p = rho * T
    # Ma = u1 / np.sqrt(T)
    #
    # m = rho * u1 * A

    # def printData():
    #     print("------------solve complete.------------")
    #     print("iteration or temporal advancement times:", timeStepNumb)
    #     print("total physical time:", totalt)
    #
    #     print("---------------------------------------")
    #     print("residual:", residual)
    #     print("ρ:", rho)
    #     print("u1:", u1)
    #     print("T:", T)
    #     print("p", p)
    #     print("Ma", Ma)
    #     return
    #
    # def drawData():
    #     plt.figure()
    #     # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
    #     plt.plot(x, Ma, '-o', linewidth=1.0, color='black', markersize=1)
    #     # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
    #     plt.show()
    #     return

    # printData()

    # drawData()

    # print("residual:", residual)
    return 0

def main():

    y, u1, E, Ren, tStepNumb = init()
    totalt = 0
    residual = np.array([],dtype=float)
    for t in range(tStepNumb):
        newu1, dt = tscheme(y, u1, E, Ren)

        # R = np.max([np.max(newU1 - U1), np.max(newU2 - U2), np.max(newU3 - U3)]) / dt
        # residual = np.append(R,residual)

        u1 = newu1
        totalt += dt
        if t == tStepNumb-1:
            postProgress(u1, y, tStepNumb, totalt, residual)

    return 0

if __name__ == "__main__":
    main()