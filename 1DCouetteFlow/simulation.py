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

def init():

    def parameterInput():
        E = 4000
        tStepNumb = 10000 #time step number and residual collection
        Ren = 5000
        return E, Ren, tStepNumb

    def originalVariablesInit():
        y = np.linspace(0,1,21)
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

        for i in range(1,len(TDM[0,:])):
            ratio = (TDM[i,i-1]/TDM[i-1,i-1])
            TDM[i,:] -= TDM[i-1,:]*ratio
            RHS[i] -= RHS[i-1]*ratio

        for i in range(len(TDM[0,:])-2,-1,-1):
            ratio = (TDM[i,i+1]/TDM[i+1,i+1])
            TDM[i,:] -= TDM[i+1,:]*(TDM[i,i+1]/TDM[i+1,i+1])
            RHS[i] -= RHS[i+1]*ratio

        for i in range(len(RHS)):
            RHS[i] /= TDM[i,i]

        return RHS

    # calculate sum of right hand side terms
    dy = y[1] - y[0]
    dt = E*Ren*dy**2
    coef = E/2
    RHS = np.zeros(len(y))
    for i in range(1,len(RHS)-1):
        # where i in RHS correspond to i+1 in u1
        if i == 1:
            RHS[i] = (1 - coef*2)*u1[i] + coef*(u1[i-1] + u1[i+1]) + u1[i-1]*coef  #K2 - A
        elif i == len(RHS)-2:
            RHS[i] = (1 - coef*2)*u1[i] + coef*(u1[i-1] + u1[i+1]) + u1[i+1]*coef  #KN - A
        else:
            RHS[i] = (1 - coef*2)*u1[i] + coef*(u1[i-1] + u1[i+1])

    # combine a tridiagonal matrix
    triDiagMatrix = combMatrix(-coef, 1 + coef*2, -coef, len(y)-2)

    # solve tridiagonal matrix with Thomas' method
    #soluVector = np.pad(), pad_width=1, mode='constant', constant_values=0)    # append head and tail
    RHS[1:-1] = solveTridiaMatrix(triDiagMatrix, RHS[1:-1])
    newu1 = BCSet(RHS)

    return newu1, dt

def postProgress(u1, y, tStepNumb, totalt):


    def printData():
        print("------------solve complete.------------")
        print("iteration or temporal advancement times:", tStepNumb)
        print("total physical time:", totalt)

        print("---------------------------------------")
        # print("ρ:", rho)
        print("u1:", u1)
        print("y:", y)
        # print("p", p)
        # print("Ma", Ma)
        return

    def drawData():
        plt.figure()
        # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
        for i in range(len(u1)):
            plt.plot(u1[i], y, '-o', linewidth=1.0, color='black', markersize=1)
        # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
        plt.show()
        return

    # printData()

    drawData()

    # print("residual:", residual)
    return 0

def main():

    y, u1, E, Ren, tStepNumb = init()
    totalt = 0
    collector_u1 = np.zeros((1,len(y)),dtype=float)
    for t in range(tStepNumb+1):
        for n in ([0,40,200,800,10000]):
            if t == n:
                collector_u1 = np.append(collector_u1, u1[np.newaxis,:], axis=0)

        newu1, dt = tscheme(y, u1, E, Ren)

        u1 = newu1
        totalt += dt

    postProgress(collector_u1, y, tStepNumb, totalt)

    return 0

if __name__ == "__main__":
    main()