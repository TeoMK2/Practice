import numpy as np
import matplotlib.pyplot as plt
import sys

# 1D incompressive Couette flow
#
# SIMPLE(Semi-Implicit Method for Pressure Linked Equations) method
#
# Free-divergence:      d(u1)/dx + d(u2)/dy = 0
#
# Momentum:             d(ρu1)/dt + d(ρu1^2)/dx + d(ρu1u2)/dy = -d(p)/dx + μ(d^2(u1)/dx^2 + d^2(u1)/dy^2)
#                       d(ρu2)/dt + d(ρu1u2)/dx + d(ρu2^2)/dy = -d(p)/dy + μ(d^2(u2)/dx^2 + d^2(u2)/dy^2)

def checkDiv(u1, u2, dx, dy):
    div = np.zeros((len(u2[:,0]),len(u1[0,:])),dtype=float)
    for i in range(1,len(div[:,0])-1):
        for j in range(1,len(div[0,:])-1):
            div[i,j] = (u1[i+1,j] - u1[i,j])/dx + (u2[i,j+1] - u2[i,j])/dy
    maxDiv = np.max(div)
    return maxDiv

def velBCupdate(u1, u2, U, outu2):
    # upper BC
    u1[:,-1] = U
    u2[:,-1] = 0
    # lower BC
    u1[:,0] = 0
    u2[:,0] = 0
    # inlet
    u1[0,:] = u1[1,:]
    # outlet
    outu2[:] = u2[-1,:]
    u1[-1,:] = u1[-2,:]
    return u1, u2, outu2

def presBCSet(pp):
    # inlet
    pp[0,:] = 0
    # outlet
    pp[-1,:] = 0
    # upper
    pp[:,-1] = 0
    # lower
    pp[:,0] = 0
    return pp

def init():

    def parameterInput():
        Lx = 0.5    #ft
        Ly = 0.01   #ft
        Nx = 21
        Ny = 11
        x = np.linspace(0,Lx, Nx)
        y = np.linspace(0,Ly, Ny)
        U = 1 #ft/s

        rho = 0.002377 #slug/ft^3
        mu = U*Ly*rho/63.6

        dt = 0.001

        return x, y, dt, U, mu, rho

    def originalVariablesInit(x, y):
        p = np.zeros((len(x),len(y)),dtype=float)
        u1 = np.zeros((len(x)+1,len(y)),dtype=float)
        u2 = np.zeros((len(x),len(y)+1),dtype=float)
        inu2 = np.zeros(len(y)+1,dtype=float)   # Ghost Cell:inlet BC
        outu2 = np.zeros(len(y)+1,dtype=float)   #Ghost Cell:outlet BC
        return u1, u2, p, inu2, outu2

    def flowFieldInit(u1, u2, U, pss):
        u1[:,:] = 0
        u2[:,:] = 0
        ps[:,:] = 0
        pp = np.zeros_like(ps)
        u1[14,4] = 0.5 #ft/s

        return u1, u2, ps, pp

    x ,y , dt, U, mu, rho = parameterInput()
    u1, u2, ps, inu2, outu2 = originalVariablesInit(x, y)
    u1, u2, ps, pp = flowFieldInit(u1, u2, U, ps)
    u1, u2, outu2 = velBCupdate(u1, u2, U, outu2)
    pp = presBCSet(pp)

    return x, y, dt, u1, u2, inu2, outu2, ps, pp, rho, mu, U

def tscheme(dx, dy, dt, u1, u2, inu2, outu2, rho, ps, pp, mu, U):

    def semiStep(u1, u2, inu2, outu2, ps, rho, dx, dy, dt, mu):
        # calculate velocity with divergence
        newRhoU1, newRhoU2 = np.zeros_like(u1), np.zeros_like(u2)
        inu2[:] = 0
        outu2[:] = u2[-1,:]
        for i in range(1, len(u1[:,0])-1):
            for j in range(1, len(u1[0,:])-1):
                pI = i - 1
                iu2 = i - 1
                ju2 = j + 1
                presTerm = (ps[pI+1,j] - ps[pI,j])/dx
                visTerm = mu*((u1[i+1,j]-2*u1[i,j]+u1[i-1,j])/dx**2 + (u1[i,j+1]-2*u1[i,j]+u1[i,j-1])/dy**2)
                convTerm = -((rho*u1[i+1,j]**2 - rho*u1[i-1,j]**2)/2/dx +
                             (rho*u1[i,j+1]*(u2[iu2,ju2+1]+u2[iu2+1,ju2+1]) - rho*u1[i,j-1]*(u2[iu2,ju2-1]+u2[iu2+1,ju2-1]))/2/2/dy)
                newRhoU1[i,j] = rho*u1[i,j] + dt*(-presTerm + convTerm + visTerm)

        for j in range(1, len(u2[0,:])-1):
            for i in range(len(u2[:,0])):
                pJ = j - 1
                iu1 = i + 1
                ju1 = j - 1
                if i == 0:
                    presTerm = (ps[i,pJ+1] - ps[i,pJ+1])/dy
                    visTerm = mu*((u2[i+1,j]-2*u2[i,j]+inu2[j])/dx**2 + (u2[i,j+1]-2*u2[i,j]+u2[i,j-1])/dy**2)
                    convTerm = -((rho*u2[i+1,j]*(u1[iu1+1,ju1]+u1[iu1+1,ju1+1]) - rho*inu2[j]*(u1[iu1-1,ju1]+u1[iu1-1,ju1+1]))/2/2/dx+
                                 (rho*u2[i,j+1]**2 - rho*u2[i,j-1]**2)/2/dy)
                    newRhoU2[i,j] = rho*u2[i,j] + dt*(-presTerm + convTerm + visTerm)
                elif i == len(u2[:,0])-1:
                    presTerm = (ps[i,pJ+1] - ps[i,pJ+1])/dy
                    visTerm = mu*((outu2[j]-2*u2[i,j]+u2[i-1,j])/dx**2 + (u2[i,j+1]-2*u2[i,j]+u2[i,j-1])/dy**2)
                    convTerm = -((rho*outu2[j]*(u1[iu1,ju1]+u1[iu1,ju1+1]) - rho*u2[i-1,j]*(u1[iu1-1,ju1]+u1[iu1-1,ju1+1]))/2/2/dx+ #u1[iu1+1,ju1] = u1[iu1,ju1]
                                 (rho*u2[i,j+1]**2 - rho*u2[i,j-1]**2)/2/dy)
                    newRhoU2[i,j] = rho*u2[i,j] + dt*(-presTerm + convTerm + visTerm)
                else:
                    presTerm = (ps[i,pJ+1] - ps[i,pJ+1])/dy
                    visTerm = mu*((u2[i+1,j]-2*u2[i,j]+u2[i-1,j])/dx**2 + (u2[i,j+1]-2*u2[i,j]+u2[i,j-1])/dy**2)
                    convTerm = -((rho*u2[i+1,j]*(u1[iu1+1,ju1]+u1[iu1+1,ju1+1]) - rho*u2[i-1,j]*(u1[iu1-1,ju1]+u1[iu1-1,ju1+1]))/2/2/dx+
                                 (rho*u2[i,j+1]**2 - rho*u2[i,j-1]**2)/2/dy)
                    newRhoU2[i,j] = rho*u2[i,j] + dt*(-presTerm + convTerm + visTerm)

        return  newRhoU1, newRhoU2

    def pCorrect(rhoU1, rhoU2, pp, dx, dy, dt):
        #   a*ps[i,j] + b*(ps[i+1,j] + ps[i-1,j]) + c*(ps[i,j+1] + ps[i,j-1]) + d = 0
        a = 2*(1/dx**2 + 1/dy**2)*dt
        b = -dt/dx**2
        c = -dt/dy**2

        dcolle = 0

        newpp = np.zeros_like(pp)
        # solve Poisson equation using Relaxation Method
        for i in range(1,len(pp[:,0])-1):
            for j in range(1,len(pp[0,:])-1):
                # iu1 = i + 1
                # ju2 = j + 1
                d = (rhoU1[i+1,j] - rhoU1[i,j])/dx + (rhoU2[i,j+1] - rhoU2[i,j])/dy
                if i == 15 and j == 5:
                    dcolle = d
                newpp[i,j] = -(b*(pp[i+1,j] + pp[i-1,j]) + c*(pp[i,j+1] + pp[i,j-1]) + d)/a
        newpp = presBCSet(newpp)
        return newpp, dcolle

    semiRhoU1, semiRhoU2 = semiStep(u1, u2, inu2, outu2, ps, rho, dx, dy, dt, mu)
    semiU1, semiU2, outu2 = velBCupdate(semiRhoU1/rho, semiRhoU2/rho, U, outu2)
    semiRhoU1, semiRhoU2 = semiU1*rho, semiU2*rho

        # sub-iteration step (Relaxation Method)
    for j in range(200):
    # while(1): # need set with iteration exit condition
        newpp, dcolle = pCorrect(semiRhoU1, semiRhoU2, pp, dx, dy, dt)
        pp = newpp

    ps += 0.1*pp

    return semiRhoU1/rho, semiRhoU2/rho, inu2, outu2, ps, pp, dcolle

def postProgress(u1, u2, inu2, outu2, ps, x, y, u1colle):
    X = np.tile(x, (len(y),1)).T
    Y = np.tile(y, (len(x),1))

    # def printData():
    #     print("------------solve complete.------------")
    #     print("iteration or temporal advancement times:", tStepNumb)
    #     print("total physical time:", totalt)
    #
    #     print("---------------------------------------")
    #     print("u1:", u1)
    #     print("y:", y)
    #
    #     return
    #
    def drawData(X, Y, data, name):
        plt.figure()
        # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
        pic = plt.contourf(X, Y, data, alpha=1, cmap='jet', levels=np.linspace(0,1,101))
        # plt.plot(x, Ma, '-o', linewidth=1.0, color='black', markersize=1)
        plt.colorbar(pic)
        plt.xlim(0, X[-1,0])
        plt.ylim(0, Y[0,-1])
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        plt.title('Evolution of ' + name)
        # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
        plt.show()
        return

    def drawData2(data, y, name):
        plt.figure()
        # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
        # pic = plt.contourf(X, Y, data, alpha=1, cmap='jet', levels=np.linspace(0,1,101))
        for i in range(len(data[:,0])):
            plt.plot(data[i], y, '-o', linewidth=1.0, color='black', markersize=1)
        plt.title('Evolution of ' + name)
        # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
        plt.show()
        return

    # printData()

    # drawData(X, Y, 0.5*(u1[:-1,:]+u1[1:,:]), 'u1')
    # drawData(X, Y, 0.5*(u2[:,:-1]+u2[:,1:]), 'u2')
    # drawData(X, Y, ps, 'p')

    drawData2(u1colle, y, 'u1')

    # print("residual:", residual)
    return 0

def main():

    x, y, dt, u1, u2, inu2, outu2, ps, pp, rho, mu, U = init()
    u1colle = np.zeros((1,len(u1[0,:])),dtype=float)

    iterNum = 0
    # main iteration step (SIMPLE Method)
    for i in range(301):
        # while(1): # need set with iteration exit condition
        u1, u2, inu2, outu2, ps, pp, dcolle = tscheme(x[1]-x[0], y[1]-y[0], dt, u1, u2, inu2, outu2, rho, ps, pp, mu, U)

        print("max divergence: ", checkDiv(u1, u2, x[1]-x[0], y[1]-y[0]), "   iterNum: ", iterNum)
        # print("[15,4] d: ", dcolle, "   iterNum: ", iterNum)
        # print("semiU1-u1: ", np.max(semiRhoU1/rho-u1), "   iterNum: ", i)
        # if i == 299:
        # print("semiU1-u1: ", np.max(semiRhoU1/rho-u1))
        # print("semiU2-u2: ", np.max(semiRhoU2/rho-u2))
        u1, u2, outu2 = velBCupdate(u1, u2, U, outu2)
        iterNum += 1
        for j in [4,20,50,150,300]:
            if i == j:
                u1colle = np.append(u1colle, u1[10,:][np.newaxis, :], axis=0)
                break

    postProgress(u1, u2, inu2, outu2, ps, x, y, u1colle)

    # postProgress(u1, u2, inu2, outu2, ps, x, y)

    return 0

if __name__ == "__main__":
    main()