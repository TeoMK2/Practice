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

def velBCSet(u1, u2, U):
    # upper BC
    u1[-1,:] = U
    u2[-1,:] = 0
    # lower BC
    u1[0,:] = 0
    u2[0,:] = 0
    # inlet BC
    u2[:,0] = 0
    return u1, u2

def presBCSet(pp):
    # upper
    pp[-1,:] = 0
    # lower
    pp[0,:] = 0
    # inlet
    pp[:,0] = 0
    # outlet
    pp[:,-1] = 0
    return pp

def init():

    def parameterInput():
        Lx = 0.5    #ft
        Ly = 0.01   #ft
        Nx = 21
        Ny = 11
        x = np.linspace(0,Lx, Nx)
        y = np.linspace(0,Ly, Ny)
        U = 0.002377 #slug/ft^3

        Courant = 0.5
        mu = 1e-5
        rho = 1

        nt = 100

        return x, y, U, mu, rho, nt, Courant

    def originalVariablesInit(x, y):
        p = np.zeros((len(x),len(y)),dtype=float)
        u1 = np.zeros((len(x)+1,len(y)),dtype=float)
        u2 = np.zeros((len(x),len(y)+1),dtype=float)
        inu2 = np.zeros((1,len(y)+1),dtype=float)   #Ghost Cell
        outu2 = np.zeros((1,len(y)+1),dtype=float)   #Ghost Cell
        return u1, u2, p, inu2, outu2

    def flowFieldInit(u1, u2, U, pstar):
        u1[:,:] = 0
        u2[:,:] = 0
        pstar[:,:] = 0
        pp = np.zeros_like(pstar)
        u1[14,4] = 0.5 #ft/s
        u1, u2 = velBCSet(u1, u2, U)

        return u1, u2, pstar, pp

    x ,y ,U, mu, rho, nt, Courant = parameterInput()
    u1, u2, ps, inu2, outu2 = originalVariablesInit(x, y)
    u1, u2, ps, pp = flowFieldInit(u1, u2, U, ps)
    pp = presBCSet(pp)

    return x, y, u1, u2, inu2, outu2, ps, pp, rho, nt, mu, Courant, U

def tscheme(x, y, u1, u2, inu2, outu2, rho, ps, pp, mu, Courant, U):

    def semiStep(u1, u2, inu2, outu2, ps, rho, dx, dy, dt, mu):
        # calculate velocity with divergence
        newRhoU1, newRhoU2 = np.zeros_like(u1), np.zeros_like(u2)

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

        for i in range(1, len(u2[:,0])-1):
            for j in range(1, len(u2[0,:])-1):
                pJ = j - 1
                iu1 = i + 1
                ju1 = j - 1
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
        # solve Poisson equation using Relaxation Method
        for i in range(1,len(pp[:,0])-1):
            for j in range(1,len(pp[0,:])-1):
                # iu1 = i + 1
                # ju2 = j + 1
                d = (rhoU1[i+1,j] - rhoU1[i,j])/dx + (rhoU2[i,j+1] - rhoU2[i,j])/dy
                pp[i,j] = (b*(pp[i+1,j] + pp[i-1,j]) + c*(pp[i,j+1] + pp[i,j-1]) + d)/a
        return pp

    def uCorrect(semiRhoU1, semiRhoU2, pp, dx, dy, dt):

        newRhoU1, newRhoU2 = np.zeros_like(semiRhoU1), np.zeros_like(semiRhoU2)

        for i in range(1,len(semiRhoU1[:,0])-1):
            for j in range(1,len(semiRhoU1[0,:])-1):
                pi = i - 1
                newRhoU1[i,j] = semiRhoU1[i,j] + dt/dx*(pp[pi+1,j] - pp[pi,j])
        for i in range(1,len(semiRhoU2[:,0])-1):
            for j in range(1,len(semiRhoU2[0,:])-1):
                pj = j - 1
                newRhoU2[i,j] = semiRhoU2[i,j] + dt/dy*(pp[i,pj+1] - pp[i,pj])
        return newRhoU1, newRhoU2

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = Courant*np.sqrt(dx**2 + dy**2)

    semiRhoU1, semiRhoU2 = semiStep(u1, u2, inu2, outu2, ps, rho, dx, dy, dt, mu)
    semiRhoU1, semiRhoU2 = velBCSet(semiRhoU1/rho, semiRhoU2/rho, U)*rho
    pp = pCorrect(semiRhoU1, semiRhoU2, pp, dx, dy, dt)
    pp = presBCSet(pp)

    newP = ps + pp
    newRhoU1, newRhoU2 = uCorrect(semiRhoU1, semiRhoU2, pp, dx, dy, dt)
    newU1, newU2 = velBCSet(newRhoU1/rho, newRhoU2/rho, U)

    return newU1, newU2, newP

def postProgress(u1, y, tStepNumb, totalt):


    def printData():
        print("------------solve complete.------------")
        print("iteration or temporal advancement times:", tStepNumb)
        print("total physical time:", totalt)

        print("---------------------------------------")
        print("u1:", u1)
        print("y:", y)

        return

    def drawData():
        plt.figure()
        # fig, ax = plt.subplots(figsize=(7, 5))  # 图片尺寸
        for i in range(len(u1)):
            plt.plot(u1[i], y, '-o', linewidth=1.0, color='black', markersize=1)
        # plt.savefig('G:/vorVel.png', bbox_inches='tight', dpi=512)  # , transparent=True
        plt.show()
        return

    printData()

    drawData()

    # print("residual:", residual)
    return 0

def main():

    x, y, u1, u2, inu2, outu2, ps, pp, rho, nt, mu, Courant, U = init()
    for t in range(nt):

        newU1, newU2, newP = tscheme(x, y, u1, u2, inu2, outu2, rho, ps, pp, mu, Courant, U)

        u1 = newU1
        u2 = newU2
        ps = newP
    # postProgress(collector_u1, y, tStepNumb, totalt, residual)

    return 0

if __name__ == "__main__":
    main()