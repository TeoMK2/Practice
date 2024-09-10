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

def BCSet(u1 ,u2, pp, U):
    # upper BC
    u1[-1,:] = U
    u2[-1,:] = 0
    pp[-1,:] = 0
    # lower BC
    u1[0,:] = 0
    u2[0,:] = 0
    pp[0,:] = 0
    # inlet BC
    u2[:,0] = 0
    pp[:,0] = 0
    # outlet BC
    pp[:,-1] = 0
    return u1, u2, pp

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

        nt = 100

        return x, y, U, mu, nt, Courant

    def originalVariablesInit(x, y):
        # rho = np.zeros((len(y),len(x)),dtype=float)
        p = np.zeros((len(y),len(x)),dtype=float)
        u1 = np.zeros((len(y),len(x)+1),dtype=float)    #rho*u1
        u2 = np.zeros((len(y)+2,len(x)+1),dtype=float)  #rho*u2
        return u1, u2, p

    def flowFieldInit(u1, u2, pstar):
        u1[:,:] = 0
        u2[:,:] = 0
        pstar[:,:] = 0
        pp = np.zeros_like(pstar)
        u1[14,4] = 0.5 #ft/s
        return u1, u2, pstar, pp

    x ,y ,U, mu, nt, Courant = parameterInput()
    u1, u2, pstar = originalVariablesInit(x, y)
    u1, u2, pstar, pp = flowFieldInit(u1, u2, pstar)
    u1, u2, pp = BCSet(u1 ,u2, pp, U)

    return x, y, u1, u2, pstar, pp, nt, mu, Courant

def tscheme(x, y, rhoU1, rhoU2, ps, pp, mu, Courant):

    def semiStep(dx, dy, dt, mu):
        # calculate velocity with divergence
        visTerm1 = mu*( (u1[i+1,j]-2*u1[i,j]+u1[i-1,j])/dx**2 + (u1[i,j+1]-2*u1[i,j]+u1[i,j-1])/dy**2 )
        presTerm1 = (p[i+1,j] - p[i,j])/dx
        convTerm1 = -((rho[i+1,j]*u1[i+1,j]**2        - rho[i,j]*u1[i,j]**2)/2/dx +
                      (rho[i,j+1]*u1[i,j+1]*u2[i,j+1] - rho[i,j]*u1[i,j]*u2[i,j])/2/dy)

        visTerm2 = mu*( (u2[i+1,j]-2*u2[i,j]+u2[i-1,j])/dx**2 + (u2[i,j+1]-2*u2[i,j]+u2[i,j-1])/dy**2 )
        presTerm2 = (p[i,j+1] - p[i,j])/dy
        convTerm2 = -((rho[i+1,j]*u1[i+1,j]*u2[i+1,j] - rho[i,j]*u1[i,j]+u2[i,j])/2/dx +
                      (rho[i,j+1]*u2[i,j+1]**2        - rho[i,j]*u2[i,j]**2)/2/dy)

        A = convTerm1 + visTerm1 - presTerm1
        B = convTerm2 + visTerm2 - presTerm2
        newRhoU1[i,j] = rho[i,j]*u1[i,j] + A*dt
        newRhoU2[i,j] = rho[i,j]*u2[i,j] + B*dt

        return  newRhoU1, newRhoU2

    def pCorrect(rhoU1, rhoU2, ps, dx, dy, dt):

        pp = pIncrement(rhoU1, rhoU2, ps, dx, dy, dt)
        newP = ps + pp

        return newP

    def pIncrement(rhoU1, rhoU2, ps, dx, dy, dt):

        #   a*ps[i,j] + b*(ps[i+1,j] + ps[i-1,j]) + c*(ps[i,j+1] + ps[i,j-1]) + d = 0
        a = 2*(1/dx**2 + 1/dy**2)*dt
        b = -dt/dx**2
        c = -dt/dy**2
        d = (rhoU1[i+1,j] - rhoU1[i,j])/dx + (rhoU2[i,j+1] - rhoU2[i,j])/dy

        # solve Poisson equation
        for i in range(Lx):
            for j in range(Ly):
                ps[i,j] = (b*(ps[i+1,j] + ps[i-1,j]) + c*(ps[i,j+1] + ps[i,j-1]) + d)/a

        return ps

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dt = Courant*np.sqrt(dx**2 + dy**2)

    newRhoU1, newRhoU2 = semiStep(dx, dy, dt, mu)

    newP = pCorrect(newRhoU1, newRhoU2, ps, dx, dy, dt)

    return newRhoU1, newRhoU2, newP

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

    x, y, u1, u2, ps, pp, nt, mu, Courant = init()
    for t in range(nt):

        newRhoU1, newRhoU2, newP = tscheme(x, y, rhoU1, rhoU2, ps, pp, mu, Courant)

        rhoU1 = newRhoU1
        rhoU2 = newRhoU2
        ps = newP
    # postProgress(collector_u1, y, tStepNumb, totalt, residual)

    return 0

if __name__ == "__main__":
    main()