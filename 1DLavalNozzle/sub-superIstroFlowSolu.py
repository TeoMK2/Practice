import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.0,3.0,31)
specificHeatRatio = 1.4
areaRatio = 1 + 2.2 * np.square(x - 1.5)

fig, axes = plt.subplots(4, 1, figsize=(5, 10))
plt.subplots_adjust(hspace=0.5)
#-----------------Mach number-----------------------
# axes[0].plot(t, data)
axes[0].set_title(r'Evolution of Ma')
axes[0].set_xlabel(r'x')
axes[0].set_ylabel(r'Ma')
# axes[0].set_xticks(np.arange(0, t[len(t)-1],1))
#--------------------pressure-----------------------
# axes[1].plot(xf, np.abs(yf))
axes[1].set_title(r'ratio of pressure')
axes[1].set_xlabel(r'x')
axes[1].set_ylabel(r'p_e/p0')
# axes[1].set_xlim(-0.5, 5)
#--------------------density------------------------
axes[2].set_title(r'ratio of density')
axes[2].set_xlabel(r'x')
axes[2].set_ylabel(r'\rho_e/\rho_0')
#------------------Temperature----------------------
axes[3].set_title(r'ratio of Temperature')
axes[3].set_xlabel(r'x')
axes[3].set_ylabel(r'T_e/T_0')


plt.savefig('solution.jpg')
plt.show()