import numpy as np

# this program is aim to calculate Prandtl-Meyer flow with numerical simulation.
#
# Finite Different Method
#
# Conservative form:    d(F)/dx              = -d(G)/dy              + J
#
# Continuity:           d(ρu)/dt             = -d(ρu)/dx             + 0
#
# Momentum:             d(ρu^2+p)/dt         = -d(ρuv)/dx            + 0
#                       d(ρuv)/dt            = -d(ρv^2+p)/dx         + 0
#
# Energy:               d(ρu(e+V^2/2)+pu)/dt = -d(ρv(e+V^2/2)+pv)/dx + 0


# F1 = rho * u
# F2 = rho * np.square(u) + p
# F3 = rho * u * v
# F4 = gamma/(gamma-1)*p*u + rho*u*(np.square(u) + np.square(v))/2

# G1 = rho * v
# G2 = rho * u * v
# G3 = rho * np.square(v) + p
# G4 = gamma/(gamma-1)*p*v + rho*v*(np.square(u) + np.square(v))/2

#TODO

