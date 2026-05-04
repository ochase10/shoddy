'''
utility functions
'''

import numpy as np

G = 4.30091727e-9 # km^2 * Mpc / s^2 / Msun
C = 2.99792458e6 # km/s
LN10 = np.log(10)


# TODO figure out whether I want these to be functional for arrays

def a2z(a:float) -> float:
    return float(np.where(a <= 1 and a > 0, 1/a - 1, np.inf))

def z2a(z:float) -> float:
    return float(np.where(z>0, 1/(1+z), 1))

def dM2dlogM(val, M):
    return val * M * LN10

def dlogM2dM(val, M):
    return val / (LN10 * M)