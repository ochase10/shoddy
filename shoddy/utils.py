'''
utility functions
'''

import numpy as np

G = 4.30091727e-9 # km^2 * Mpc / s^2 / Msun
C = 2.99792458e6  # km/s
LN10 = np.log(10)

_trapz = getattr(np, 'trapezoid', np.trapz)


def a2z(a: float) -> float:
    if a <= 0 or a > 1:
        return float('inf')
    return 1.0 / a - 1.0

def z2a(z: float) -> float:
    return 1.0 / (1.0 + z)
