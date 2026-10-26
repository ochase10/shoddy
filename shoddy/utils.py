'''
utility functions
'''

from abc import ABC
import numpy as np

G = 4.30091727e-9 # km^2 * Mpc / s^2 / Msun
C = 2.99792458e5  # km/s
LN10 = np.log(10)

REQUIRED = object()

_trapz = getattr(np, 'trapezoid', np.trapz)

def lookup(reg_dict, name, label):
    if isinstance(name, type):
        return name
    try:
        return reg_dict[name.lower()]
    except KeyError:
        raise ValueError(f"Unknown {label} '{name}'. Options: {list(reg_dict)}") from None


class Parameterized(ABC):

    PARS = {}

    def __init__(self, **pars):
        self.pars = self.PARS.copy()
        self._check_args(pars.keys())
        self.pars.update(pars)
        self._verify()
        self._setup()


    def __init_subclass__(cls):
        cls.PARS = {**super(cls, cls).PARS, **cls.PARS} # type:ignore


    def _check_args(self, keys):
        for par in keys:
            if par not in self.pars.keys():
                raise ValueError(f"{par} not a recognized parameter")

    def _verify(self):
        for key, value in self.pars.items():
            if value is REQUIRED:
                raise ValueError(f"Required parameter {key} not defined")


    def update_pars(self, **new_pars):
        self._check_args(new_pars.keys())
        self.pars.update(new_pars)
        self._setup()


    def with_pars(self, new_pars):
        return type(self)(**{**self.pars, **new_pars})


    def _setup(self):
        return


def a2z(a: float) -> float:
    if a <= 0 or a > 1:
        return float('inf')
    return 1.0 / a - 1.0

def z2a(z: float) -> float:
    return 1.0 / (1.0 + z)
