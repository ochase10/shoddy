
import numpy as np
from .utils import Parameterized

class GalaxyOccupation(Parameterized):

    PARS = {'corr': -1}



def r2sig(r):
    return r / np.sqrt(2*np.log(2))

class Gaussian(GalaxyOccupation):

    PARS = {"R": 0}

    def _setup(self):
        self.sig2 = r2sig(self.pars['R'])**2

    def rho(self, r):
        norm = (2 * np.pi * self.sig2)**(-3/2)
        prof = np.exp(-r**2 / (2*self.sig2))
        return norm * prof

    def u(self, k):
        return np.exp(-k**2 * self.sig2 / 2)


class Exponential(GalaxyOccupation):

    PARS = {"R": 0}

    def rho(self, r):
        norm = (8 * np.pi * self.pars['R']**3)
        prof = np.exp(-r / self.pars['R'])
        return prof / norm

    def u(self, k):
        return (1 + k**2 * self.pars['R']**2)**-2


MODELS = {'gauss': Gaussian,
          'expo': Exponential}