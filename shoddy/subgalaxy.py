
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from scipy.special import sici


class GalaxyOccupation(ABC):

    def __init__(self):
        self.pars = {}
        return

    def update_pars(self, **new_pars):
        self.pars.update(new_pars)


class Gaussian(GalaxyOccupation):

    PARS = ()

    def u(self, ks):

        return



MODELS = {'gauss': Gaussian}