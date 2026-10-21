
import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from scipy.special import sici


class GalaxyOccupation(ABC):

    def __init__(self):
        self.pars = {}
        return



class Gaussian(GalaxyOccupation):


    def u(self, ks):

        return

    
    


MODELS = {'gauss': Gaussian}