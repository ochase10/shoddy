
from abc import ABC, abstractmethod
import numpy as np
from scipy.special import erf
from numpy.typing import ArrayLike, NDArray



class HOD(ABC):

    def __init__(self, **kwargs):
        self.pars = {}


    @abstractmethod
    def satellites(self, M_halo) -> NDArray[np.floating]:
        pass


    @abstractmethod
    def centrals(self, M_halo) -> NDArray[np.floating]:
        pass


    def N_hod(self, M_halo) -> NDArray[np.floating]:
        return self.centrals(M_halo) + self.satellites(M_halo)
    
        
    def update_pars(self, **new_pars):
        self.pars.update(new_pars)
    

    def avg_NcNs(self, M_halo):
        return self.centrals(M_halo) * self.satellites(M_halo)


    def avg_Ns2(self, M_halo):
        return self.satellites(M_halo)**2




class Zheng07(HOD):


    def __init__(self,
                 M_min,
                 sig_logM,
                 M0,
                 M1,
                 alpha,
                 dc=1.,
                 **kwargs):
        
        super().__init__(**kwargs)
        
        self.pars = {'M_min': M_min,
                     'sig_logM': sig_logM, 
                     'M0': M0,
                     'M1': M1,
                     'alpha': alpha,
                     'dc': dc}


    def centrals(self, M_halo) -> NDArray[np.floating]:
        return 0.5 * self.pars['dc'] * (1 + erf((np.log10(M_halo) - np.log10(self.pars['M_min'])) / self.pars['sig_logM']))

    def satellites(self, M_halo) -> NDArray[np.floating]:
        return self.pars['dc'] * (np.where(M_halo>self.pars['M0'],(M_halo - self.pars['M0']),0) / self.pars['M1'])**self.pars['alpha'] * self.centrals(M_halo)
    
    def __str__(self) -> str:
        return 'Zheng07' #TODO add current pars to print

