
from abc import abstractmethod
import numpy as np
from scipy.special import erf
from numpy.typing import NDArray
from .utils import REQUIRED, Parameterized



class HOD(Parameterized):

    PARS = {'dc': 1}

    @abstractmethod
    def satellites(self, M_halo) -> NDArray[np.floating]:
        pass


    @abstractmethod
    def centrals(self, M_halo) -> NDArray[np.floating]:
        pass


    def N_hod(self, M_halo) -> NDArray[np.floating]:
        return self.centrals(M_halo) + self.satellites(M_halo)
    

    def avg_NcNs(self, M_halo):
        return self.centrals(M_halo) * self.satellites(M_halo)


    def avg_Ns2(self, M_halo):
        return self.satellites(M_halo)**2



class Zheng07(HOD):

    PARS = {'M_min':REQUIRED,
            'sig_logM':REQUIRED,
            'M0':REQUIRED,
            'M1':REQUIRED,
            'alpha':REQUIRED}

    def centrals(self, M_halo) -> NDArray[np.floating]:
        return self.pars['dc'] * 0.5 * (1 + erf((np.log10(M_halo) - np.log10(self.pars['M_min'])) / self.pars['sig_logM']))

    def satellites(self, M_halo) -> NDArray[np.floating]:
        return (np.where(M_halo>self.pars['M0'],(M_halo - self.pars['M0']),0) / self.pars['M1'])**self.pars['alpha'] * self.centrals(M_halo)
    
    def __str__(self) -> str:
        return 'Zheng07' #TODO add current pars to print



MODELS = {'zheng07': Zheng07}