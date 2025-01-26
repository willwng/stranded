import numpy as np

from rod.rod import RodState, InitialRodState, RodParams


class Energy:
    def __init__(self):
        pass

    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, rod_state: RodState, init_rod_state: InitialRodState,
                       rod_params: RodParams):
        """ Returns the energy """
        raise NotImplementedError

    @staticmethod
    def d_energy_d_theta(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams):
        """ Updates grad to include the gradient of the energy wrt theta """
        raise NotImplementedError

    @staticmethod
    def d_energy_d_pos(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams):
        """ Updates grad to include the gradient of the energy wrt position """
        raise NotImplementedError
