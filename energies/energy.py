import jax.numpy as np

from rod.rod import RodState, InitialRodState, RodParams


class Energy:
    """ Abstract class for all energies """

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
    def d2_energy_d_theta2(hess: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                           init_rod_state: InitialRodState, rod_params: RodParams):
        """ Updates hess to include the hessian of the energy wrt theta """
        raise NotImplementedError

    @staticmethod
    def d_energy_d_pos(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams):
        """ Updates grad to include the gradient of the energy wrt position """
        raise NotImplementedError
