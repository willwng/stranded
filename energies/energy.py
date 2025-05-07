import cupy as cp

from rod.rod import RodState, InitialRodState, RodParams


class Energy:
    """ Abstract class for all energies (GPU-compatible) """

    def __init__(self):
        pass

    @staticmethod
    def compute_energy(pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams) -> float:
        """ Returns the energy """
        raise NotImplementedError

    @staticmethod
    def d_energy_d_theta(grad: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams) -> cp.ndarray:
        """ Updates grad to include the gradient of the energy wrt theta """
        raise NotImplementedError

    @staticmethod
    def d2_energy_d_theta2(hess: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                           init_rod_state: InitialRodState, rod_params: RodParams) -> cp.ndarray:
        """ Updates hess to include the hessian of the energy wrt theta """
        raise NotImplementedError

    @staticmethod
    def d_energy_d_pos(grad: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams) -> cp.ndarray:
        """ Updates grad to include the gradient of the energy wrt position """
        raise NotImplementedError

