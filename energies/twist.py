import cupy as cp

from energies.energy import Energy
from rod.rod import RodState, InitialRodState, RodParams


class Twist(Energy):
    @staticmethod
    def compute_energy(pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams) -> float:
        l = init_rod_state.l_bar
        energy = cp.sum(rod_params.beta * (theta[1:] - theta[:-1])**2 / l[1:])
        return energy

    @staticmethod
    def d_energy_d_theta(grad: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams) -> cp.ndarray:
        l_bar = init_rod_state.l_bar
        beta = rod_params.beta
        diff = theta[1:] - theta[:-1]
        grad[1:] += 2 * beta * diff / l_bar[1:]
        grad[:-1] -= 2 * beta * diff / l_bar[1:]
        return grad

    @staticmethod
    def d2_energy_d_theta2(hess: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                           init_rod_state: InitialRodState, rod_params: RodParams) -> cp.ndarray:
        l_bar = init_rod_state.l_bar
        beta = rod_params.beta
        j = cp.arange(1, theta.size)

        hess[j, j - 1] -= 2 * beta / l_bar[j]
        hess[j, j + 1] -= 2 * beta / l_bar[j + 1]
        hess[j, j] += 2 * beta / l_bar[j] + 2 * beta / l_bar[j + 1]
        return hess

    @staticmethod
    def d_energy_d_pos(grad: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams) -> cp.ndarray:
        return grad  # gradient wrt pos is handled in BendTwist
