from energies.energy import Energy
from rod.rod import RodState, InitialRodState, RodParams

import numpy as np


class Twist(Energy):
    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, rod_state: RodState, init_rod_state: InitialRodState,
                       rod_params: RodParams):
        l = init_rod_state.l_bar
        energy = np.sum(rod_params.beta * (theta[1:] - theta[:-1]) ** 2 / l[1:])
        return energy

    @staticmethod
    def d_energy_d_theta(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams):
        l_bar = init_rod_state.l_bar
        beta = rod_params.beta
        # All but first edge, all but last theta
        grad[1:] += 2 * beta * (theta[1:] - theta[:-1]) / l_bar[1:]
        grad[:-1] -= 2 * beta * (theta[1:] - theta[:-1]) / l_bar[1:]
        return grad

    @staticmethod
    def d_energy_d_pos(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams):
        return grad  # Handled in BendTwist
