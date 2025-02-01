import numpy as np

from energies.energy import Energy
from rod.rod import RodState, InitialRodState, RodParams
from solver.solver_params import SolverParams


class Gravity(Energy):
    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, rod_state: RodState, init_rod_state: InitialRodState,
                       rod_params: RodParams):
        energy = np.sum(rod_params.mass * rod_params.g * pos[:, 2])
        return energy

    @staticmethod
    def d_energy_d_theta(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams):
        return grad  # No theta dependence

    @staticmethod
    def d_energy_d_pos(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState,
                       rod_params: RodParams):
        grad = grad.at[:, 2].add(np.multiply(rod_params.mass, rod_params.g))
        return grad
