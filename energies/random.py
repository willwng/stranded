import numpy as np

from energies.energy import Energy
from rod.helix_util import HelixUtil
from rod.rod import RodState, InitialRodState, RodParams
from solver.solver_params import SolverParams


class RandomForce(Energy):
    def __init__(self, seed: int):
        super().__init__()
        self.seed = seed

    def compute_energy(self, pos: np.ndarray, theta: np.ndarray, rod_state: RodState, init_rod_state: InitialRodState,
                       rod_params: RodParams):
        return HelixUtil.compute_random_potential(pos, self.seed)

    @staticmethod
    def d_energy_d_theta(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams):
        return grad  # No theta dependence

    def d_energy_d_pos(self, grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams):
        grad += HelixUtil.compute_random_force(pos, self.seed)
        return grad
