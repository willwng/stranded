import numpy as np

from energies.energy import Energy
from solver.solver_params import SolverParams


class Gravity(Energy):
    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        energy = np.sum(solver_params.mass * solver_params.g * pos[:, 2])
        return energy

    @staticmethod
    def d_energy_d_theta(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        return grad  # No theta dependence

    @staticmethod
    def d_energy_d_pos(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        grad[:, 2] += np.multiply(solver_params.mass, solver_params.g)
        return grad
