import numpy as np

from energies.energy import Energy
from solver.solver_params import SolverParams


class Gravity(Energy):
    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        energy = 0.0
        for i in range(solver_params.n + 2):
            energy += solver_params.mass[i] * solver_params.g * pos[i, 2]

        return energy

    @staticmethod
    def d_energy_d_theta(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        d_energy_d_theta = np.zeros_like(theta)
        return d_energy_d_theta

    @staticmethod
    def d_energy_d_pos(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        d_energy_d_pos = np.zeros_like(pos)
        for i in range(solver_params.n + 2):
            d_energy_d_pos[i, 2] = solver_params.mass[i] * solver_params.g
        return d_energy_d_pos
