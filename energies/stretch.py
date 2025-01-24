import numpy as np

from energies.energy import Energy
from solver.solver_params import SolverParams


class Stretch(Energy):
    """ Represents the stretching energy of a rod (based on inextensible edges) """

    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        energy = 0.0
        # All edges
        for i in range(solver_params.n + 1):
            e_i = pos[i + 1] - pos[i]
            l_i = np.linalg.norm(e_i)
            l_bar_i = solver_params.l_bar_edge[i]
            energy += 0.5 * solver_params.k * (l_i - l_bar_i) ** 2 / l_bar_i
        return energy

    @staticmethod
    def d_energy_d_theta(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        # Trivial
        d_energy_d_theta = np.zeros_like(theta)
        return d_energy_d_theta

    @staticmethod
    def d_energy_d_pos(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        d_energy_d_pos = np.zeros_like(pos)
        # All edges, update gradient of previous and next node
        for i in range(solver_params.n + 1):
            e_i = pos[i + 1] - pos[i]
            l_i = np.linalg.norm(e_i)
            l_bar_i = solver_params.l_bar_edge[i]
            d_energy_d_pos[i] -= solver_params.k * (l_i - l_bar_i) * e_i / l_i
            d_energy_d_pos[i + 1] += solver_params.k * (l_i - l_bar_i) * e_i / l_i
        return d_energy_d_pos