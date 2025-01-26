import numpy as np

from energies.energy import Energy
from solver.solver_params import SolverParams


class Stretch(Energy):
    """
    Represents the stretching energy of a rod (based on inextensible edges)
        - Don't use this! For inextensibility we use XPBD to solve the constraint
            rather than a specific energy
    """

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
    def d_energy_d_theta(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        return grad  # No theta dependence

    @staticmethod
    def d_energy_d_pos(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        # All edges, update gradient of previous and next node
        for i in range(solver_params.n + 1):
            e_i = pos[i + 1] - pos[i]
            l_i = np.linalg.norm(e_i)
            l_bar_i = solver_params.l_bar_edge[i]
            grad[i] -= solver_params.k * (l_i - l_bar_i) * e_i / l_i
            grad[i + 1] += solver_params.k * (l_i - l_bar_i) * e_i / l_i
        return grad
