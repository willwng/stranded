import numpy as np

from energies.energy import Energy
from solver.solver_params import SolverParams


class Inextensibility(Energy):
    @staticmethod
    def constraint(pos: np.ndarray, solver_params: SolverParams):
        pos = pos.reshape(-1, 3)
        violation = 0.0
        # All edges
        for i in range(solver_params.n + 1):
            e_i = pos[i + 1] - pos[i]
            l_i_sq = np.linalg.norm(e_i) ** 2
            l_i_bar_sq = solver_params.l_bar_edge[i] ** 2
            violation += l_i_sq - l_i_bar_sq
        return violation

    @staticmethod
    def d_constraint_d_pos(pos: np.ndarray, solver_params: SolverParams):
        pos = pos.reshape(-1, 3)
        d_constraint_d_pos = np.zeros_like(pos)
        # All edges, update gradient of previous and next node
        for i in range(solver_params.n + 1):
            e_i = pos[i + 1] - pos[i]
            l_i_sq = np.linalg.norm(e_i) ** 2
            l_i_bar_sq = solver_params.l_bar_edge[i] ** 2

            d_constraint_d_pos[i] += 2 * e_i * (l_i_sq - l_i_bar_sq)
            d_constraint_d_pos[i + 1] -= 2 * e_i * (l_i_sq - l_i_bar_sq)

        return d_constraint_d_pos
