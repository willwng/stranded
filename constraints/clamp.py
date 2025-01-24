import numpy as np

from energies.energy import Energy
from solver.solver_params import SolverParams


class Clamp(Energy):
    @staticmethod
    def constraint(pos: np.ndarray, solver_params: SolverParams):
        pos = pos.reshape(-1, 3)
        violation = np.linalg.norm(pos[-1] - solver_params.pos0) ** 2
        return violation

    @staticmethod
    def d_constraint_d_pos(pos: np.ndarray, solver_params: SolverParams):
        pos = pos.reshape(-1, 3)
        d_constraint_d_pos = np.zeros_like(pos)
        d_constraint_d_pos[-1] = 2 * (pos[-1] - solver_params.pos0)
        return d_constraint_d_pos
