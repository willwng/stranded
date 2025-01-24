import numpy as np

from solver.solver_params import SolverParams

class Energy:
    def __init__(self):
        pass

    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        raise NotImplementedError

    @staticmethod
    def d_energy_d_theta(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        raise NotImplementedError

    @staticmethod
    def d_energy_d_pos(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        raise NotImplementedError
