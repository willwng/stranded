from energies.energy import Energy
from solver.solver_params import SolverParams

import numpy as np


class Twist(Energy):
    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        # From edge 1 to edge n
        ind = np.arange(1, solver_params.n + 1)
        theta_i, theta_im1 = theta[ind], theta[ind - 1]
        l_i = solver_params.l_bar[ind]
        m_i = theta_i - theta_im1
        energy = np.sum(solver_params.beta * m_i ** 2 / l_i)
        return energy

    @staticmethod
    def d_energy_d_theta(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        # All edges
        for j in range(0, solver_params.n + 1):
            theta_j = theta[j]
            if j != 0:
                theta_jm1 = theta[j - 1]
                l_j = solver_params.l_bar[j]
                m_j = theta_j - theta_jm1
                grad[j] += 2 * solver_params.beta * (m_j / l_j)

            # Second term, only if not last edge
            if j != solver_params.n:
                theta_jp1 = theta[j + 1]
                l_jp1 = solver_params.l_bar[j + 1]
                m_jp1 = theta_jp1 - theta_j
                grad[j] -= 2 * solver_params.beta * (m_jp1 / l_jp1)
        return grad

    @staticmethod
    def d_energy_d_pos(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        return grad  # Handled in BendTwist
