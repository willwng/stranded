import numpy as np

from energies.energy import Energy
from maths.vectors import Vector
from solver.solver_params import SolverParams


class Bend(Energy):
    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        energy = 0.0
        # From edge 1 to n
        for i in range(1, solver_params.n + 1):
            den = 1 / (2 * solver_params.l_bar[i])
            for j_idx, j in enumerate([i - 1, i]):
                omega_bar_ij = solver_params.omega_bar[i, j_idx]
                omega_ij = Bend.compute_omega(pos, theta, i, j, solver_params)
                d_omega_ij = omega_ij - omega_bar_ij
                energy += den * np.dot(d_omega_ij, solver_params.B[j] @ d_omega_ij)
        return energy

    @staticmethod
    def d_energy_d_theta(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        d_energy_d_theta = np.zeros_like(theta)
        # All edges
        for j in range(0, solver_params.n + 1):
            l_j = solver_params.l_bar[j]
            B_j = solver_params.B[j]

            # First term
            if j != 0:
                omega_jj = Bend.compute_omega(pos, theta, j, j, solver_params)
                omega_bar_jj = solver_params.omega_bar[j, 1]
                d_energy_d_theta[j] = (1 / l_j) * np.dot(omega_jj, Vector.J @ (B_j @ (omega_jj - omega_bar_jj)))

            # Second term, only if not last edge
            if j != solver_params.n:
                l_jp1 = solver_params.l_bar[j + 1]
                omega_jp1j = Bend.compute_omega(pos, theta, j + 1, j, solver_params)
                omega_bar_jp1j = solver_params.omega_bar[j + 1, 0]
                d_energy_d_theta[j] += (1 / l_jp1) * np.dot(omega_jp1j, Vector.J @ (B_j @ (omega_jp1j - omega_bar_jp1j)))
        return d_energy_d_theta

    @staticmethod
    def compute_curvature_binormal(pos: np.ndarray, i: int, solver_params: SolverParams):
        """ Computes the curvature binormal vector at vertex i """
        e_i, e_im1 = (pos[i + 1] - pos[i]), (pos[i] - pos[i - 1])
        kb_i = np.cross(2 * e_im1, e_i)
        l_bar_i, l_bar_im1 = solver_params.l_bar_edge[i], solver_params.l_bar_edge[i - 1]
        kb_i /= (l_bar_i * l_bar_im1 + np.dot(e_im1, e_i))
        return kb_i

    @staticmethod
    def compute_omega(pos: np.ndarray, theta: np.ndarray, i: int, j: int, solver_params: SolverParams):
        """ Computes the material curvature """
        kb_i = Bend.compute_curvature_binormal(pos, i, solver_params)
        # Compute the material frame
        u, v = solver_params.bishop_frame[j]
        theta_j = theta[j]
        m_1 = np.cos(theta_j) * u + np.sin(theta_j) * v
        m_2 = -np.sin(theta_j) * u + np.cos(theta_j) * v
        omega_ij = np.array([np.dot(kb_i, m_2), -np.dot(kb_i, m_1)])
        return omega_ij

    @staticmethod
    def d_energy_d_pos(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        d_energy_d_pos = np.zeros_like(pos)
        return d_energy_d_pos