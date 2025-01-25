"""
Hacky, only used for gradient calculation wrst position
"""
import numpy as np

from energies.bending import Bend
from energies.energy import Energy
from maths.vectors import Vector
from solver.solver_params import SolverParams


class BendTwist(Energy):
    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        energy = 0.0
        return energy

    @staticmethod
    def d_energy_d_theta(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        d_energy_d_theta = np.zeros_like(theta)
        return d_energy_d_theta

    @staticmethod
    def d_energy_d_pos(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams):
        n = solver_params.n

        # Compute partial E / partial x_i, for all nodes
        d_energy_d_pos = np.zeros_like(pos)
        for i in range(n + 2):
            # k from 1 to n
            for k in range(1, n + 1):
                den = 1 / solver_params.l_bar[k]
                # j from k-1 to k
                for j_idx, j in enumerate([k - 1, k]):
                    B_j = solver_params.B[j]
                    omega_kj = Bend.compute_omega(pos, theta, k, j, solver_params)
                    omega_bar_kj = solver_params.omega_bar[k, j_idx]
                    d_omega_kj = omega_kj - omega_bar_kj

                    # Compute nabla_i omega_kj. Requires m_1j, m_2j
                    theta_j = theta[j]
                    u, v = solver_params.bishop_frame[j]
                    m_1j = np.cos(theta_j) * u + np.sin(theta_j) * v
                    m_2j = -np.sin(theta_j) * u + np.cos(theta_j) * v
                    m_T = np.array([m_2j, -m_1j])
                    # Now compute nabla_i (kb)_k
                    nabla_i_kb_k = BendTwist.nabla_kb(pos=pos, i=k, d=i - k, solver_params=solver_params)
                    # nabla_i psi_j
                    nabla_i_psi_j = BendTwist.nabla_i_psi_j(pos, i, j, solver_params)
                    nabla_i_omega_kj = m_T @ nabla_i_kb_k - Vector.J @ np.outer(omega_kj, nabla_i_psi_j)
                    grad = den * nabla_i_omega_kj.T @ (B_j @ d_omega_kj)
                    d_energy_d_pos[i] += grad

        # Second term of d E / d x_i = -partial E / partial theta_n * partial theta_n / partial x_i
        l_bar_n = solver_params.l_bar[n]
        omega_nn = Bend.compute_omega(pos, theta, n, n, solver_params)
        omega_bar_nn = solver_params.omega_bar[n, 1]
        B_n = solver_params.B[n]
        m_n = theta[n] - theta[n - 1]
        d_energy_d_theta_n = (1 / l_bar_n) * (
                np.dot(omega_nn, Vector.J @ (B_n @ (omega_nn - omega_bar_nn))) + 2 * solver_params.beta * m_n)
        for i in range(n + 2):
            d_energy_d_pos[i] -= d_energy_d_theta_n * BendTwist.nabla_i_psi_j(pos, i, n, solver_params)
        return d_energy_d_pos

    @staticmethod
    def nabla_kb(pos: np.ndarray, i: int, d: int, solver_params: SolverParams):
        """
        Compute nabla_{i-1}(kb)_i, nabla_i(kb)_i, nabla_{i+1}(kb)_i
            represented by d [-1, 0, 1], respectively
        """
        # Only nabla_{i-1}, nabla_i, nabla_{i+1} are non-zero
        if d not in [-1, 0, 1]:
            return np.zeros((3, 3))
        # First compute (kb)_i, and denominator of the gradients
        e_i, e_im1 = (pos[i + 1] - pos[i]), (pos[i] - pos[i - 1])
        l_bar_i, l_bar_im1 = solver_params.l_bar_edge[i], solver_params.l_bar_edge[i - 1]
        kb_i = Bend.compute_curvature_binormal(pos, i, solver_params)
        den = l_bar_i * l_bar_im1 + np.dot(e_im1, e_i)

        # Gradients of curvature binormal
        nabla_im1 = (2 * Vector.skew_sym(e_i) + np.outer(kb_i, e_i)) / den
        nabla_ip1 = (2 * Vector.skew_sym(e_im1) - np.outer(kb_i, e_im1)) / den
        nabla_i = -(nabla_im1 + nabla_ip1)
        if d == -1:
            return nabla_im1
        elif d == 1:
            return nabla_ip1
        return nabla_i

    @staticmethod
    def nabla_di_psi_i(pos: np.ndarray, i: int, d: int, solver_params: SolverParams):
        """ Computes nabla_{i-1} psi_i, nabla_i psi_i, nabla_{i+1} psi_i """
        # Only nabla_{i-1}, nabla_i, nabla_{i+1} are non-zero
        if d not in [-1, 0, 1]:
            return 0.0

        # First compute the curvature binormal
        kb_i = Bend.compute_curvature_binormal(pos, i, solver_params)

        # Gradients given by Eq. 9
        if d == 0:
            mag_ei_bar, mag_eim1_bar = solver_params.l_bar_edge[i], solver_params.l_bar_edge[i - 1]
            nabla_im1 = kb_i / (2 * mag_eim1_bar)
            nabla_ip1 = -kb_i / (2 * mag_ei_bar)
            nabla = -(nabla_im1 + nabla_ip1)
        elif d == -1:
            mag_eim1_bar = solver_params.l_bar_edge[i - 1]
            nabla = kb_i / (2 * mag_eim1_bar)
        else:
            mag_ei_bar = solver_params.l_bar_edge[i]
            nabla = -kb_i / (2 * mag_ei_bar)
        return nabla

    @staticmethod
    def nabla_i_psi_j(pos: np.ndarray, i: int, j: int, solver_params: SolverParams):
        """ Computes the gradient of Psi_j wrst i """
        grad = np.zeros(3)
        # From k = 1 to j
        for k in range(1, j + 1):
            grad += BendTwist.nabla_di_psi_i(pos, k, i - k, solver_params)
        return grad
