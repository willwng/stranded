"""
Hacky, only used for gradient calculation wrst position
"""
import numpy as np

from energies.bend import Bend
from energies.energy import Energy
from math_util.vectors import Vector
from rod.rod import RodState, InitialRodState, RodParams
from rod.rod_util import RodUtil
from solver.solver_params import SolverParams


class BendTwist(Energy):
    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, rod_state: RodState, init_rod_state: InitialRodState,
                       rod_params: RodParams):
        return 0.0  # Handled in Bend and Twist separately

    @staticmethod
    def d_energy_d_theta(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams):
        return grad  # Handled in Bend and Twist separately

    @staticmethod
    def d_energy_d_pos(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams):
        # TODO: vectorize
        n = theta.shape[0] - 1
        omega = RodUtil.compute_omega(theta=theta, curvature_binormal=rod_state.kb, bishop_frame=rod_state.bishop_frame)
        # Compute partial E / partial x_i, for all nodes (noting for our situation partial E / partial theta_i = 0)
        for i in range(n + 2):
            # k from 1 to n
            for k in range(1, n + 1):
                den = 1 / init_rod_state.l_bar[k]
                # j from k-1 to k
                for j_idx, j in enumerate([k - 1, k]):
                    B_j = rod_params.B[j]
                    omega_kj = omega[k, j_idx]
                    omega_bar_kj = init_rod_state.omega_bar[k, j_idx]
                    d_omega_kj = omega_kj - omega_bar_kj

                    # Compute nabla_i omega_kj. Requires m_1j, m_2j
                    theta_j = theta[j]
                    u, v = rod_state.bishop_frame[j]
                    m_1j = np.cos(theta_j) * u + np.sin(theta_j) * v
                    m_2j = -np.sin(theta_j) * u + np.cos(theta_j) * v
                    m_T = np.array([m_2j, -m_1j])
                    # Now compute nabla_i (kb)_k
                    nabla_i_kb_k = BendTwist.nabla_kb(pos=pos, i=k, d=i - k, init_rod_state=init_rod_state,
                                                     rod_state=rod_state)
                    # nabla_i psi_j
                    nabla_i_psi_j = BendTwist.nabla_i_psi_j(i, j, init_rod_state, rod_state)
                    nabla_i_omega_kj = m_T @ nabla_i_kb_k - Vector.J @ np.outer(omega_kj, nabla_i_psi_j)
                    grad[i] += den * nabla_i_omega_kj.T @ (B_j @ d_omega_kj)
        return grad

    @staticmethod
    def nabla_kb(pos: np.ndarray, i: int, d: int, init_rod_state: InitialRodState, rod_state: RodState):
        """
        Compute nabla_{i-1}(kb)_i, nabla_i(kb)_i, nabla_{i+1}(kb)_i
            represented by d [-1, 0, 1], respectively
        """
        # Only nabla_{i-1}, nabla_i, nabla_{i+1} are non-zero
        if d not in [-1, 0, 1]:
            return np.zeros((3, 3))
        # First compute (kb)_i, and denominator of the gradients
        e_i, e_im1 = (pos[i + 1] - pos[i]), (pos[i] - pos[i - 1])
        l_bar_i, l_bar_im1 = init_rod_state.l_bar_edge[i], init_rod_state.l_bar_edge[i - 1]
        kb_i = rod_state.kb[i]
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
    def nabla_di_psi_i(i: int, d: int, init_rod_state: InitialRodState, rod_state: RodState):
        """ Computes nabla_{i-1} psi_i, nabla_i psi_i, nabla_{i+1} psi_i """
        # Only nabla_{i-1}, nabla_i, nabla_{i+1} are non-zero
        if d not in [-1, 0, 1]:
            return 0.0

        # First compute the curvature binormal
        kb_i = rod_state.kb[i]

        # Gradients given by Eq. 9
        if d == 0:
            mag_ei_bar, mag_eim1_bar = init_rod_state.l_bar_edge[i], init_rod_state.l_bar_edge[i - 1]
            nabla_im1 = kb_i / (2 * mag_eim1_bar)
            nabla_ip1 = -kb_i / (2 * mag_ei_bar)
            nabla = -(nabla_im1 + nabla_ip1)
        elif d == -1:
            mag_eim1_bar = init_rod_state.l_bar_edge[i - 1]
            nabla = kb_i / (2 * mag_eim1_bar)
        else:
            mag_ei_bar = init_rod_state.l_bar_edge[i]
            nabla = -kb_i / (2 * mag_ei_bar)
        return nabla

    @staticmethod
    def nabla_i_psi_j(i: int, j: int, init_rod_state: InitialRodState, rod_state: RodState):
        """ Computes the gradient of Psi_j wrst i """
        grad = np.zeros(3)
        # From k = 1 to j
        for k in range(1, j + 1):
            grad += BendTwist.nabla_di_psi_i(k, i - k, init_rod_state, rod_state)
        return grad
