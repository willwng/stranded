"""
Hacky, only used for gradient calculation wrst position
"""
import numpy as np
from numba import njit

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
        n = theta.shape[0] - 1
        omega = RodUtil.compute_omega(theta=theta, kb=rod_state.kb, bishop_frame=rod_state.bishop_frame)
        # Compute partial E / partial x_i, for all nodes (noting for our situation partial E / partial theta_i = 0)
        for i in range(n + 2):
            # k from 1 to n, but nabla_i_kb_k is only nonzero for k = i-1, i, i+1
            k_ind_nz = np.arange(max(1, i - 1), min(n + 1, i + 2))
            # Furthermore nabla_i_psi_j will be zero when j < i-2
            k_ind_zero = np.setdiff1d(np.arange(i + 2, n + 1), k_ind_nz)
            for k in k_ind_nz:
                den = 1 / init_rod_state.l_bar[k]
                # j from k-1 to k
                for j_idx, j in enumerate([k - 1, k]):
                    B_j = rod_params.B[j]
                    omega_kj = omega[k, j_idx]
                    omega_bar_kj = init_rod_state.omega_bar[k, j_idx]
                    d_omega_kj = omega_kj - omega_bar_kj

                    # Compute nabla_i omega_kj. Requires m_1j, m_2j
                    m_1j, m_2j = rod_state.material_frame[j]
                    m_T = np.array([m_2j, -m_1j])

                    # Now compute nabla_i (kb)_k
                    nabla_i_kb_k = rod_state.nabla_kb[k, i - k + 1]

                    # nabla_i psi_j
                    nabla_i_psi_j = BendTwist.nabla_i_psi_j(i, j, rod_state)
                    nabla_i_omega_kj = m_T @ nabla_i_kb_k - Vector.J @ np.outer(omega_kj, nabla_i_psi_j)
                    grad[i] += den * nabla_i_omega_kj.T @ (B_j @ d_omega_kj)

            # Now for the zero terms
            for k in k_ind_zero:
                den = 1 / init_rod_state.l_bar[k]
                # j from k-1 to k
                for j_idx, j in enumerate([k - 1, k]):
                    B_j = rod_params.B[j]
                    omega_kj = omega[k, j_idx]
                    omega_bar_kj = init_rod_state.omega_bar[k, j_idx]
                    d_omega_kj = omega_kj - omega_bar_kj
                    # nabla_i psi_j
                    nabla_i_psi_j = BendTwist.nabla_i_psi_j(i, j, rod_state)
                    nabla_i_omega_kj = -Vector.J @ np.outer(omega_kj, nabla_i_psi_j)
                    grad[i] += den * nabla_i_omega_kj.T @ (B_j @ d_omega_kj)

        return grad

    @staticmethod
    def nabla_i_psi_j(i: int, j: int, rod_state: RodState):
        """ Computes the gradient of Psi_j wrst i """
        # From m = 1 to j. Note: only nabla_{i-1}, nabla_i, nabla_{i+1} are non-zero
        m = np.arange(max(1, i - 1), min(j + 1, i + 2))
        return np.sum(rod_state.nabla_psi[m, i - m + 1], axis=0)
