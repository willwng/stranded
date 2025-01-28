"""
Hacky, only used for gradient calculation wrst position
"""
import numpy as np

from energies.energy import Energy
from math_util.vectors import Vector
from rod.rod import RodState, InitialRodState, RodParams
from rod.rod_util import RodUtil


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
        # Compute all omega and omega-omega_bar
        omega = RodUtil.compute_omega(theta=theta, kb=rod_state.kb, bishop_frame=rod_state.bishop_frame)
        d_omega = omega - init_rod_state.omega_bar
        # Denominator is 1/\bar{l_i}
        den = np.concatenate([np.zeros(1), 1 / init_rod_state.l_bar[1:]])

        # Precompute all B @ d_omega. B_d_omega[i, 0] = B_{i-1} @ d_omega_{i-1}, B_d_omega[i, 1] = B_i @ d_omega_i
        B_d_omega = np.zeros((n, 2, 2))
        B_d_omega[:, 0] = Vector.matrix_multiply(rod_params.B[:-1], d_omega[1:, 0])
        B_d_omega[:, 1] = Vector.matrix_multiply(rod_params.B[1:], d_omega[1:, 1])
        B_d_omega = np.concatenate([np.zeros((1, 2, 2)), B_d_omega])

        # Precompute all J @ omega
        J_omega = Vector.single_matrix_multiply(Vector.J, omega)

        # Create m_T matrix
        m_1, m_2 = rod_state.material_frame[:, 0], rod_state.material_frame[:, 1]
        m_T = np.stack([m_2, -m_1], axis=-2)

        # Compute partial E / partial x_i, for all nodes (noting for our situation partial E / partial theta_i = 0)
        for i in range(n + 2):
            # k from 1 to n, but nabla_i_kb_k is only nonzero for k = i-1, i, i+1
            k_ind_nz = np.arange(max(1, i - 1), min(n + 1, i + 2))
            for k in k_ind_nz:
                # j from k-1 to k
                for j_idx, j in enumerate([k - 1, k]):
                    nabla_i_kb_k = rod_state.nabla_kb[k, i - k + 1]

                    # nabla_i psi_j
                    m = np.arange(max(1, i - 1), min(j + 1, i + 2))
                    nabla_i_psi_j = np.sum(rod_state.nabla_psi[m, i - m + 1], axis=0)
                    nabla_i_omega_kj = m_T[j] @ nabla_i_kb_k - np.outer(J_omega[k, j_idx], nabla_i_psi_j)
                    grad[i] += den[k] * nabla_i_omega_kj.T @ (B_d_omega[k, j_idx])

            # Terms where nabla_i_kb_k is zero. Also, note nabla_i_psi_j will be zero when j < i-2
            k_ind_zero = np.setdiff1d(np.arange(i + 2, n + 1), k_ind_nz)
            # Since we know j >= i+1, we can move this expression out here
            m = np.arange(max(1, i - 1), min(i + 2, n + 1))
            nabla_i_psi_j = np.sum(rod_state.nabla_psi[m, i - m + 1], axis=0)

            # Compute J_omega @ nabla_i_psi_j.T
            J_omega_nabla_i_psi_j = Vector.outer_product_helper(J_omega, nabla_i_psi_j)
            # Compute (J_omega @ nabla_i_psi_j).T @ B_d_omega
            J_omega_nabla_B_d_omega = np.einsum('nijk,nij->nik', J_omega_nabla_i_psi_j, B_d_omega)
            # Compute den[k] * -J_omega_nabla_B_d_omega
            grads_nz = -den[k_ind_zero, None, None] * J_omega_nabla_B_d_omega[k_ind_zero]
            # Update gradient
            grads_nz = np.sum(grads_nz, axis=(0, 1))
            grad[i] += grads_nz

        return grad
