import cupy as cp

from energies.energy import Energy
from math_util.vectors import Vector
from rod.rod import RodState, InitialRodState, RodParams
from rod.rod_util import RodUtil


class BendTwist(Energy):
    @staticmethod
    def compute_energy(pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState, init_rod_state: InitialRodState,
                       rod_params: RodParams):
        return 0.0  # handled in Bend and Twist separately

    @staticmethod
    def d_energy_d_theta(grad: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams):
        return grad  # handled in Bend and Twist separately

    @staticmethod
    def d_energy_d_pos(grad: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams):
        n = theta.shape[0] - 1
        omega = RodUtil.compute_omega(theta=theta, kb=rod_state.kb, bishop_frame=rod_state.bishop_frame)
        d_omega = omega - init_rod_state.omega_bar

        den = cp.concatenate([cp.zeros(1), 1 / init_rod_state.l_bar[1:]])

        B_d_omega = cp.zeros((n, 2, 2))
        B_d_omega[:, 0] = Vector.matrix_multiply(rod_params.B[:-1], d_omega[1:, 0])
        B_d_omega[:, 1] = Vector.matrix_multiply(rod_params.B[1:], d_omega[1:, 1])
        B_d_omega = cp.concatenate([cp.zeros((1, 2, 2)), B_d_omega])

        J_omega = Vector.single_matrix_multiply(Vector.J, omega)

        m_1, m_2 = rod_state.material_frame[:, 0], rod_state.material_frame[:, 1]
        m_T = cp.stack([m_2, -m_1], axis=-2)

        j_ind = cp.stack([cp.arange(0, n), cp.arange(1, n + 1)], axis=-1)

        for i in range(n + 2):
            k_ind_nz = cp.arange(max(1, i - 1), min(n + 1, i + 2))
            for k in k_ind_nz:
                nabla_i_kb_k = rod_state.nabla_kb[k, i - k + 1]
                j_ind_nz = j_ind[k - 1]
                for j_idx, j in enumerate(j_ind_nz):
                    m = cp.arange(max(1, i - 1), min(j + 1, i + 2))
                    nabla_i_psi_j = cp.sum(rod_state.nabla_psi[m, i - m + 1], axis=0)
                    nabla_i_omega_kj = m_T[j] @ nabla_i_kb_k - cp.outer(J_omega[k, j_idx], nabla_i_psi_j)
                    grad[i] += den[k] * (nabla_i_omega_kj.T @ B_d_omega[k, j_idx])

            k_ind_zero = cp.setdiff1d(cp.arange(i + 2, n + 1), k_ind_nz)
            m = cp.arange(max(1, i - 1), min(i + 2, n + 1))
            nabla_i_psi_j = cp.sum(rod_state.nabla_psi[m, i - m + 1], axis=0)

            J_omega_nabla_i_psi_j = Vector.outer_product_helper(J_omega, nabla_i_psi_j)
            J_omega_nabla_B_d_omega = cp.einsum('nijk,nij->nik', J_omega_nabla_i_psi_j, B_d_omega)
            grads_nz = -den[k_ind_zero, None, None] * J_omega_nabla_B_d_omega[k_ind_zero]
            grads_nz = cp.sum(grads_nz, axis=(0, 1))
            grad[i] += grads_nz

        return grad

