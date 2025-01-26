import numpy as np

from energies.energy import Energy
from math_util.vectors import Vector
from rod.rod import RodState, InitialRodState, RodParams
from rod.rod_util import RodUtil


class Bend(Energy):
    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, rod_state: RodState, init_rod_state: InitialRodState,
                       rod_params: RodParams):
        l_bar = init_rod_state.l_bar
        omega_bar = init_rod_state.omega_bar
        omega = RodUtil.compute_omega(theta=theta, curvature_binormal=rod_state.kb, bishop_frame=rod_state.bishop_frame)
        d_omega = omega - omega_bar

        # Compute d_omega @ B @ d_omega across all edges
        B_d_omega = rod_params.B @ d_omega
        d_B_d_omega = np.einsum('ijk,ijk->i', B_d_omega, d_omega)

        # Sum, scaling by edge lengths
        energy = 0.5 * np.sum((1 / l_bar[1:]) * d_B_d_omega[1:])
        return energy

    @staticmethod
    def d_energy_d_theta(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams):
        # TODO: vectorize
        # All edges
        n = theta.shape[0] - 1
        omega = RodUtil.compute_omega(theta=theta, curvature_binormal=rod_state.kb, bishop_frame=rod_state.bishop_frame)
        for j in range(0, n + 1):
            l_j = init_rod_state.l_bar[j]
            B_j = rod_params.B[j]

            # First term
            if j != 0:
                omega_jj = omega[j, 1]
                omega_bar_jj = init_rod_state.omega_bar[j, 1]
                grad[j] += (1 / l_j) * np.dot(omega_jj, Vector.J @ (B_j @ (omega_jj - omega_bar_jj)))

            # Second term, only if not last edge
            if j != n:
                l_jp1 = init_rod_state.l_bar[j + 1]
                omega_jp1j = omega[j + 1, 0]
                omega_bar_jp1j = init_rod_state.omega_bar[j + 1, 0]
                grad[j] += (1 / l_jp1) * np.dot(omega_jp1j, Vector.J @ (B_j @ (omega_jp1j - omega_bar_jp1j)))
        return grad

    @staticmethod
    def d_energy_d_pos(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams):
        return grad  # Handled in BendTwist
