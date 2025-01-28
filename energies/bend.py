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
        omega = RodUtil.compute_omega(theta=theta, kb=rod_state.kb, bishop_frame=rod_state.bishop_frame)
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
        l_bar = init_rod_state.l_bar
        omega_bar = init_rod_state.omega_bar

        omega = RodUtil.compute_omega(theta=theta, kb=rod_state.kb, bishop_frame=rod_state.bishop_frame)
        d_omega = omega - omega_bar

        # All edges except the first
        omega1 = omega[:, 1, :]
        d_omega1 = d_omega[:, 1, :]
        B_d_omega1 = np.einsum('ijk,ik->ij', rod_params.B, d_omega1)
        J_B_d_omega1 = np.dot(B_d_omega1, Vector.J.T)
        grad[1:] += (1 / l_bar[1:]) * Vector.inner_products(J_B_d_omega1, omega1)[1:]

        # All edges except the last
        omega0 = omega[:, 0, :]
        d_omega0 = d_omega[:, 0, :]
        B_d_omega0 = np.einsum('ijk,ik->ij', rod_params.B, d_omega0)
        J_B_d_omega0 = np.dot(B_d_omega0, Vector.J.T)
        grad[:-1] += (1 / l_bar[1:]) * Vector.inner_products(J_B_d_omega0, omega0)[1:]
        return grad

    @staticmethod
    def d_energy_d_pos(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams):
        return grad  # Handled in BendTwist
