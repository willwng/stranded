import cupy as cp

from energies.energy import Energy
from math_util.vectors import Vector
from rod.rod import RodState, InitialRodState, RodParams
from rod.rod_util import RodUtil


class Bend(Energy):
    @staticmethod
    def compute_energy(pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState, init_rod_state: InitialRodState,
                       rod_params: RodParams):
        l_bar = init_rod_state.l_bar
        omega_bar = init_rod_state.omega_bar
        omega = RodUtil.compute_omega(theta=theta, kb=rod_state.kb, bishop_frame=rod_state.bishop_frame)
        d_omega = omega - omega_bar

        B_d_omega = rod_params.B @ d_omega  # (n x 2 x 2) @ (n x 2 x 2)
        d_B_d_omega = cp.einsum('ijk,ijk->i', B_d_omega, d_omega)

        energy = 0.5 * cp.sum((1 / l_bar[1:]) * d_B_d_omega[1:])
        return energy

    @staticmethod
    def d_energy_d_theta(grad: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams):
        l_bar = init_rod_state.l_bar
        omega_bar = init_rod_state.omega_bar

        omega = RodUtil.compute_omega(theta=theta, kb=rod_state.kb, bishop_frame=rod_state.bishop_frame)
        d_omega = omega - omega_bar

        # Edge-wise terms
        omega1 = omega[:, 1, :]
        d_omega1 = d_omega[:, 1, :]
        B_d_omega1 = cp.einsum('ijk,ik->ij', rod_params.B, d_omega1)
        J_B_d_omega1 = B_d_omega1 @ Vector.J.T
        grad[1:] += (1 / l_bar[1:]) * Vector.inner_products(J_B_d_omega1, omega1)[1:]

        omega0 = omega[:, 0, :]
        d_omega0 = d_omega[:, 0, :]
        B_d_omega0 = cp.einsum('ijk,ik->ij', rod_params.B, d_omega0)
        J_B_d_omega0 = B_d_omega0 @ Vector.J.T
        grad[:-1] += (1 / l_bar[1:]) * Vector.inner_products(J_B_d_omega0, omega0)[1:]

        return grad

    @staticmethod
    def d2_energy_d_theta2(hess: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                           init_rod_state: InitialRodState, rod_params: RodParams):
        raise NotImplementedError

    @staticmethod
    def d_energy_d_pos(grad: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams):
        return grad  # gradient wrt pos is handled in BendTwist

