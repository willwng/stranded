import numpy as np

from rod.helix import Helix


class HelixUtil:

    @staticmethod
    def propagate(helix: Helix) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the centerline and the material frames from the generalized coordinates [q]

        n_i(s) = n_{i, L}^{Q ||} + n_{i, L}^{Q perp} cos(Omega(s - s_L^Q)) + omega \cross n_{i, L}^{Q perp} sin(Omega(s - s_L^Q))
        """
        # Centerline and material frames
        r = np.zeros((helix.n_elems, 3))
        n = np.zeros((helix.n_elems, 3, 3))

        # Starting with the clamped material frame, integrate forward
        r[0, :] = helix.r0
        n[0, :] = helix.n0
        for i in range(1, helix.n_elems):
            # Left hand side of interval (previous element)
            r_L = r[i - 1]
            n_L = n[i - 1]
            s, s_L = helix.s[i], helix.s[i - 1]
            s_sL = s - s_L
            # Twist and curvature
            tau, k_1, k_2 = helix.q[3 * i - 3:3 * i]
            # Darboux vector and unit vector aligned with the Darboux vector
            omega = tau * n_L[0, :] + k_1 * n_L[1, :] + k_2 * n_L[2, :]
            omega_norm = np.linalg.norm(omega)
            w = omega / omega_norm

            # Projection of vector parallel to and perpendicular to w
            n_L_par = np.dot(n_L, w)[:, np.newaxis] * w
            n_L_perp = n_L - n_L_par

            # Compute the material frame
            n_i = n_L_par + n_L_perp * np.cos(omega_norm * s_sL) + np.cross(w, n_L_perp) * np.sin(omega_norm * s_sL)
            n[i] = n_i

            # Compute the centerline
            n_0_L = n_L[0]
            n_0_parallel = np.dot(n_0_L, w) * w
            n_0_perp = n_0_L - n_0_parallel
            r_i = (r_L + n_0_parallel * s_sL + n_0_perp * np.sin(omega_norm * s_sL) / omega_norm +
                   np.cross(w, n_0_perp) * (1 - np.cos(omega_norm * s_sL)) / omega_norm)
            r[i] = r_i
        return r, n

    @staticmethod
    def compute_stiffness_matrix(helix: Helix) -> np.ndarray:
        """ Computes the stiffness matrix for the helix """
        # Compute the length associated with each element
        l = helix.s[1:] - helix.s[:-1]
        K = np.diag(helix.EI[3:] * l.repeat(3))
        return K

    @staticmethod
    def compute_inv_stiffness_matrix(helix: Helix) -> np.ndarray:
        """ Computes the inverse of the stiffness matrix for the helix """
        # Compute the length associated with each element
        l = helix.s[1:] - helix.s[:-1]
        K_inv = np.diag(1 / (helix.EI[3:] * l.repeat(3)))
        return K_inv

    @staticmethod
    def compute_mass_matrix(helix: Helix) -> np.ndarray:
        """
        Computes the mass matrix for the helix
        M_{ij} = \rho S \int_0^L (\partial r / \partial q_i)^T (\partial r / \partial q_j) ds
        """

    @staticmethod
    def compute_internal_potential_loop(helix: Helix) -> float:
        """
        Computes the internal potential energy of the helix using a loop
        U_in(q) = 0.5 \int_0^L \sum_{i=0}^{2} (EI)_i (k_i(s) - k_i^0)^2 ds
        """
        U_in = 0.0
        for i in range(1, helix.n_elems):
            # Stiffness, differential arc length, and twist and curvature
            ds = helix.s[i] - helix.s[i - 1]
            EI_i = helix.EI[3 * i:3 * i + 3]
            tau, k_1, k_2 = helix.q[3 * i:3 * i + 3]
            tau_0, k_1_0, k_2_0 = helix.q0[3 * i:3 * i + 3]

            # Compute the change in twist and curvature
            dk = np.array([tau - tau_0, k_1 - k_1_0, k_2 - k_2_0])
            U_in += 0.5 * np.dot(EI_i, dk ** 2) * ds
        return U_in

    @staticmethod
    def compute_internal_potential(helix: Helix) -> float:
        """
        Computes the internal potential energy of the helix using matrix operations
        U_in(q) = 0.5 \int_0^L \sum_{i=0}^{2} (EI)_i (k_i(s) - k_i^0)^2 ds
                = 0.5 (q - q0)^T K (q - q0)
        """
        K = HelixUtil.compute_stiffness_matrix(helix)
        q_min_q0 = (helix.q - helix.q0)[3:]
        return float(0.5 * q_min_q0.T @ (K @ q_min_q0))

    @staticmethod
    def compute_gravity_potential_pos(helix: Helix, r: np.ndarray, g: float, rhoS: float) -> float:
        """
        Naive implementation of the gravitational potential energy
        """
        # Compute the center of mass of each element
        r_com = 0.5 * (r[1:] + r[:-1])
        l = helix.s[1:] - helix.s[:-1]
        mass = rhoS * l
        return g * np.sum(mass * r_com[:, 2])

    @staticmethod
    def compute_gen_gravity_force(helix: Helix, g: float, rhoS: float) -> np.ndarray:
        """
        Computes the generalized gravity force using numerical differentiation
        """
        grad = np.zeros(3 * (helix.n_elems - 1))
        eps = 1e-6

        q_free = helix.q.copy()[3:]
        for i in range(3 * (helix.n_elems - 1)):
            q_plus = q_free.copy()
            q_plus[i] += eps
            helix.q = np.concatenate([helix.q[:3], q_plus])
            r_plus, _ = HelixUtil.propagate(helix)
            U_g_plus = HelixUtil.compute_gravity_potential_pos(helix, r_plus, g, rhoS)

            q_minus = q_free.copy()
            q_minus[i] -= eps
            helix.q = np.concatenate([helix.q[:3], q_minus])
            r_minus, _ = HelixUtil.propagate(helix)
            U_g_minus = HelixUtil.compute_gravity_potential_pos(helix, r_minus, g, rhoS)
            # Finite difference
            grad[i] = (U_g_plus - U_g_minus) / (2 * eps)
        return -grad
