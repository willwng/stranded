import numpy as np
from scipy.sparse import diags, spmatrix

from rod.helix import Helix


class HelixUtil:

    @staticmethod
    def propagate(helix: Helix) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the centerline and the material frames from the generalized coordinates [q]

        n_i(s) = n_{i, L}^{Q ||} + n_{i, L}^{Q perp} cos(Omega(s - s_L^Q)) + omega \cross n_{i, L}^{Q perp} sin(Omega(s - s_L^Q))
        """
        # Centerline and material frames
        r = np.zeros((helix.n_sites, 3))
        n = np.zeros((helix.n_sites, 3, 3))

        # Starting with the clamped material frame, integrate forward
        r[0, :] = helix.r0
        n[0, :] = helix.n0
        for i in range(1, helix.n_sites):
            # Left hand side of interval (previous element)
            r_L = r[i - 1]
            n_L = n[i - 1]
            s, s_L = helix.s[i], helix.s[i - 1]
            s_sL = s - s_L
            # Twist and curvature
            tau, k_1, k_2 = helix.q[3 * i - 3:3 * i]
            # Darboux vector and unit vector aligned with the Darboux vector
            Omega = tau * n_L[0, :] + k_1 * n_L[1, :] + k_2 * n_L[2, :]
            Omega_norm = np.linalg.norm(Omega)

            # Degenerate case: straight line/no change in material frame
            if Omega_norm < 1e-12:
                n[i] = n_L
                r[i] = r_L + n_L[0] * s_sL
                continue

            w = Omega / Omega_norm

            # Projection of vector parallel to and perpendicular to w
            n_L_par = np.dot(n_L, w)[:, np.newaxis] * w
            n_L_perp = n_L - n_L_par

            # Compute the material frame
            n_i = n_L_par + n_L_perp * np.cos(Omega_norm * s_sL) + np.cross(w, n_L_perp) * np.sin(Omega_norm * s_sL)
            n[i] = n_i

            # Compute the centerline
            n_0_parallel = n_L_par[0]
            n_0_perp = n_L_perp[0]
            r_i = (r_L + n_0_parallel * s_sL + n_0_perp * np.sin(Omega_norm * s_sL) / Omega_norm +
                   np.cross(w, n_0_perp) * (1 - np.cos(Omega_norm * s_sL)) / Omega_norm)
            r[i] = r_i
        return r, n

    @staticmethod
    def compute_stiffness_matrix(helix: Helix) -> spmatrix:
        """ Computes the stiffness matrix for the helix """
        # Compute the length associated with each element
        l = helix.s[1:] - helix.s[:-1]
        K = diags(helix.EI[3:] * l.repeat(3))
        return K

    @staticmethod
    def compute_pointwise_stiffness_matrix(helix: Helix) -> spmatrix:
        """ Computes the stiffness matrix for the helix, where we want point-wise quantities
            versus integrated quantities (following DER paper) """
        l = helix.s[1:] - helix.s[:-1]
        K = diags(helix.EI[3:] * 2 * l.repeat(3))
        return K

    @staticmethod
    def compute_inv_stiffness_matrix(helix: Helix) -> spmatrix:
        """ Computes the inverse of the stiffness matrix for the helix """
        # Compute the length associated with each element
        l = helix.s[1:] - helix.s[:-1]
        K_inv = diags(1 / (helix.EI[3:] * l.repeat(3)))
        return K_inv

    @staticmethod
    def compute_inv_pointwise_stiffness_matrix(helix: Helix) -> spmatrix:
        """ Computes the inverse of the stiffness matrix for the helix, where we want point-wise quantities
            versus integrated quantities (following DER paper) """
        l = helix.s[1:] - helix.s[:-1]
        K_inv = diags(1 / (helix.EI[3:] * 2 * l.repeat(3)))
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
        for i in range(1, helix.n_sites):
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
    def compute_gen_potential_pos(helix: Helix, r: np.ndarray, g: float, rhoS: float, seed: int) -> float:
        """
        Naive implementation of the gravitational potential energy
        """
        # Compute the center of mass of each element
        r_com = 0.5 * (r[1:] + r[:-1])
        l = helix.s[1:] - helix.s[:-1]
        mass = rhoS * l
        return g * np.sum(mass * r_com[:, 2]) # + HelixUtil.compute_random_potential(r_com, seed=seed)

    @staticmethod
    def compute_gen_force(helix: Helix, g: float, rhoS: float, seed: int) -> np.ndarray:
        """
        Computes the generalized gravity force using numerical differentiation
        """
        grad = np.zeros(3 * (helix.n_sites - 1))
        eps = 1e-6

        q_free = helix.q.copy()[3:]
        for i in range(3 * (helix.n_sites - 1)):
            q_plus = q_free.copy()
            q_plus[i] += eps
            helix.q = np.concatenate([helix.q[:3], q_plus])
            r_plus, _ = HelixUtil.propagate(helix)
            U_g_plus = HelixUtil.compute_gen_potential_pos(helix, r_plus, g, rhoS, seed)

            q_minus = q_free.copy()
            q_minus[i] -= eps
            helix.q = np.concatenate([helix.q[:3], q_minus])
            r_minus, _ = HelixUtil.propagate(helix)
            U_g_minus = HelixUtil.compute_gen_potential_pos(helix, r_minus, g, rhoS, seed)
            # Finite difference
            grad[i] = (U_g_plus - U_g_minus) / (2 * eps)
        return -grad

    @staticmethod
    def compute_random_potential(r: np.ndarray, seed: int) -> float:
        """
        Computes a random potential (constant for given seed)
        """
        rng = np.random.RandomState(seed)
        # Each node is given a random potential
        mag = 1e-2
        potential_c = rng.rand(r.shape[0], 3) * (2 * mag) - mag
        return np.sum(potential_c * r)

    @staticmethod
    def compute_random_force(r: np.ndarray, seed: int) -> np.ndarray:
        """
        Computes a random, conservative force (by taking the gradient of the random potential)
        """
        rng = np.random.RandomState(seed)
        mag = 1e-2
        force_c = rng.rand(r.shape[0], 3) * (2 * mag) - mag
        return force_c

    @staticmethod
    def smoothen(helix: Helix, k: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Smoothen the helix by averaging the twist and curvature over a window of size k
        """
        q = helix.q.reshape(-1, 3)
        s = helix.s
        tau, kappa1, kappa2 = q[:, 0], q[:, 1], q[:, 2]
        tau_smooth, kappa1_smooth, kappa2_smooth = tau.copy(), kappa1.copy(), kappa2.copy()
        s_smooth = s.copy()
        for i in range(k, helix.n_sites - k):
            tau_smooth[i] = np.mean(tau[i - k:i + k])
            kappa1_smooth[i] = np.mean(kappa1[i - k:i + k])
            kappa2_smooth[i] = np.mean(kappa2[i - k:i + k])
            s_smooth[i] = np.mean(s[i - k:i + k])
        # End points average over smaller window
        for i in range(k):
            tau_smooth[i] = np.mean(tau[:2 * i + 1])
            kappa1_smooth[i] = np.mean(kappa1[:2 * i + 1])
            kappa2_smooth[i] = np.mean(kappa2[:2 * i + 1])
            s_smooth[i] = np.mean(s[:2 * i + 1])
        for i in range(helix.n_sites - k, helix.n_sites):
            tau_smooth[i] = np.mean(tau[2 * i - helix.n_sites + 1:])
            kappa1_smooth[i] = np.mean(kappa1[2 * i - helix.n_sites + 1:])
            kappa2_smooth[i] = np.mean(kappa2[2 * i - helix.n_sites + 1:])
            s_smooth[i] = np.mean(s[2 * i - helix.n_sites + 1:])
        tau_smooth[-1] = tau_smooth[-2]
        kappa1_smooth[-1] = kappa1_smooth[-2]
        kappa2_smooth[-1] = kappa2_smooth[-2]

        return np.stack([tau_smooth, kappa1_smooth, kappa2_smooth], axis=1).ravel(), s_smooth

    @staticmethod
    def create_smoothing_kernel(window_size):
        """Create a uniform smoothing kernel of given size"""
        return np.ones(window_size) / window_size

    @staticmethod
    def smooth_endpoints(arr, k):
        """Smooth endpoints with variable window sizes"""
        n = len(arr)
        # Front endpoints
        front_windows = [2 * i + 1 for i in range(k)]
        front_means = np.array([arr[:w].mean() for w in front_windows])

        # Back endpoints
        back_indices = range(n - k, n)
        back_windows = [arr[2 * i - n + 1:] for i in back_indices]
        back_means = np.array([w.mean() for w in back_windows])

        return front_means, back_means

    @staticmethod
    def smoothen2(helix: Helix, k: int):

        # Reshape and extract parameters
        q = helix.q.reshape(-1, 3)
        s = helix.s
        tau, kappa1, kappa2 = q[:, 0], q[:, 1], q[:, 2]

        # Initialize smoothed arrays
        tau_smooth = tau.copy()
        kappa1_smooth = kappa1.copy()
        kappa2_smooth = kappa2.copy()
        s_smooth = s.copy()

        # Create kernel for middle section
        kernel = HelixUtil.create_smoothing_kernel(2 * k + 1)

        # Smooth middle sections using convolution
        for arr_smooth, arr in [(tau_smooth, tau),
                                (kappa1_smooth, kappa1),
                                (kappa2_smooth, kappa2),
                                (s_smooth, s)]:
            # Smooth middle section
            middle_smooth = np.convolve(arr, kernel, mode='valid')
            arr_smooth[k:-k] = middle_smooth

            # Smooth endpoints
            front_means, back_means = HelixUtil.smooth_endpoints(arr, k)
            arr_smooth[:k] = front_means
            arr_smooth[-k:] = back_means

            # Copy second-to-last value to last position
            arr_smooth[-1] = arr_smooth[-2]

        return np.stack([tau_smooth, kappa1_smooth, kappa2_smooth], axis=1).ravel(), s_smooth

    @staticmethod
    def increase_resolution(helix: Helix) -> Helix:
        """
        Increase the resolution of the helix by linear interpolation
        """
        n_sites = helix.n_sites
        n_new_sites = 2 * n_sites - 1
        q = helix.q.reshape(-1, 3)

        tau, kappa1, kappa2 = q[:, 0], q[:, 1], q[:, 2]
        tau_inc, kappa1_inc, kappa2_inc = np.zeros(n_new_sites), np.zeros(n_new_sites), np.zeros(n_new_sites)
        s_new = np.zeros(n_new_sites)
        for i in range(n_new_sites):
            tau_inc[i] = tau[i // 2]
            kappa1_inc[i] = kappa1[i // 2]
            kappa2_inc[i] = kappa2[i // 2]
            if i % 2 == 0:
                s_new[i] = helix.s[i // 2]
            else:
                s_new[i] = 0.5 * (helix.s[i // 2] + helix.s[i // 2 + 1])

        q_increased = np.stack([tau_inc, kappa1_inc, kappa2_inc], axis=1).ravel()
        # Repeat
        EI = np.zeros(3 * n_new_sites)
        EI[::2] = helix.EI[:-1]
        EI[1::2] = helix.EI[1:-1]

        return Helix(r0=helix.r0, n0=helix.n0, q=q_increased, EI=EI, s=s_new, L=helix.L,
                     n_sites=n_new_sites, q0=q_increased.copy())
