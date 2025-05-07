import cupy as cp
from scipy.sparse import diags, spmatrix

from rod.helix import Helix


class HelixUtil:
    @staticmethod
    def propagate_q(q, r0, n0, n_sites, s, r, n):
        r[0, :] = r0
        n[0, :] = n0
        for i in range(1, n_sites):
            r_L = r[i - 1]
            n_L = n[i - 1]
            s_R, s_L = s[i], s[i - 1]
            s_sL = s_R - s_L
            tau, k_1, k_2 = q[3 * i - 3:3 * i]
            Omega = tau * n_L[0, :] + k_1 * n_L[1, :] + k_2 * n_L[2, :]
            Omega_norm = cp.linalg.norm(Omega)

            if Omega_norm < 1e-12:
                n[i] = n_L
                r[i] = r_L + n_L[0] * s_sL
                continue

            w = Omega / Omega_norm
            n_L_par = cp.dot(n_L, w)[:, cp.newaxis] * w
            n_L_perp = n_L - n_L_par

            n_i = n_L_par + n_L_perp * cp.cos(Omega_norm * s_sL) + cp.cross(w, n_L_perp) * cp.sin(Omega_norm * s_sL)
            n[i] = n_i

            n_0_parallel = n_L_par[0]
            n_0_perp = n_L_perp[0]
            r_i = (r_L + n_0_parallel * s_sL + n_0_perp * cp.sin(Omega_norm * s_sL) / Omega_norm +
                   cp.cross(w, n_0_perp) * (1 - cp.cos(Omega_norm * s_sL)) / Omega_norm)
            r[i] = r_i
        return r, n

    @staticmethod
    def propagate(helix: Helix):
        r = cp.zeros((helix.n_sites, 3))
        n = cp.zeros((helix.n_sites, 3, 3))
        HelixUtil.propagate_q(helix.q, helix.r0, helix.n0, helix.n_sites, helix.s, r, n)
        return r, n

    @staticmethod
    def compute_stiffness_matrix(helix: Helix) -> spmatrix:
        l = cp.asnumpy(helix.s[1:] - helix.s[:-1])
        return diags(cp.asnumpy(helix.EI[3:] * cp.repeat(l, 3)))

    @staticmethod
    def compute_pointwise_stiffness_matrix(helix: Helix) -> spmatrix:
        l = cp.asnumpy(helix.s[1:] - helix.s[:-1])
        return diags(cp.asnumpy(helix.EI[3:] * 2 * cp.repeat(l, 3)))

    @staticmethod
    def compute_inv_stiffness_matrix(helix: Helix) -> spmatrix:
        l = cp.asnumpy(helix.s[1:] - helix.s[:-1])
        return diags(cp.asnumpy(1 / (helix.EI[3:] * cp.repeat(l, 3))))

    @staticmethod
    def compute_inv_pointwise_stiffness_matrix(helix: Helix) -> spmatrix:
        l = cp.asnumpy(helix.s[1:] - helix.s[:-1])
        return diags(cp.asnumpy(1 / (helix.EI[3:] * 2 * cp.repeat(l, 3))))

    @staticmethod
    def compute_internal_potential_loop(helix: Helix):
        U_in = 0.0
        for i in range(1, helix.n_sites):
            ds = helix.s[i] - helix.s[i - 1]
            EI_i = helix.EI[3 * i:3 * i + 3]
            tau, k_1, k_2 = helix.q[3 * i:3 * i + 3]
            tau_0, k_1_0, k_2_0 = helix.q0[3 * i:3 * i + 3]
            dk = cp.array([tau - tau_0, k_1 - k_1_0, k_2 - k_2_0])
            U_in += 0.5 * cp.dot(EI_i, dk ** 2) * ds
        return float(U_in)

    @staticmethod
    def compute_internal_potential(helix: Helix):
        K = HelixUtil.compute_stiffness_matrix(helix)
        q_diff = cp.asnumpy((helix.q - helix.q0)[3:])
        return float(0.5 * q_diff.T @ (K @ q_diff))

    @staticmethod
    def compute_gen_potential_pos(helix: Helix, r, g: float, rhoS: float, seed: int):
        r_com = 0.5 * (r[1:] + r[:-1])
        l = helix.s[1:] - helix.s[:-1]
        mass = rhoS * l
        return float(g * cp.sum(mass * r_com[:, 2]))

    @staticmethod
    def compute_gen_force(helix: Helix, g: float, rhoS: float, seed: int):
        grad = cp.zeros(3 * (helix.n_sites - 1))
        eps = 1e-6

        q_free = helix.q.copy()[3:]
        for i in range(grad.shape[0]):
            q_plus = q_free.copy()
            q_plus[i] += eps
            helix.q = cp.concatenate([helix.q[:3], q_plus])
            r_plus, _ = HelixUtil.propagate(helix)
            U_plus = HelixUtil.compute_gen_potential_pos(helix, r_plus, g, rhoS, seed)

            q_minus = q_free.copy()
            q_minus[i] -= eps
            helix.q = cp.concatenate([helix.q[:3], q_minus])
            r_minus, _ = HelixUtil.propagate(helix)
            U_minus = HelixUtil.compute_gen_potential_pos(helix, r_minus, g, rhoS, seed)

            grad[i] = (U_plus - U_minus) / (2 * eps)

        return -grad
