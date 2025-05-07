import cupy as cp

from math_util.rotation import Quaternion
from math_util.vectors import Vector


class RodUtil:
    @staticmethod
    def compute_edge_lengths(pos: cp.ndarray) -> cp.ndarray:
        return cp.linalg.norm(pos[1:] - pos[:-1], axis=1)

    @staticmethod
    def compute_node_lengths(edge_lengths: cp.ndarray) -> cp.ndarray:
        return cp.concatenate([cp.zeros(1), (edge_lengths[1:] + edge_lengths[:-1]) / 2])

    @staticmethod
    def compute_curvature_binormal(pos: cp.ndarray, rest_edge_lengths: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
        e = pos[1:] - pos[:-1]
        e_i, e_im1 = e[1:], e[:-1]
        kb = cp.cross(2 * e_im1, e_i)
        den = (rest_edge_lengths[1:] * rest_edge_lengths[:-1] + Vector.inner_products(e_im1, e_i))
        kb /= den[:, None]
        return cp.concatenate([cp.zeros((1, 3)), kb]), den

    @staticmethod
    def compute_material_frames(theta: cp.ndarray, bishop_frame: cp.ndarray):
        u, v = bishop_frame[:, 0, :], bishop_frame[:, 1, :]
        cos_theta, sin_theta = cp.cos(theta)[:, None], cp.sin(theta)[:, None]
        m_1 = cos_theta * u + sin_theta * v
        m_2 = -sin_theta * u + cos_theta * v
        return cp.stack([m_1, m_2], axis=1)

    @staticmethod
    def compute_omega(theta: cp.ndarray, kb: cp.ndarray, bishop_frame: cp.ndarray) -> cp.ndarray:
        material_frames = RodUtil.compute_material_frames(theta=theta, bishop_frame=bishop_frame)

        m_1, m_2 = material_frames[:, 0, :], material_frames[:, 1, :]
        omega_i1 = Vector.inner_products(kb, m_2)
        omega_i2 = -Vector.inner_products(kb, m_1)

        m_1m1, m_2m1 = material_frames[:-1, 0, :], material_frames[:-1, 1, :]
        omega_im1_1 = Vector.inner_products(kb[1:], m_2m1)
        omega_im1_2 = -Vector.inner_products(kb[1:], m_1m1)

        omega_i = cp.column_stack([omega_i1, omega_i2])
        omega_im1 = cp.column_stack([omega_im1_1, omega_im1_2])
        omega_im1 = cp.concatenate([cp.zeros((1, 2)), omega_im1])

        omegas = cp.zeros((omega_i.shape[0], 2, 2))
        omegas[:, 0, :] = omega_im1
        omegas[:, 1, :] = omega_i
        return omegas

    @staticmethod
    def update_bishop_frames(pos: cp.ndarray, bishop_frame: cp.ndarray) -> cp.ndarray:
        t0 = pos[1] - pos[0]
        t0 /= cp.linalg.norm(t0)
        u = Vector.compute_orthogonal_vec(t0)
        v = cp.cross(t0, u)
        u, v = u / cp.linalg.norm(u), v / cp.linalg.norm(v)
        bishop_frame[0] = cp.stack([u, v])

        n = bishop_frame.shape[0]
        for i in range(1, n):
            t_i = pos[i + 1] - pos[i]
            t_im1 = pos[i] - pos[i - 1]
            t_i /= cp.linalg.norm(t_i)
            t_im1 /= cp.linalg.norm(t_im1)

            if cp.dot(t_im1, t_i) > 1 - 1e-6:
                P_i = Quaternion.identity()
            else:
                rot_axis = cp.cross(t_im1, t_i)
                rot_axis /= cp.linalg.norm(rot_axis)
                rot_angle = cp.arccos(cp.clip(cp.dot(t_im1, t_i), -1.0, 1.0))
                P_i = Quaternion.from_angle_axis(rot_angle, rot_axis)
                P_i.normalize()

            u = P_i.rotate_vec(u)
            v = cp.cross(t_i, u)
            u, v = u / cp.linalg.norm(u), v / cp.linalg.norm(v)
            bishop_frame[i] = cp.stack([u, v])
        return bishop_frame

    @staticmethod
    def compute_nabla_kb(pos: cp.ndarray, kb: cp.ndarray, kb_den: cp.ndarray) -> cp.ndarray:
        e = pos[1:] - pos[:-1]
        e_skew_sym = Vector.skew_sym(e)
        kb_i = kb[1:]

        kb_e_T = Vector.outer_products(kb_i, e[1:])
        num_im1 = 2 * e_skew_sym[1:] + kb_e_T

        kb_em1_T = Vector.outer_products(kb_i, e[:-1])
        num_ip1 = 2 * e_skew_sym[:-1] - kb_em1_T

        nabla_im1 = num_im1 / kb_den[:, None, None]
        nabla_ip1 = num_ip1 / kb_den[:, None, None]
        nabla_i = -(nabla_im1 + nabla_ip1)

        nabla_kb = cp.stack([nabla_im1, nabla_i, nabla_ip1], axis=1)
        nabla_kb = cp.concatenate([cp.zeros((1, 3, 3, 3)), nabla_kb])
        return nabla_kb

    @staticmethod
    def compute_nabla_psi(kb: cp.ndarray, rest_edge_lengths: cp.ndarray) -> cp.ndarray:
        nabla_im1 = kb[1:] / (2 * rest_edge_lengths[:-1, None])
        nabla_ip1 = -kb[1:] / (2 * rest_edge_lengths[1:, None])
        nabla_i = -(nabla_im1 + nabla_ip1)
        nabla_psi = cp.stack([nabla_im1, nabla_i, nabla_ip1], axis=1)
        nabla_psi = cp.concatenate([cp.zeros((1, 3, 3)), nabla_psi])
        return nabla_psi
