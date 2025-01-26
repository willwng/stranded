import numpy as np

from math_util.quaternion import Quaternion
from math_util.vectors import Vector


class RodUtil:
    @staticmethod
    def compute_edge_lengths(pos: np.ndarray) -> np.ndarray:
        """
        Computes the length of every edge in the rod
        """
        return np.linalg.norm(pos[1:] - pos[:-1], axis=1)

    @staticmethod
    def compute_node_lengths(edge_lengths: np.ndarray) -> np.ndarray:
        """
        l_i = ( |e_i| + |e_{i-1}| ) / 2. 0 if i = 0
        """
        return np.concatenate([np.zeros(1), (edge_lengths[1:] + edge_lengths[:-1]) / 2])

    @staticmethod
    def compute_curvature_binormal(pos: np.ndarray, rest_edge_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        (kb)_i = (2 * e_{i-1} x e_i) / (|\bar e_{i-1}| |\bar e_i| + e_{i-1} . e_i)
            also returns the denominator
        """
        e = pos[1:] - pos[:-1]
        e_i, e_im1 = e[1:], e[:-1]
        kb = np.cross(2 * e_im1, e_i)
        den = (rest_edge_lengths[1:] * rest_edge_lengths[:-1] + Vector.inner_products(e_im1, e_i))
        kb /= den[:, None]
        return np.concatenate([np.zeros((1, 3)), kb]), den

    @staticmethod
    def compute_material_frames(theta: np.ndarray, bishop_frame: np.ndarray):
        """ Computes the material frame for each edge """
        u, v = bishop_frame[:, 0, :], bishop_frame[:, 1, :]
        cos_theta, sin_theta = np.cos(theta)[:, None], np.sin(theta)[:, None]
        m_1 = cos_theta * u + sin_theta * v
        m_2 = -sin_theta * u + cos_theta * v
        return np.stack([m_1, m_2], axis=1)

    @staticmethod
    def compute_omega(theta: np.ndarray, kb: np.ndarray, bishop_frame: np.ndarray) -> np.ndarray:
        """
        Computes omega for all edges where
            omega[i][0] = omega_i^{i-1} and omega[i][1] = omega_i^i
        """
        material_frames = RodUtil.compute_material_frames(theta=theta, bishop_frame=bishop_frame)

        # j = i
        m_1, m_2 = material_frames[:, 0, :], material_frames[:, 1, :]
        omega_i1 = Vector.inner_products(kb, m_2)
        omega_i2 = -Vector.inner_products(kb, m_1)

        # j = i - 1
        m_1m1, m_2m1 = material_frames[:-1, 0, :], material_frames[:-1, 1, :]
        omega_im1_1 = Vector.inner_products(kb[1:], m_2m1)
        omega_im1_2 = -Vector.inner_products(kb[1:], m_1m1)

        # Stack into shape (n, 2)
        omega_i = np.column_stack([omega_i1, omega_i2])
        omega_im1 = np.column_stack([omega_im1_1, omega_im1_2])
        omega_im1 = np.concatenate([np.zeros((1, 2)), omega_im1])

        # Put it all together
        omegas = np.zeros((omega_i.shape[0], 2, 2))
        omegas[:, 0, :] = omega_im1
        omegas[:, 1, :] = omega_i
        return omegas

    @staticmethod
    def update_bishop_frames(pos: np.ndarray, theta: np.ndarray, bishop_frame: np.ndarray) -> np.ndarray:
        """
        Computes the Bishop frame for each edge in the rod
        """
        # First compute the bishop frame vector for edge 0
        t0 = pos[1] - pos[0]
        t0 /= np.linalg.norm(t0)

        # Get vector orthogonal to t0 to define the bishop frame
        u = Vector.compute_orthogonal_vec(t0)
        v = np.cross(t0, u)
        u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
        bishop_frame[0] = np.array([u, v])

        # Parallel transport the frame along the strand
        n = theta.shape[0]
        for i in range(1, n):
            # Get edge and the previous edge
            t_i, t_im1 = (pos[i + 1] - pos[i]), (pos[i] - pos[i - 1])
            t_i, t_im1 = t_i / np.linalg.norm(t_i), t_im1 / np.linalg.norm(t_im1)

            # Compute the rotation that goes from the previous edge to the current edge
            if np.dot(t_im1, t_i) > 1 - 1e-6:
                P_i = Quaternion.identity()
            else:
                rot_axis = np.cross(t_im1, t_i)
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                rot_angle = np.arccos(np.dot(t_im1, t_i))
                P_i = Quaternion.from_angle_axis(rot_angle, rot_axis)
                P_i.normalize()
            # Parallel transport (rotate) u
            u = P_i.rotate_vec(u)
            v = np.cross(t_i, u)
            u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)

            # Update the frame
            bishop_frame[i] = np.array([u, v])
        return bishop_frame

    @staticmethod
    def compute_nabla_kb(pos: np.ndarray, kb: np.ndarray, kb_den: np.ndarray) -> np.ndarray:
        """
        For each edge i, computes nabla_{i-1}, nabla_i, nabla_{i+1} of (kb)_i (the other terms are 0)
        """
        e = pos[1:] - pos[:-1]
        e_skew_sym = Vector.skew_sym(e)

        kb_i = kb[1:]  # Undefined for i = 0

        # nabla_{i-1}
        kb_e_T = Vector.outer_products(kb_i, e[1:])
        num_im1 = 2 * e_skew_sym[1:] + kb_e_T

        # nabla_{i+1}
        kb_em1_T = Vector.outer_products(kb[1:], e[:-1])
        num_ip1 = 2 * e_skew_sym[:-1] - kb_em1_T

        nabla_im1 = num_im1 / kb_den[:, None, None]
        nabla_ip1 = num_ip1 / kb_den[:, None, None]
        nabla_i = -(nabla_im1 + nabla_ip1)

        nabla_kb = np.stack([nabla_im1, nabla_i, nabla_ip1], axis=1)
        # Concat zeros for i=0
        nabla_kb = np.concatenate([np.zeros((1, 3, 3, 3)), nabla_kb])

        return nabla_kb

    @staticmethod
    def compute_nabla_psi(kb: np.ndarray, rest_edge_lengths: np.ndarray) -> np.ndarray:
        """
        Computes nabla_{i-1} psi_i, nabla_i psi_i, nabla_{i+1} psi_i for each edge i
        """
        # Gradients given by Eq. 9
        nabla_im1 = kb[1:] / (2 * rest_edge_lengths[:-1, None])
        nabla_ip1 = -kb[1:] / (2 * rest_edge_lengths[1:, None])
        nabla_i = -(nabla_im1 + nabla_ip1)

        nabla_psi = np.stack([nabla_im1, nabla_i, nabla_ip1], axis=1)
        # Concat zeros for i=0
        nabla_psi = np.concatenate([np.zeros((1, 3, 3)), nabla_psi])
        return nabla_psi
