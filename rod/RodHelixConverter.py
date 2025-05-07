import cupy as cp
#from cupyx.scipy.optimize import minimize

from math_util.rotation import RotationUtil, Quaternion
from rod.helix import Helix
from rod.helix_util import HelixUtil
from rod.rod_util import RodUtil


class RodHelixConverter:
    @staticmethod
    def rod_to_helix(pos: cp.ndarray, theta: cp.ndarray, s: cp.ndarray = None) -> Helix:
        n_sites = pos.shape[0]
        q = cp.zeros(3 * n_sites)

        e = pos[1:] - pos[:-1]
        edge_lengths = cp.linalg.norm(e, axis=1)
        if s is None:
            s = cp.cumsum(edge_lengths)
            s = cp.concatenate([cp.array([0], dtype=s.dtype), s])
        arc = s[1:] - s[:-1]
        bishop_frame = RodUtil.compute_bishop_frames(pos)
        material_frame = RodUtil.compute_material_frames(theta, bishop_frame)

        m_prev, m_next = material_frame[0], material_frame[1]
        t_prev, t_next = e[0] / edge_lengths[0], e[1] / edge_lengths[1]
        #edge_frame_prev = cp.stack([t_prev.ravel(), m_prev[0].ravel(), m_prev[1].ravel()])
        edge_frame_next = cp.stack([t_prev, m_prev[0], m_prev[1]])
        edge_frame_next = cp.stack([t_next, m_next[0], m_next[1]])
        #edge_frame_next = cp.stack([t_next.ravel(), m_next[0].ravel(), m_next[1].ravel()])

        rotation = RotationUtil.compute_rotation_matrix(edge_frame_prev, edge_frame_next)
        rotation = RotationUtil.interpolate_rotation(rotation, arc[0] / (arc[0] + arc[1]))
        n0 = rotation.T @ edge_frame_prev

        site_material_frames = cp.zeros((n_sites, 3, 3))
        site_material_frames[0] = n0
        for i in range(1, n_sites - 1):
            m_prev, m_next = material_frame[i - 1], material_frame[i]
            t_prev, t_next = e[i - 1] / edge_lengths[i - 1], e[i] / edge_lengths[i]
            #edge_frame_prev = cp.stack([t_prev, m_prev[0], m_prev[1]])
            #edge_frame_prev = cp.stack([t_prev.ravel(), m_prev[0].ravel(), m_prev[1].ravel()])
            #edge_frame_next = cp.stack([t_next, m_next[0], m_next[1]])
            #edge_frame_next = cp.stack([t_next.ravel(), m_next[0].ravel(), m_next[1].ravel()])
            edge_frame_next = cp.stack([t_prev, m_prev[0], m_prev[1]])
            edge_frame_next = cp.stack([t_next, m_next[0], m_next[1]])

            rotation = RotationUtil.compute_rotation_matrix(edge_frame_prev, edge_frame_next)
            inter_fraction = arc[i - 1] / (arc[i] + arc[i - 1])
            rotation = RotationUtil.interpolate_rotation(rotation, inter_fraction)
            site_material_frames[i] = rotation @ edge_frame_prev
            if i == n_sites - 2:
                site_material_frames[-1] = rotation @ edge_frame_next

        for i in range(n_sites - 1):
            prev_frame = site_material_frames[i]
            next_frame = site_material_frames[i + 1]
            Omega = RotationUtil.compute_darboux_vector(prev_frame.T, next_frame.T, arc[i])
            curvatures = cp.linalg.solve(prev_frame.T, Omega)
            q[3 * i:3 * i + 3] = curvatures

        s = cp.cumsum(arc)
        s = cp.concatenate([cp.array([0], dtype=s.dtype), s])
        L = cp.max(s)
        r0 = pos[0]
        return Helix(q=q, q0=q.copy(), n_sites=n_sites, s=s, L=L, r0=r0, n0=n0, EI=cp.ones(3 * n_sites))

    @staticmethod
    def normalize_strand(pos, normalize_positions: bool, normalize_direction: bool, normalize_length: bool):
        if normalize_positions:
            pos -= pos[0]
        if normalize_length:
            e = pos[1:] - pos[:-1]
            edge_lengths = cp.linalg.norm(e, axis=1)
            pos /= cp.mean(edge_lengths)
        if normalize_direction:
            direction = pos[-1] - pos[0]
            direction /= cp.linalg.norm(direction)
            z_axis = cp.array([0, 0, 1])
            rot_axis = cp.cross(direction, z_axis)
            rot_axis /= cp.linalg.norm(rot_axis)
            rot_angle = cp.arccos(cp.dot(direction, z_axis))
            P_i = Quaternion.from_angle_axis(rot_angle, rot_axis)
            P_i.normalize()
            for i in range(pos.shape[0]):
                pos[i] = P_i @ pos[i]
        return pos

    @staticmethod
    def helix_to_rod(helix: Helix):
        r, n = HelixUtil.propagate(helix)
        pos = r.copy()
        bishop_frame = RodUtil.compute_bishop_frames(pos)
        theta = cp.zeros(pos.shape[0] - 1)
        for i in range(n.shape[0] - 1):
            e = pos[i + 1] - pos[i]
            t = e / cp.linalg.norm(e)
            b1, b2 = bishop_frame[i]
            rotation = RotationUtil.compute_rotation_matrix(n[i], n[i + 1])
            rotation = RotationUtil.interpolate_rotation(rotation, 0.5)
            rotated_frame = rotation @ n[i]
            m1 = rotated_frame[1]
            b1_proj = b1 - cp.dot(b1, t) * t
            b1_proj /= cp.linalg.norm(b1_proj)
            m1_proj = m1 - cp.dot(m1, t) * t
            m1_proj /= cp.linalg.norm(m1_proj)
            cos_theta = cp.dot(b1_proj, m1_proj)
            cross_prod = cp.cross(b1_proj, m1_proj)
            sign = cp.sign(cp.dot(cross_prod, t))
            angle = cp.arccos(cp.clip(cos_theta, -1.0, 1.0))
            theta[i] = sign * angle
        return pos, theta
