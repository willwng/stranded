import numpy as np

from math_util.rotation import RotationUtil
from rod.helix import Helix
from rod.helix_util import HelixUtil
from rod.rod_util import RodUtil


class RodHelixConverter:
    @staticmethod
    def rod_to_helix(pos: np.ndarray, theta: np.ndarray) -> Helix:
        """
        Converts a rod (explicit representation) to a helix (implicit representation)
        """
        n_sites = pos.shape[0]
        q = np.zeros(3 * n_sites)

        # Edge lengths and material frames of each edge
        e = pos[1:] - pos[:-1]
        edge_lengths = np.linalg.norm(e, axis=1)
        bishop_frame = RodUtil.compute_bishop_frames(pos=pos)
        material_frame = RodUtil.compute_material_frames(theta=theta, bishop_frame=bishop_frame)

        # Estimate n0 by interpolating back from the first two material frames
        m_prev, m_next = material_frame[0], material_frame[1]
        t_prev, t_next = e[0] / edge_lengths[0], e[1] / edge_lengths[1]
        edge_frame_prev, edge_frame_next = np.array([t_prev, m_prev[0], m_prev[1]]), np.array(
            [t_next, m_next[0], m_next[1]])
        rotation = RotationUtil.compute_rotation_matrix(edge_frame_prev, edge_frame_next)
        rotation = RotationUtil.interpolate_rotation(rotation, edge_lengths[0] / (edge_lengths[0] + edge_lengths[1]))
        n0 = rotation @ edge_frame_prev

        # For helices, we need to prescribe each site with a material frame
        site_material_frames = np.zeros((n_sites, 3, 3))
        site_material_frames[0] = n0
        for i in range(1, n_sites - 1):
            # Material frames of two edges that meet at this site
            m_prev, m_next = material_frame[i - 1], material_frame[i]
            t_prev, t_next = e[i - 1] / edge_lengths[i - 1], e[i] / edge_lengths[i]
            edge_frame_prev = np.array([t_prev, m_prev[0], m_prev[1]])
            edge_frame_next = np.array([t_next, m_next[0], m_next[1]])
            # Interpolate the material frames (based on distance of node from edge centers)
            rotation = RotationUtil.compute_rotation_matrix(edge_frame_prev, edge_frame_next)
            inter_fraction = edge_lengths[i - 1] / (edge_lengths[i] + edge_lengths[i - 1])
            rotation = RotationUtil.interpolate_rotation(rotation, inter_fraction)
            site_material_frames[i] = rotation @ edge_frame_prev
            # Final site, just propagate forward
            if i == n_sites - 2:
                site_material_frames[-1] = rotation @ edge_frame_next

        # Now we can compute the generalized coordinates
        for i in range(n_sites - 1):
            # Collect material frame at this site and next site
            prev_frame = site_material_frames[i]
            next_frame = site_material_frames[i + 1]
            # Compute the Darboux vector
            Omega = RotationUtil.compute_darboux_vector(prev_frame.T, next_frame.T, edge_lengths[i])
            # Compute curvatures through linear solve
            curvatures = np.linalg.solve(prev_frame.T, Omega)
            q[3 * i:3 * i + 3] = curvatures

        # Compute extra helix data
        s = np.cumsum(edge_lengths)
        s = np.insert(s, 0, 0)
        L = np.sum(edge_lengths)
        r0 = pos[0]
        return Helix(q=q, q0=q.copy(), n_sites=n_sites, s=s, L=L, r0=r0, n0=n0, EI=np.ones(3 * n_sites))

    @staticmethod
    def helix_to_rod(helix: Helix):
        r, n = HelixUtil.propagate(helix)
        pos = r.copy()

        # Compute theta from bishop frames and material frames
        bishop_frame = RodUtil.compute_bishop_frames(pos=pos)
        theta = np.zeros(pos.shape[0] - 1)
        for i in range(n.shape[0] - 1):
            # Collect bishop frame
            e = pos[i + 1] - pos[i]
            t = e / np.linalg.norm(e)
            b1, b2 = bishop_frame[i]

            # Interpolate the material frame between the two sites
            rotation = RotationUtil.compute_rotation_matrix(n[i], n[i + 1])
            rotation = RotationUtil.interpolate_rotation(rotation, 0.5)
            rotated_frame = rotation @ n[i]
            m1 = rotated_frame[1]

            # Find the rotation angle that takes [b1, b2] to [m1, m2], rotation about t
            b1_proj = b1 - np.dot(b1, t) * t
            b1_proj = b1_proj / np.linalg.norm(b1_proj)

            # Project m1 onto plane perpendicular to t
            m1_proj = m1 - np.dot(m1, t) * t
            m1_proj = m1_proj / np.linalg.norm(m1_proj)

            # Calculate angle using dot product
            cos_theta = np.dot(b1_proj, m1_proj)

            # We need to determine the sign of the rotation
            # Use cross product to check if we need to negate theta
            cross_prod = np.cross(b1_proj, m1_proj)
            sign = np.sign(np.dot(cross_prod, t))

            t = np.arccos(np.clip(cos_theta, -1.0, 1.0))
            t = sign * t
            theta[i] = t

        return pos, theta
