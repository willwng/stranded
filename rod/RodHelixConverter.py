import numpy as np

from math_util.rotation import RotationUtil
from math_util.vectors import Vector
from rod.helix import Helix
from rod.helix_util import HelixUtil
from rod.rod_util import RodUtil


class RodHelixConverter:
    @staticmethod
    def rod_to_helix(pos: np.ndarray, theta: np.ndarray, init_bishop_frame: np.ndarray, n0: np.ndarray) -> np.ndarray:
        """
        Converts a rod (explicit representation) to a helix (implicit representation)
        """
        n_sites = pos.shape[0]
        q = np.zeros(3 * n_sites)
        r0 = pos[0]

        e = pos[1:] - pos[:-1]
        edge_lengths = np.linalg.norm(e, axis=1)
        bishop_frame = RodUtil.compute_bishop_frames(pos=pos, m0=init_bishop_frame)
        material_frame = RodUtil.compute_material_frames(theta=theta, bishop_frame=bishop_frame)

        # For helices, we need to prescribe each site with a material frame
        site_material_frames = np.zeros((n_sites, 3, 3))
        site_material_frames[0] = n0
        curr_frame = n0
        for i in range(n_sites - 1):
            m = material_frame[i]
            t = e[i] / np.linalg.norm(e[i])
            edge_frame = np.array([t, m[0], m[1]])
            # Propagate the material frames
            rotation = RotationUtil.compute_rotation_matrix(curr_frame, edge_frame)
            rotation = RotationUtil.interpolate_rotation(rotation, 2.0)
            curr_frame = rotation @ curr_frame
            site_material_frames[i + 1] = curr_frame

        # Now we can compute the generalized coordinates
        for i in range(n_sites - 1):
            # Collect material frame at this site and next site
            prev_frame = site_material_frames[i]
            next_frame = site_material_frames[i + 1]
            # Compute the Darboux vector
            Omega = RotationUtil.compute_darboux_vector(prev_frame.T, next_frame.T, edge_lengths[i])
            # Compute curvatures through solve
            curvatures = np.linalg.solve(prev_frame.T, Omega)
            q[3 * i:3 * i + 3] = curvatures

        return q

    @staticmethod
    def helix_to_rod(helix: Helix):
        r, n = HelixUtil.propagate(helix)
        pos = r
        # The material frame of the first edge
        rotation = RotationUtil.compute_rotation_matrix(n[0], n[1])
        rotation = RotationUtil.interpolate_rotation(rotation, 0.5)
        init_bishop_frame = (rotation @ n[0])[1:]
        bishop_frame = RodUtil.compute_bishop_frames(pos=pos, m0=init_bishop_frame)

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

        return pos, theta, init_bishop_frame
