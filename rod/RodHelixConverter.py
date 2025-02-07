import numpy as np

from math_util.vectors import Vector
from rod.helix import Helix
from rod.helix_util import HelixUtil
from rod.rod_util import RodUtil


class RodHelixConverter:
    @staticmethod
    def rod_to_helix(pos: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """
        Converts a rod (explicit representation) to a helix (implicit representation)
        """
        q = np.zeros(3 * (pos.shape[0]))
        r0 = pos[0]

        e = pos[1:] - pos[:-1]
        bishop_frame = np.zeros((theta.shape[0], 2, 3))
        bishop_frame = RodUtil.update_bishop_frames(pos=pos, bishop_frame=bishop_frame)
        material_frame = RodUtil.compute_material_frames(theta=theta, bishop_frame=bishop_frame)

        for i in range(theta.shape[0]):
            # Material frame for the ith edge
            t_i = e[i] / np.linalg.norm(e[i])
            n_i1, n_i2 = material_frame[i]

            # Next material frame
            t_ip1 = e[i + 1] / np.linalg.norm(e[i + 1])
            n_ip1_1, n_ip1_2 = material_frame[i + 1]

            # Compute the Darboux vector

        return q

    @staticmethod
    def helix_to_rod(helix: Helix):
        r, n = HelixUtil.propagate(helix)
        pos = r
        # Compute the bishop frames, so we can get theta
        bishop_frame = np.zeros((n.shape[0] - 1, 2, 3))
        # The material frame of the first helix "edge"
        m0 = (helix.n0 + n[1]) / 2
        bishop_frame = RodUtil.update_bishop_frames(pos=pos, bishop_frame=bishop_frame, m0=m0[1:])

        theta = np.zeros(pos.shape[0] - 1)
        for i in range(n.shape[0] - 1):
            # Collect material frames
            _, mi1, mi2 = n[i]
            _, mp1, mp2 = n[i + 1]

            # Interpolate the material frame (for the edge)
            m1 = (mi1 + mp1) / 2
            m2 = (mi2 + mp2) / 2

            # Collect bishop frame
            e = pos[i + 1] - pos[i]
            t = e / np.linalg.norm(e)
            b1, b2 = bishop_frame[i]

            # Find the rotation angle that takes [b1, b2] to [m1, m2],
            #  rotation about t
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
