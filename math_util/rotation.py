""" A barebones quaternion class """
import numpy as np


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def length2(self) -> float:
        return self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2

    def length(self) -> float:
        return np.sqrt(self.length2())

    def inv_length(self) -> float:
        return 1.0 / self.length()

    def normalize(self):
        inv_l = self.inv_length()
        self.w *= inv_l
        self.x *= inv_l
        self.y *= inv_l
        self.z *= inv_l

    def rotate_vec(self, v: np.ndarray):
        pure = np.array([self.x, self.y, self.z])
        pure_x_v = np.cross(pure, v)
        pure_x_pure_x_v = np.cross(pure, pure_x_v)
        return v + 2.0 * ((pure_x_v * self.w) + pure_x_pure_x_v)

    def __matmul__(self, other):
        """ Overloads the @ operator to represent quaternion multiplication """
        return self.rotate_vec(other)

    @staticmethod
    def from_angle_axis(angle: float, axis: np.ndarray):
        """
        Returns a quaternion representing a rotation of angle about the axis
        """
        cos_half = np.cos(angle / 2)
        sin_half = np.sin(angle / 2)
        return Quaternion(cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half)

    @staticmethod
    def identity():
        return Quaternion(1.0, 0.0, 0.0, 0.0)


class RotationUtil:
    @staticmethod
    def compute_rotation_matrix(basis1: np.ndarray, basis2: np.ndarray) -> np.ndarray:
        """ Computes the rotation matrix that takes basis1 to basis2
            basis1 and basis2 have orthonormal rows
        """
        R = np.dot(basis2, basis1.T)
        return R

    @staticmethod
    def interpolate_rotation(R, t):
        """ Interpolate between identity and rotation matrix R using parameter t. """
        # Convert rotation matrix to axis-angle representation
        R_trace = min(3, np.trace(R))
        theta = np.arccos((R_trace - 1) / 2)
        if np.abs(theta) < 1e-10:
            return np.eye(3)

        K = (R - R.T) / (2 * np.sin(theta))
        axis = np.array([K[2, 1], K[0, 2], K[1, 0]])

        # Interpolate angle
        theta_t = t * theta

        # Rodrigues rotation formula
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        R_t = np.eye(3) + np.sin(theta_t) * K + (1 - np.cos(theta_t)) * np.dot(K, K)
        return R_t

    @staticmethod
    def compute_darboux_vector(frame1: np.ndarray, frame2: np.ndarray, ds: float) -> np.ndarray:
        """
        Computes the Darboux vector that transforms frame1 into frame2 over a small displacement ds.
            frame1 has orthonormal columns [t1, m1, n1]
            frame2 has orthonormal columns [t2, m2, n2]
        """
        # Compute the rotation matrix R
        R = frame2 @ frame1.T

        # Compute the skew-symmetric part
        W = (R - R.T) / (2 * ds)

        # Extract the Darboux vector components
        darboux_vector = np.array([
            W[2, 1],
            W[0, 2],
            W[1, 0]
        ])

        return darboux_vector
