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