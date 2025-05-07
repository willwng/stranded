import cupy as cp


class Quaternion:
    def __init__(self, w: float, x: float, y: float, z: float):
        self.w = w
        self.x = x
        self.y = y
        self.z = z

    def length2(self) -> float:
        return self.w ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2

    def length(self) -> float:
        return cp.sqrt(self.length2())

    def inv_length(self) -> float:
        return 1.0 / self.length()

    def normalize(self):
        inv_l = self.inv_length()
        self.w *= inv_l
        self.x *= inv_l
        self.y *= inv_l
        self.z *= inv_l

    def rotate_vec(self, v: cp.ndarray) -> cp.ndarray:
        pure = cp.array([self.x, self.y, self.z])
        pure_x_v = cp.cross(pure, v)
        pure_x_pure_x_v = cp.cross(pure, pure_x_v)
        return v + 2.0 * ((pure_x_v * self.w) + pure_x_pure_x_v)

    def __matmul__(self, other: cp.ndarray) -> cp.ndarray:
        """ Overloads @ to apply rotation """
        return self.rotate_vec(other)

    @staticmethod
    def from_angle_axis(angle: float, axis: cp.ndarray) -> "Quaternion":
        cos_half = cp.cos(angle / 2)
        sin_half = cp.sin(angle / 2)
        return Quaternion(cos_half, axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half)

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(1.0, 0.0, 0.0, 0.0)


class RotationUtil:
    @staticmethod
    def compute_rotation_matrix(basis1: cp.ndarray, basis2: cp.ndarray) -> cp.ndarray:
        """ Computes the rotation matrix that maps basis1 to basis2. Assumes orthonormal rows. """
        return basis2 @ basis1.T

    @staticmethod
    def interpolate_rotation(R: cp.ndarray, t: float) -> cp.ndarray:
        """ Interpolate between identity and rotation matrix R using parameter t. """
        R_trace = min(3.0, cp.trace(R).item())  # convert to Python float to ensure min() works
        theta = cp.arccos((R_trace - 1) / 2)

        if cp.abs(theta) < 1e-10:
            return cp.eye(3)

        K = (R - R.T) / (2 * cp.sin(theta))
        axis = cp.array([K[2, 1], K[0, 2], K[1, 0]])

        theta_t = t * theta
        K_mat = cp.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        R_t = cp.eye(3) + cp.sin(theta_t) * K_mat + (1 - cp.cos(theta_t)) * (K_mat @ K_mat)
        return R_t

    @staticmethod
    def compute_darboux_vector(frame1: cp.ndarray, frame2: cp.ndarray, ds: float) -> cp.ndarray:
        """
        Computes the Darboux vector that transforms frame1 into frame2 over a small displacement ds.
        Assumes orthonormal columns.
        """
        R = frame2 @ frame1.T
        W = (R - R.T) / (2 * ds)
        darboux_vector = cp.array([
            W[2, 1],
            W[0, 2],
            W[1, 0]
        ])
        return darboux_vector

