import numpy as np


class Vector:
    # Matrix that does a counterclockwise pi/2 rotation to 2d vectors
    J = np.array([[0, -1], [1, 0]])

    @staticmethod
    def compute_orthogonal_vec(v) -> np.ndarray:
        """ Returns a vector orthogonal to v"""
        helper = np.array([1.0, 0.0, 0.0])
        if np.dot(v, helper) > 1 - 1e-6:
            helper = np.array([0.0, 1.0, 0.0])
        return np.cross(v, helper)

    @staticmethod
    def skew_sym(v) -> np.ndarray:
        """
        Returns the skew symmetric matrix of a 3d vector
        If v is an array of 3d vectors, returns an array of skew symmetric matrices
         """
        # Input is a single 3d vector
        if v.ndim == 1:
            v = v.reshape(1, 3)
        skew_matrices = np.zeros((v.shape[0], 3, 3))  # (n x 3 x 3)

        skew_matrices[:, 0, 1] = -v[:, 2]
        skew_matrices[:, 0, 2] = v[:, 1]
        skew_matrices[:, 1, 0] = v[:, 2]
        skew_matrices[:, 1, 2] = -v[:, 0]
        skew_matrices[:, 2, 0] = -v[:, 1]
        skew_matrices[:, 2, 1] = v[:, 0]

        return skew_matrices.squeeze()

    @staticmethod
    def outer_products(u, v) -> np.ndarray:
        """
        Given two arrays of vectors u and v (shape nx3),
            returns the outer product between corresponding vectors (shape nx3x3)
        """
        return np.einsum("ij,ik->ijk", u, v)

    @staticmethod
    def inner_products(u, v) -> np.ndarray:
        """
        Given two arrays of vectors u and v (shape nx3),
            returns the inner product between corresponding vectors (shape n)
        """
        return np.einsum("ij,ij->i", u, v)
