import numpy as np


class Vector:
    # Matrix that does a counterclockwise pi/2 rotation to 2d vectors
    J = np.array([[0, -1], [1, 0]])

    @staticmethod
    def compute_orthogonal_vec(v):
        """ Returns a vector orthogonal to v"""
        helper = np.array([1.0, 0.0, 0.0])
        if np.dot(v, helper) > 1 - 1e-6:
            helper = np.array([0.0, 1.0, 0.0])
        return np.cross(v, helper)

    @staticmethod
    def skew_sym(v):
        """ Returns the skew symmetric matrix of a 3d vector """
        return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
