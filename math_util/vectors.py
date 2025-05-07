'''
Future: operate on a batch of vectors w/ shape (B, C)
'''

import cupy as cp

class Vector:
    # Matrix that does a counterclockwise pi/2 rotation to 2D vectors
    J = cp.array([[0, -1], [1, 0]])

    @staticmethod
    def compute_orthogonal_vec(v: cp.ndarray) -> cp.ndarray:
        """Returns a vector orthogonal to v"""
        helper = cp.array([1.0, 0.0, 0.0])
        if cp.dot(v, helper) > 1 - 1e-6:
            helper = cp.array([0.0, 1.0, 0.0])
        return cp.cross(v, helper)

    @staticmethod
    def skew_sym(v: cp.ndarray) -> cp.ndarray:
        """
        Returns the skew symmetric matrix of a 3D vector or array of vectors
        Output shape: (n, 3, 3) if batched, or (3, 3) if input is a single vector
        """
        if v.ndim == 1:
            v = v.reshape(1, 3)

        skew = cp.zeros((v.shape[0], 3, 3), dtype=cp.float64)
        skew[:, 0, 1] = -v[:, 2]
        skew[:, 0, 2] =  v[:, 1]
        skew[:, 1, 0] =  v[:, 2]
        skew[:, 1, 2] = -v[:, 0]
        skew[:, 2, 0] = -v[:, 1]
        skew[:, 2, 1] =  v[:, 0]

        return skew.squeeze()  # remove singleton dimension if input was 1D

    @staticmethod
    def outer_products(u: cp.ndarray, v: cp.ndarray) -> cp.ndarray:
        """Outer product for batched 3D vectors (n x 3 x 3)"""
        return cp.einsum("ij,ik->ijk", u, v)

    @staticmethod
    def inner_products(u: cp.ndarray, v: cp.ndarray) -> cp.ndarray:
        """Inner product for batched 3D vectors (n,)"""
        return cp.einsum("ij,ij->i", u, v)

    @staticmethod
    def matrix_multiply(M: cp.ndarray, v: cp.ndarray) -> cp.ndarray:
        """
        Matrix-vector multiplication
        M: (n x i x j), v: (n x j)
        returns: (n x i)
        """
        return cp.einsum("nij,nj->ni", M, v)

    @staticmethod
    def single_matrix_multiply(M: cp.ndarray, v: cp.ndarray) -> cp.ndarray:
        """
        M: (i x j), v: (n x l x j)
        returns: (n x l x i)
        """
        return cp.einsum("ij,klj->kli", M, v)

    @staticmethod
    def outer_product_helper(M: cp.ndarray, v: cp.ndarray) -> cp.ndarray:
        """
        M: (n x i x j), v: (k,)
        returns: (n x i x j x k)
        """
        return cp.einsum("nij,k->nijk", M, v)
