import numpy as np

from maths.quaternion import Quaternion
from maths.vectors import Vector


class Rod:
    """
    A rod has n+1 vertices and n edges,
    - each vertex is numbered from 0 to n+1
        - position is represented using as a 3-vector
        - length is defined as average of the lengths of the two edges connected to it
        - a curvature binormal kb_i
    - each edge e_i is the edge between vertices i and i+1
        - e_i has an angle theta_i (rotation wrst the Bishop frame)
        - e_i has the bishop frame (u_i, v_i) associated with it
     - omega_bar is the material curvature
        - omega_bar[i][0] is \bar{omega}_{i}^{i-1} and omega_bar[i][1] is \bar{omega}_{i}^{i}
     """
    n: int

    # The following uniquely define a rod and are updated during solve
    pos: np.ndarray  # (n+1) x 3
    vel: np.ndarray  # (n+1) x 3
    theta: np.ndarray  # n x 1

    # Update after center-lines (positions) are updated
    bishop_frame: np.ndarray  # n x 2 x 3

    # Pre-computed values
    l_bar: np.ndarray  # n x 1. Rest length of node
    l_bar_edge: np.ndarray  # n x 1. Rest length of edge
    omega_bar: np.ndarray  # n x 2 x 2. Rest material curvature

    def __init__(self, pos: np.ndarray, thetas: np.ndarray):
        assert thetas.shape[0] == pos.shape[0] - 1
        self.pos = pos
        self.vel = np.zeros_like(pos)
        self.theta = thetas
        self.n = thetas.shape[0] - 1

        n_vertices = self.n + 2
        n_edges = self.n + 1

        # Compute rest lengths associated with each vertex (except ends)
        self.l_bar = np.zeros(n_vertices)
        for i in range(1, n_vertices - 1):
            self.l_bar[i] = self.compute_length(i)

        # Compute rest lengths for each edge
        self.l_bar_edge = np.zeros(n_edges)
        for i in range(n_edges):
            self.l_bar_edge[i] = np.linalg.norm(self.get_edge(i))

        # Compute the Bishop frames for each edge
        self.bishop_frame = np.zeros((n_edges, 2, 3))
        self.update_bishop_frame()

        # Compute omega_bar (all but the first edge)
        self.omega_bar = np.zeros((n_edges, 2, 2))
        for i in range(1, n_edges):
            for j_idx, j in enumerate([i - 1, i]):
                omega_ij = self.compute_omega(i, j)
                self.omega_bar[i][j_idx] = omega_ij
        return

    def compute_length(self, i: int):
        """ Computes the length of the edge i """
        assert 0 < i < (self.n + 1), f"Invalid index {i}"
        e_i, e_im1 = self.get_edge(i), self.get_edge(i - 1)
        mag_ei, mag_eim1 = np.linalg.norm(e_i), np.linalg.norm(e_im1)
        return (mag_ei + mag_eim1) / 2

    def compute_omega(self, i: int, j: int):
        """ Computes the material curvature """
        kb_i = self.compute_curvature_binormal(i)
        # Compute the material frame
        u, v = self.bishop_frame[j]
        theta_j = self.theta[j]
        m_1 = np.cos(theta_j) * u + np.sin(theta_j) * v
        m_2 = -np.sin(theta_j) * u + np.cos(theta_j) * v
        omega_ij = np.array([np.dot(kb_i, m_2), -np.dot(kb_i, m_1)])
        return omega_ij

    def get_edge(self, i: int):
        assert 0 <= i <= self.n, f"Invalid edge index {i}"
        return self.pos[i + 1] - self.pos[i]

    def compute_curvature_binormal(self, i: int):
        """
        Computes the curvature binormal vector at vertex i
        """
        assert 0 <= i <= self.n, f"Invalid index {i}"
        e_i, e_im1 = self.get_edge(i), self.get_edge(i - 1)
        kb_i = np.cross(2 * e_im1, e_i)
        kb_i /= (np.linalg.norm(e_im1) * np.linalg.norm(e_i) + np.dot(e_im1, e_i))
        return kb_i

    def update_bishop_frame(self):
        """
        Updates the Bishop frames of each edge in the strand by parallel transport
        """
        # First compute the bishop frame vector for edge 0
        t0 = self.get_edge(0)
        t0 /= np.linalg.norm(t0)

        # Get vector orthogonal to t0 to define the bishop frame
        u = Vector.compute_orthogonal_vec(t0)
        v = np.cross(t0, u)
        u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
        self.bishop_frame[0] = np.array([u, v])

        # Parallel transport the frame along the strand
        for i in range(1, self.n):
            # Get edge and the previous edge
            t_i, t_im1 = self.get_edge(i), self.get_edge(i - 1)
            t_i, t_im1 = t_i / np.linalg.norm(t_i), t_im1 / np.linalg.norm(t_im1)

            # Compute the rotation that goes from the previous edge to the current edge
            if np.dot(t_im1, t_i) > 1 - 1e-6:
                P_i = Quaternion.identity()
            else:
                rot_axis = np.cross(t_im1, t_i)
                rot_axis = rot_axis / np.linalg.norm(rot_axis)
                rot_angle = np.arccos(np.dot(t_im1, t_i))
                P_i = Quaternion.from_angle_axis(rot_angle, rot_axis)
                P_i.normalize()
            # Parallel transport (rotate) u
            u = P_i.rotate_vec(u)
            v = np.cross(t_i, u)
            u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)

            # Update the frame
            self.bishop_frame[i] = np.array([u, v])
        return
