import numpy as np
import open3d as o3d

class CL_Simulator:
    def __init__(self, strands, height_scale=0.5, radius=1.0, alpha=2.0, timestep=0.01, mass=1.0):
        """
        strands: (N_strands, N_points, 3) array of initial positions
        """
        self.strands = strands
        self.radius = radius
        self.alpha = alpha
        self.timestep = timestep
        self.mass = mass
        self.height_scale = height_scale

        self.num_strands, self.num_points, _ = strands.shape
        self.num_total_points = self.num_strands * self.num_points

        # Flatten positions and initialize velocities
        self.positions = strands.reshape(-1, 3).copy()
        self.velocities = np.zeros_like(self.positions)

        # Precompute arc lengths
        self.arc_lengths = self.precompute_arc_lengths()

        # Build strand ID map
        self.point_strand_ids = np.repeat(np.arange(self.num_strands), self.num_points)

        # Build octree and cache PCD
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.positions)
        self.octree = o3d.geometry.Octree(max_depth=8)
        self.octree.convert_from_point_cloud(self.pcd, size_expand=0.01)

    def precompute_arc_lengths(self):
        arc_lengths = np.zeros((self.num_strands, self.num_points))
        for i in range(self.num_strands):
            diffs = np.linalg.norm(np.diff(self.strands[i], axis=0), axis=1)
            arc_lengths[i, 1:] = np.cumsum(diffs)
        return arc_lengths.flatten()

    def collect_neighbors(self, query_point):
        neighbors = []

        def callback(node, node_info):
            if isinstance(node, o3d.geometry.OctreeLeafNode):
                for idx in node.indices:
                    pt = np.asarray(self.pcd.points)[idx]
                    dist = np.linalg.norm(pt - query_point)
                    if dist <= self.radius:
                        neighbors.append((idx, pt))
            return False

        self.octree.traverse(callback)
        return neighbors

    def compute_force(self, idx: int) -> np.ndarray:
        q = self.positions[idx]
        sid = self.point_strand_ids[idx]
        s = self.arc_lengths[idx]

        # root‑to‑tip amplification (α > 0)
        strand_length = (self.num_points - 1) * self.height_scale
        decay = np.exp(self.alpha * (s / strand_length)) # 1 at root → e^{α} at tip

        # neighbour query
        nbrs = self.collect_neighbors(q)
        if not nbrs:
            return np.zeros(3)

        n_idx  = np.fromiter((i for i, _ in nbrs), dtype=np.int64)
        nbr_pt = self.positions[n_idx]

        mask = self.point_strand_ids[n_idx] != sid    # ignore same strand
        if not mask.any():
            return np.zeros(3)

        diff   = q - nbr_pt[mask]
        mag2   = (diff**2).sum(1)
        good   = mag2 > 1e-12
        if not good.any():
            return np.zeros(3)

        diff   = diff[good]
        mag    = np.sqrt(mag2[good])

        direction = diff / mag[:, None]               # unit vectors
        weight    =  10 * mag                               # linear growth kernel

        force_vec = -(weight[:, None] * direction).sum(0)
        return decay * force_vec

    def step(self):
        # Update octree with current positions
        self.pcd.points = o3d.utility.Vector3dVector(self.positions)
        self.octree.clear()
        self.octree.convert_from_point_cloud(self.pcd, size_expand=0.01)

        # Compute forces
        forces = np.zeros_like(self.positions)
        for i in range(self.num_total_points):
            forces[i] = self.compute_force(i)

        # Semi-implicit Euler update
        self.velocities += self.timestep * forces / self.mass
        self.positions += self.timestep * self.velocities

    def run(self, num_steps=100):
        for _ in range(num_steps):
            self.step()

    def get_strands(self):
        return self.positions.reshape(self.num_strands, self.num_points, 3)
