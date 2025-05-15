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
        self.inv_mass = np.full((strands.shape[1] * strands.shape[0]), 1 / mass)
        self.height_scale = height_scale

        self.num_strands, self.num_points, _ = strands.shape
        self.num_total_points = self.num_strands * self.num_points

        # Flatten positions and initialize velocities
        self.positions = strands.reshape(-1, 3).copy()
        self.velocities = np.zeros_like(self.positions)

        # Finding edges, rest_lengths, and root indices
        edges        = []          # (m, 2) indices of neighbouring points
        rest_lengths = []          # (m,) target segment lengths
        root_idx     = []          # list of point indices you want fixed

        for strand_idx, strand in enumerate(strands):
            root_idx.append(strand_idx * self.num_points)  # global index of first point
            for i in range(self.num_points - 1):
                a = strand_idx * self.num_points + i
                b = strand_idx * self.num_points + i + 1
                edges.append((a, b))
                rest_lengths.append(np.linalg.norm(strand[i + 1] - strand[i]))

        edges = np.asarray(edges, dtype=np.int32)
        rest_lengths = np.asarray(rest_lengths, dtype=np.float64)
        root_idx = np.asarray(root_idx, dtype=np.int32)

        self.rest_lengths = rest_lengths
        self.edges = edges
        self.root_idx = root_idx

        # Precompute arc lengths
        self.arc_lengths = self.precompute_arc_lengths() # arc length is cumulative, rest_lengths are deltas btwn two points

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

        direction = diff / mag[:, None] # unit vectors
        weight    =  10 * mag # linear growth kernel

        force_vec = -(weight[:, None] * direction).sum(0)
        return decay * force_vec
    
    def xpbd_project(pos_pred, edges, rest_lengths,
                inv_mass,  # (N,) inverse masses; 0 for pinned roots
                dt, comp,  # comp = compliance (0 ⇒ perfectly rigid)
                n_iter=8):

        lambdas = np.zeros(len(edges))     # Lagrange multipliers

        alpha   = comp / dt**2             # "compliance" term
        for _ in range(n_iter):
            for c, (i, j) in enumerate(edges):
                xi, xj   = pos_pred[i], pos_pred[j]
                diff     = xi - xj
                dist     = np.linalg.norm(diff)
                if dist == 0.:             # degenerate
                    continue

                C        = dist - rest_lengths[c]
                grad     = diff / dist
                w_sum    = inv_mass[i] + inv_mass[j]
                if w_sum == 0.:
                    continue               # both points pinned

                # XPBD delta‑lambda update
                dl       = (-C - alpha * lambdas[c]) / (w_sum + alpha)
                lambdas[c] += dl

                corr     = dl * grad
                pos_pred[i] += inv_mass[i] * corr
                pos_pred[j] -= inv_mass[j] * corr

    def step(self):
        dt = self.timestep
        N = self.num_total_points

        # Update octree with current positions
        self.pcd.points = o3d.utility.Vector3dVector(self.positions)
        self.octree.clear()
        self.octree.convert_from_point_cloud(self.pcd, size_expand=0.01)

        # Compute forces
        forces = np.zeros_like(self.positions)
        for i in range(self.num_total_points):
            forces[i] = self.compute_force(i)

        # semi implicit velocity update
        print(self.inv_mass)
        self.velocities += dt * forces * self.inv_mass[:, None]   # inv_mass = 1/m
        pos_pred = self.positions + dt * self.velocities

        # pinning roots by setting inv_mass = 0
        self.inv_mass[self.root_idx] = 0.0

        # constraint solve for inextensitiliby
        xpbd_project(pos_pred, self.edges, self.rest_lengths, self.inv_mass, dt, comp=0.0, n_iter=8)

        # Semi-implicit Euler update
        self.velocities = (pos_pred - self.positions) / dt
        self.positions = pos_pred

    def run(self, num_steps=100):
        for _ in range(num_steps):
            self.step()

    def get_strands(self):
        return self.positions.reshape(self.num_strands, self.num_points, 3)
    
    def extensibility_check(self):
        # current positions of each edge’s two endpoints
        p0 = self.positions[self.edges[:, 0]]
        p1 = self.positions[self.edges[:, 1]]

        # current edge lengths
        curr_len = np.linalg.norm(p0 - p1, axis=1)  # (m,)

        # absolute and relative errors
        abs_err  = curr_len - self.rest_lengths
        rel_err  = abs_err / self.rest_lengths

        max_abs  = np.max(np.abs(abs_err))
        max_rel  = np.max(np.abs(rel_err))

        # Option A: return a tuple for flexible logging
        return max_abs, max_rel
   

