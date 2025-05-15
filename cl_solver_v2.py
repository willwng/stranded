import numpy as np
import open3d as o3d
from typing import Tuple, List


class CLSimulator:
    """A minimal centre‑line simulator for multiple hair strands.

    The model treats each strand as a chain of lumped point masses
    connected by inextensible edges.  Forces grow linearly with the
    separation between points on *different* strands and are
    exponentially amplified toward the strand tips.  Root points are
    pinned via XPBD distance constraints (zero compliance).
    """

    # ───────────────────────────── public API ──────────────────────────────
    def __init__(
        self,
        strands: np.ndarray,
        *,
        radius: float = 1.0,
        alpha: float = 2.0,
        timestep: float = 1e-2,
        mass: float = 1.0,
        force_scale: float = 10.0,
        compliance: float = 0.0,
        proj_iters: int = 8,
    ) -> None:
        """Construct the simulator.

        Parameters
        ----------
        strands : (S, P, 3) array_like
            Initial positions of *S* strands, each containing *P* points.
        radius : float, optional
            Neighbourhood radius for inter‑strand force evaluation.
        alpha : float, optional
            Exponential factor that amplifies forces from root→tip.
        timestep : float, optional
            Integration timestep Δt.
        mass : float, optional
            Mass of each point (shared for simplicity).
        force_scale : float, optional
            Global multiplier on the distance‑growth kernel `k·r`.
        compliance : float, optional
            XPBD compliance (0 ⇒ perfectly inextensible edges).
        proj_iters : int, optional
            Number of Gauss‑Seidel iterations used in the XPBD projector.
        """
        strands = np.asarray(strands, dtype=np.float64)
        if strands.ndim != 3 or strands.shape[-1] != 3:
            raise ValueError("`strands` must have shape (S, P, 3)")

        # ‑‑ basic geometry -----------------------------------------------
        self.strands = strands.copy()
        self.n_strands, self.n_pts, _ = strands.shape
        self.n_total = self.n_strands * self.n_pts

        # Indices ↔ strand helper
        self.point_strand_ids = np.repeat(np.arange(self.n_strands), self.n_pts)

        # ‑‑ physical parameters -----------------------------------------
        self.radius = float(radius)
        self.alpha = float(alpha)
        self.dt = float(timestep)
        self.mass = float(mass)
        self.inv_mass_base = np.full(self.n_total, 1.0 / mass)
        self.force_scale = float(force_scale)
        self.compliance = float(compliance)
        self.proj_iters = int(proj_iters)

        # ‑‑ dynamic state -----------------------------------------------
        self.positions = self.strands.reshape(-1, 3)
        self.velocities = np.zeros_like(self.positions)

        # ‑‑ edge topology (for XPBD) ------------------------------------
        self.edges, self.rest_lengths, self.root_idx, self.bend_triples, self.rest_angles = self._build_topology()

        # ‑‑ pre‑computed cumulative arc‑length per point -----------------
        self.arc_lengths = self._compute_arc_lengths()

        # ‑‑ Open3D octree for neighbour queries --------------------------
        self.pcd = o3d.geometry.PointCloud()
        self.octree = o3d.geometry.Octree(max_depth=8)
        self._refresh_octree()  # builds from current positions

    # ─────────────────────────── simulation steps ─────────────────────────
    def step(self) -> None:
        """Advance the system by one timestep Δt."""
        dt = self.dt

        # (1) external forces ------------------------------------------------
        forces = np.zeros_like(self.positions)
        for i in range(self.n_total):
            forces[i] = self._compute_force(i)

        # (2) semi‑implicit Euler velocity update ---------------------------
        inv_mass = self.inv_mass_base.copy()
        inv_mass[self.root_idx] = 0.0  # pin roots
        self.velocities += dt * forces * inv_mass[:, None]

        # (3) predicted positions (unconstrained) ---------------------------
        pos_pred = self.positions + dt * self.velocities

        # (4) XPBD projection (distance constraints) ------------------------
        self._xpbd_project(
            pos_pred,
            self.edges,
            self.rest_lengths,
            inv_mass,
            dt,
            self.compliance,
            self.proj_iters,
            self.bend_triples,
            self.rest_angles,
            bend_comp=1e-4
        )

        # (5) update velocities & commit -----------------------------------
        self.velocities = (pos_pred - self.positions) / dt
        self.velocities[self.root_idx] = 0.0
        self.positions = pos_pred

        # (6) update neighbour structure for next frame --------------------
        self._refresh_octree()

    def run(self, n_steps: int = 100) -> None:
        for _ in range(n_steps):
            self.step()

    def get_strands(self) -> np.ndarray:
        """Return positions with original (S, P, 3) shape."""
        return self.positions.reshape(self.n_strands, self.n_pts, 3)

    # ─────────────────────── diagnostic helpers ───────────────────────────
    def extensibility_error(self) -> Tuple[float, float]:
        """Return (max_abs, max_rel) edge‑length violations."""
        p0 = self.positions[self.edges[:, 0]]
        p1 = self.positions[self.edges[:, 1]]
        curr_len = np.linalg.norm(p0 - p1, axis=1)
        abs_err = curr_len - self.rest_lengths
        rel_err = abs_err / self.rest_lengths
        return float(np.max(np.abs(abs_err))), float(np.max(np.abs(rel_err)))

    # ───────────────────────── internal helpers ───────────────────────────
    def _build_topology(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (edges, rest_lengths, root_indices, bend_triples, rest_angles)."""
        edges: List[Tuple[int, int]] = []
        rest: List[float] = []
        roots: List[int] = []
        for s in range(self.n_strands):
            base = s * self.n_pts
            roots.append(base)  # first point in strand
            for p in range(self.n_pts - 1):
                i, j = base + p, base + p + 1
                edges.append((i, j))
                rest.append(np.linalg.norm(self.positions[i] - self.positions[j]))

        bend_triples = [] # (i-1, i, i+1) indices
        rest_angles = [] # θ_i^0

        for s in range(self.n_strands):
            base = s * self.n_pts
            for p in range(1, self.n_pts - 1):
                i0, i1, i2 = base + p - 1, base + p, base + p + 1
                bend_triples.append((i0, i1, i2))
                e0 = self.positions[i1] - self.positions[i0]
                e1 = self.positions[i2] - self.positions[i1]
                cos0 = np.clip(np.dot(e0, e1) /
                            (np.linalg.norm(e0) * np.linalg.norm(e1)), -1.0, 1.0)
                rest_angles.append(np.arccos(cos0))

        return (
            np.asarray(edges, dtype=np.int32),
            np.asarray(rest, dtype=np.float64),
            np.asarray(roots, dtype=np.int32),
            np.asarray(bend_triples, dtype=np.int32),
            np.asarray(rest_angles, dtype=np.float64)
        )

    def _compute_arc_lengths(self) -> np.ndarray:
        """Cumulative distance from root for every point (flattened)."""
        arc = np.zeros((self.n_strands, self.n_pts))
        for s in range(self.n_strands):
            diffs = np.linalg.norm(np.diff(self.strands[s], axis=0), axis=1)
            arc[s, 1:] = np.cumsum(diffs)
        return arc.reshape(-1)

    # ‑‑ neighbour search ---------------------------------------------------
    def _refresh_octree(self) -> None:
        self.pcd.points = o3d.utility.Vector3dVector(self.positions)
        self.octree.clear()
        self.octree.convert_from_point_cloud(self.pcd, size_expand=0.01)

    def _collect_neighbors(self, query: np.ndarray) -> List[int]:
        """Return list of point indices within `self.radius` of `query`."""
        idxs: List[int] = []

        def cb(node, node_info):
            if isinstance(node, o3d.geometry.OctreeLeafNode):
                for i in node.indices:
                    pt = self.positions[i]
                    if np.linalg.norm(pt - query) <= self.radius:
                        idxs.append(i)
            return False

        self.octree.traverse(cb)
        return idxs

    # ‑‑ physics -----------------------------------------------------------
    def _compute_force(self, idx: int) -> np.ndarray:
        """Inter‑strand force on point *idx* (root→tip amplified).
            Contributions summed for all strands
        """
        q = self.positions[idx]
        s_id = self.point_strand_ids[idx]
        s_arc = self.arc_lengths[idx]

        # tip amplification (1 at root → e^α at tip)
        strand_length = self.arc_lengths[s_id * self.n_pts + self.n_pts - 1]
        amp = np.exp(self.alpha * (s_arc / (strand_length + 1e-12)))

        # gather neighbours in other strands
        nbr_idx = [j for j in self._collect_neighbors(q) if self.point_strand_ids[j] != s_id]
        if not nbr_idx:
            return np.zeros(3)

        diff = q - self.positions[nbr_idx]
        mag2 = np.einsum("ij,ij->i", diff, diff)
        valid = mag2 > 1e-12
        if not np.any(valid):
            return np.zeros(3)

        diff = diff[valid]
        mag = np.sqrt(mag2[valid])
        direction = diff / mag[:, None]
        weight = self.force_scale * mag  # linear growth k·r
        #return -amp * (weight[:, None] * direction).sum(axis=0)
        return amp * (weight[:, None] * direction).sum(axis=0)

    # ‑‑ XPBD distance projector ------------------------------------------
    @staticmethod
    def _xpbd_project(
        pos: np.ndarray,
        edges: np.ndarray,
        rest: np.ndarray,
        inv_mass: np.ndarray,
        dt: float,
        comp: float,
        n_iter: int,
        bend_triples=None,
        rest_angles=None,
        bend_comp=1e-4
    ) -> None:
        # bending constraints
        if bend_triples is not None:
            lamb_bend = np.zeros(len(bend_triples))
            alpha_b = bend_comp / (dt * dt)
            for _ in range(n_iter):
                for k, (i0, i1, i2) in enumerate(bend_triples):
                    x0, x1, x2 = pos[i0], pos[i1], pos[i2]
                    e0 = x1 - x0
                    e1 = x2 - x1

                    n0 = np.linalg.norm(e0)
                    n1 = np.linalg.norm(e1)
                    if n0 < 1e-8 or n1 < 1e-8:
                        continue

                    # current angle
                    cos_t = np.clip(np.dot(e0, e1) / (n0 * n1), -1.0, 1.0)
                    theta = np.arccos(cos_t)
                    C     = theta - rest_angles[k]

                    # angle gradient wrt the three points
                    J0 =  (1/n0) * ( (cos_t / n0)*e0 - e1 / n1 ) / np.sin(theta)
                    J2 =  (1/n1) * ( (cos_t / n1)*e1 - e0 / n0 ) / np.sin(theta)
                    J1 = -(J0 + J2)

                    w_sum = (inv_mass[i0]*np.dot(J0, J0) +
                            inv_mass[i1]*np.dot(J1, J1) +
                            inv_mass[i2]*np.dot(J2, J2))

                    if w_sum == 0:
                        continue

                    dl = (-C - alpha_b * lamb_bend[k]) / (w_sum + alpha_b)
                    lamb_bend[k] += dl

                    pos[i0] += inv_mass[i0] * dl * J0
                    pos[i1] += inv_mass[i1] * dl * J1
                    pos[i2] += inv_mass[i2] * dl * J2

        # inextensibility constraints
        lambdas = np.zeros(len(edges), dtype=np.float64)
        alpha = comp / (dt * dt)

        for _ in range(n_iter):
            for k, (i, j) in enumerate(edges):
                xi = pos[i]
                xj = pos[j]
                diff = xi - xj
                dist = np.linalg.norm(diff)
                if dist < 1e-12:
                    continue

                C = dist - rest[k]
                grad = diff / dist
                w_sum = inv_mass[i] + inv_mass[j]
                if w_sum == 0.0:
                    continue

                dl = (-C - alpha * lambdas[k]) / (w_sum + alpha)
                lambdas[k] += dl
                corr = dl * grad
                pos[i] += inv_mass[i] * corr
                pos[j] -= inv_mass[j] * corr
