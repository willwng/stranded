import numpy as np
import open3d as o3d
from scipy.interpolate import RBFInterpolator
from typing import Tuple, List
import tqdm
from tqdm import tqdm


class CLSimulator:
    def __init__(self, strands: np.ndarray, timestep: float=0.04, mass: float=1.0):
        # strands: (n_strands, n_points, 3)
        self.strands = strands.copy()
        self.n_strands, self.n_pts, _ = strands.shape
        self.n_total = self.n_strands * self.n_pts

        self.positions = self.strands.reshape(-1, 3).copy() 
        self.velocities = np.zeros_like(self.positions)
        self.root_idx = np.arange(0, self.n_total, self.n_pts)

        self.dt = timestep
        self.mass = mass
        self.inv_mass_base = np.full(self.n_total, 1.0 / mass)
        self.field = None
        self.compliance: float = 0.0
        self.proj_iters: int = 8
    
        edges = []
        rest_lengths = []
        for s in range(self.n_strands):
            for i in range(self.n_pts - 1):
                idx0 = s * self.n_pts + i
                idx1 = s * self.n_pts + i + 1
                edges.append([idx0, idx1])
                rest_lengths.append(
                    np.linalg.norm(self.positions[idx0] - self.positions[idx1])
                )
        self.edges = np.array(edges, dtype=int)
        self.rest_lengths = np.array(rest_lengths, dtype=float)

    def create_field(self):
        # constructing scalar field
        roots = [strand[0] for strand in self.strands]
        roots = np.asarray(roots)

        values = np.zeros(roots.shape[0])
        offsets = np.linspace(-0.1, 1.0, num=24)
        normals = np.stack(roots, axis=1) # for now, just approx. sphere at (0, 0, 0)

        all_points = []
        all_values = []

        for d in offsets:
            displaced_points = roots + d * normals.T
            all_points.append(displaced_points)
            all_values.append(np.full(roots.shape[0], d))


        all_points = np.vstack(all_points)      # shape (5 * n_samples, 3)
        all_values = np.hstack(all_values)      # shape (5 * n_samples,)

        rbf = RBFInterpolator(all_points, all_values, neighbors=100, kernel='thin_plate_spline')  # or 'multiquadric', 'linear', etc.

        # meshgrid
        nx, ny, nz = 10, 10, 10
        x = np.linspace(-0.3, 0.3, nx)
        y = np.linspace(-0.3, 0.3, ny)
        z = np.linspace(-0.3, 0.3, nz)

        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)  # shape (nx*ny*nz, 3)

        field_vals = rbf(coords).reshape((nx, ny, nz))

        # constructing vector field from scalar field
        dx = (x[1] - x[0]) # x, y, z come from original construction of meshgrid
        dy = (y[1] - y[0])
        dz = (z[1] - z[0])
        #np.exp(-mag**2 / (2 * radius**2))
        grad_x, grad_y, grad_z = np.gradient(field_vals, dx, dy, dz)
        gradient = np.stack([grad_x, grad_y, grad_z], axis=-1)  # shape (nx, ny, nz, 3)
        norm = np.linalg.norm(gradient, axis=-1, keepdims=True) + 1e-8
        unit_vector = gradient / norm  # shape (nx, ny, nz, 3)
        vector_field = unit_vector * field_vals[..., None]  # shape (nx, ny, nz, 3), linear scaling with distance

        self.field = vector_field
        self.grid_x = x
        self.grid_y = y
        self.grid_z = z
        print("generated field")
        return

    def extensibility_error(self) -> Tuple[float, float]:
        """Return (max_abs, max_rel) edge‑length violations."""
        p0 = self.positions[self.edges[:, 0]]
        p1 = self.positions[self.edges[:, 1]]
        curr_len = np.linalg.norm(p0 - p1, axis=1)
        abs_err = curr_len - self.rest_lengths
        rel_err = abs_err / self.rest_lengths
        return float(np.max(np.abs(abs_err))), float(np.max(np.abs(rel_err)))

    def trilinear_lookup(self, field, grid_x, grid_y, grid_z, point):
        xi = np.clip(np.searchsorted(grid_x, point[0]) - 1, 0, len(grid_x) - 2)
        yi = np.clip(np.searchsorted(grid_y, point[1]) - 1, 0, len(grid_y) - 2)
        zi = np.clip(np.searchsorted(grid_z, point[2]) - 1, 0, len(grid_z) - 2)

        x0, x1 = grid_x[xi], grid_x[xi+1]
        y0, y1 = grid_y[yi], grid_y[yi+1]
        z0, z1 = grid_z[zi], grid_z[zi+1]

        xd = (point[0] - x0) / (x1 - x0 + 1e-8)
        yd = (point[1] - y0) / (y1 - y0 + 1e-8)
        zd = (point[2] - z0) / (z1 - z0 + 1e-8)

        c000 = field[xi, yi, zi]
        c001 = field[xi, yi, zi+1]
        c010 = field[xi, yi+1, zi]
        c011 = field[xi, yi+1, zi+1]
        c100 = field[xi+1, yi, zi]
        c101 = field[xi+1, yi, zi+1]
        c110 = field[xi+1, yi+1, zi]
        c111 = field[xi+1, yi+1, zi+1]

        c00 = c000 * (1 - xd) + c100 * xd
        c01 = c001 * (1 - xd) + c101 * xd
        c10 = c010 * (1 - xd) + c110 * xd
        c11 = c011 * (1 - xd) + c111 * xd

        c0 = c00 * (1 - yd) + c10 * yd
        c1 = c01 * (1 - yd) + c11 * yd

        c = c0 * (1 - zd) + c1 * zd
        return c

    def step(self) -> None:
        """Advance the system by one timestep dt"""
        # calculating vorces
        forces = np.zeros_like(self.positions)
        for i, p in enumerate(self.positions):
            forces[i] = self.trilinear_lookup(self.field, self.grid_x, self.grid_y, self.grid_z, p)

        # semi‑implicit euler velocity update
        inv_mass = self.inv_mass_base.copy()
        inv_mass[self.root_idx] = 0.0  # pin roots
        self.velocities += self.dt * forces * inv_mass[:, None]

        pos_pred = self.positions + self.dt * self.velocities

        # constraint solve for inextensibility
        self._xpbd_project(
            pos_pred,
            self.edges,
            self.rest_lengths,
            inv_mass,
            self.dt,
            self.compliance,
            self.proj_iters,
        )

        self.velocities = (pos_pred - self.positions) / self.dt
        self.velocities[self.root_idx] = 0.0
        self.positions = pos_pred
        return

    def run(self, n_steps: int = 100) -> None:
        for _ in tqdm(range(n_steps), position=0, leave=True):
            self.step()
        return
    
    def get_strands(self) -> np.ndarray:
        return self.positions.reshape(self.n_strands, self.n_pts, 3)

    def _xpbd_project(self, 
        pos: np.ndarray,
        edges: np.ndarray,
        rest: np.ndarray,
        inv_mass: np.ndarray,
        dt: float,
        comp: float,
        n_iter: int,
    ) -> None:
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
        return



