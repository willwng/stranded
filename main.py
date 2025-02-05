import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm

from energies.bend_twist import BendTwist
from energies.bend import Bend
from energies.gravity import Gravity
from energies.twist import Twist
from rod.reduced import Reduced
from rod.rod_generator import RodGenerator
from solver.sim import Sim
from visualization.visualizer import Visualizer


def create_frame(pos: np.ndarray,
                 material_frame: np.ndarray,
                 point_radii: np.ndarray,
                 forces: np.ndarray,
                 ax1_radii: np.ndarray,
                 ax2_radii: np.ndarray,
                 point_style: list[str],
                 i: int):
    Visualizer.strand_to_obj(pos=pos, material_frame=material_frame, point_radii=point_radii, forces=forces,
                             ax1_radii=ax1_radii, ax2_radii=ax2_radii, point_style=point_style,
                             output_file=f"output/obj/obj_{i}.obj")
    # Visualizer.to_simple_obj(pos=pos, output_file=f"output/obj/obj_{i}.obj")
    return


def main():
    # Create a rod
    # pos, theta = RodGenerator.straight_rod(n_points = 10)
    import_pos, import_theta = RodGenerator.from_obj(file_path="../../blender/sarah_1.obj", scale=10)
    # import_pos, import_theta = RodGenerator.example_rod(n=20)

    # Straighten the rod (keeping edge lengths constant)
    straight_pos, straight_theta = import_pos.copy(), import_theta.copy()
    edge_lengths = np.linalg.norm(import_pos[1:] - import_pos[:-1], axis=1)
    for i in range(0, import_pos.shape[0]):
        straight_pos[i] = np.array([0, 0, straight_pos[0, 2] - np.sum(edge_lengths[:i])], dtype=np.float64)
    # Translate so pos[0] are unchanged
    correction = import_pos[0] - straight_pos[0]
    straight_pos[:, 0] += correction[0]
    straight_pos[:, 1] += correction[1]
    straight_pos[:, 2] += correction[2]

    # Define properties of the rod
    n_vertices, n_edges = import_pos.shape[0], import_theta.shape[0]

    # Just set the mass to 1 for now
    mass = np.ones(n_vertices) * 1

    # For each edge, the bending stiffness matrix
    B = np.zeros((n_edges, 2, 2))
    for i in range(n_edges):
        B[i, 0, 0] = 1.0
        B[i, 1, 1] = 1.0

    # Which nodes/edges are frozen
    frozen_pos_indices = np.array([0])
    frozen_theta_indices = np.array([0])

    # Twisting stiffness
    beta = 0.1

    # Stretching stiffness (unused)
    k = 0.0

    # Scale down gravity for now
    g = 9.81 * 1e-3
    # g = 0

    # Simulation parameters (damping for integration, time step, and number of XPBD steps)
    damping = 0.2
    dt = 0.04
    xpbd_steps = 10

    # Visualization parameters
    point_scale = 0.1
    edge_scale = 0.05
    point_radii = np.cbrt(mass) * point_scale
    ax1_radii = np.sqrt(B[:, 0, 0]) * edge_scale
    ax2_radii = np.sqrt(B[:, 1, 1]) * edge_scale
    point_style = ["cube"] + ["sphere"] * (n_vertices - 1)

    energies = [Gravity(), Twist(), Bend(), BendTwist()]

    # Initialize with the imported rod, create a frame
    sim = Sim(pos=import_pos, theta=import_theta, frozen_pos_indices=frozen_pos_indices,
              frozen_theta_indices=frozen_theta_indices, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies,
              damping=damping, dt=dt, xpbd_steps=xpbd_steps)
    forces = np.zeros_like(straight_pos)
    create_frame(pos=import_pos, material_frame=sim.state.material_frame, point_radii=point_radii, forces=forces,
                 ax1_radii=ax1_radii, ax2_radii=ax2_radii, point_style=point_style, i=0)

    start = time.time()

    # --- Run the optimization ---
    target_pos, target_theta = import_pos.copy(), import_theta.copy()
    total_length = np.sum(np.linalg.norm(target_pos[1:] - target_pos[:-1], axis=1))

    # Draw the optimized rest state
    pos_rest, theta_rest = import_pos.copy(), import_theta.copy()
    create_frame(pos=pos_rest, material_frame=sim.state.material_frame, point_radii=point_radii,
                 forces=np.zeros_like(target_pos), ax1_radii=ax1_radii, ax2_radii=ax2_radii, point_style=point_style,
                 i=2)

    def project_out_force(f: np.ndarray, p: np.ndarray) -> np.ndarray:
        # Zero out force of first node (fixed)
        f[0] = 0.0
        # Project out the force in direction of the edges (inextensibility)
        e = p[1:] - p[:-1]
        e /= np.linalg.norm(e, axis=1)[:, None]
        f[1:] -= np.sum(f[1:] * e, axis=1)[:, None] * e
        return f

    # Solve the minimization in reduced coordinates
    r, polar = Reduced.to_polar_coordinates(target_pos)
    def obj(p: np.ndarray, t: np.ndarray):
        pos_rest_k = Reduced.to_cartesian_coordinates(r, p.reshape(-1, 2), target_pos[0])
        sim.define_rest_state(pos=pos_rest_k, theta=t)
        f = sim.compute_force(pos_test=target_pos, theta_test=target_theta)
        # Zero out force of first node
        f[0] = 0.0
        # Project out the force in direction of the edges
        e = target_pos[1:] - target_pos[:-1]
        e /= np.linalg.norm(e, axis=1)[:, None]
        f[1:] -= np.sum(f[1:] * e, axis=1)[:, None] * e
        return np.linalg.norm(f)

    counter = 3
    def callback(xk):
        nonlocal counter
        print(f"Force: {obj(xk, target_theta)}, Counter: {counter}")
        pos_rest_k = Reduced.to_cartesian_coordinates(r, xk.reshape(-1, 2), target_pos[0])
        sim.define_rest_state(pos=pos_rest_k, theta=target_theta)
        forces = sim.compute_force(pos_test=target_pos, theta_test=target_theta)
        forces = project_out_force(forces, target_pos)
        create_frame(pos=pos_rest_k, material_frame=sim.state.material_frame, point_radii=point_radii,
                     forces=forces, ax1_radii=ax1_radii, ax2_radii=ax2_radii, point_style=point_style, i=counter)
        counter += 1
        return

    from scipy.optimize import minimize
    res = minimize(obj, x0=polar.ravel(), method='bfgs', tol=1e-8, args=(target_theta,), options={'disp': True},
                   callback=callback)
    polar_optimized = res.x
    pos_optimized = Reduced.to_cartesian_coordinates(r, polar_optimized.reshape(-1, 2), target_pos[0])

    # Draw the forces at the optimized rest state
    force = obj(polar_optimized, target_theta)
    print(f"Force magnitude: {force}")

    print(f"Time: {time.time() - start:.2f}s")
    # Run the simulation
    pos, theta = pos_optimized, target_theta
    sim.define_rest_state(pos=pos, theta=theta)
    save_freq = 10
    progress = tqdm(range(save_freq, 10000))
    for i in progress:
        pos, theta = sim.step(pos=pos, theta=theta)
        if i % save_freq == 0:
            create_frame(pos=pos, material_frame=sim.state.material_frame, point_radii=point_radii,
                         forces=sim.analytics.force, ax1_radii=ax1_radii, ax2_radii=ax2_radii, point_style=point_style,
                         i=i // save_freq + counter)
            progress.set_description(f"Frame {i // save_freq + counter}")

    plt.show()
    return


if __name__ == "__main__":
    main()
