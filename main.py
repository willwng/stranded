import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm

from energies.bend_twist import BendTwist
from energies.bend import Bend
from energies.gravity import Gravity
from energies.twist import Twist
from rod.rod_generator import RodGenerator
from solver.sim import Sim
from solver.solver_params import update_csv_analytics
from visualization.visualizer import Visualizer


def create_frame(pos: np.ndarray,
                 material_frame: np.ndarray,
                 point_radii: np.ndarray,
                 ax1_radii: np.ndarray,
                 ax2_radii: np.ndarray,
                 point_style: list[str],
                 i: int):
    Visualizer.strand_to_obj(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                             ax2_radii=ax2_radii, point_style=point_style, output_file=f"output/obj/obj_{i}.obj")
    return


def main():
    # Create a rod
    n_points = 50
    pos, theta = RodGenerator.example_rod(n_points)

    # Define properties of the rod
    n_vertices, n_edges = pos.shape[0], theta.shape[0]

    # Just set the mass to 1 for now
    mass = np.ones(n_vertices) * 1

    # For each edge, the bending stiffness matrix
    B = np.zeros((n_edges, 2, 2))
    for i in range(n_edges):
        B[i, 0, 0] = 1.0
        B[i, 1, 1] = 1.0

    # Visualization parameters
    point_scale = 0.1
    edge_scale = 0.05
    point_radii = np.cbrt(mass) * point_scale
    ax1_radii = np.sqrt(B[:, 0, 0]) * edge_scale
    ax2_radii = np.sqrt(B[:, 1, 1]) * edge_scale
    point_style = ["cube"] + ["sphere"] * (n_vertices - 1)

    # Twisting stiffness
    beta = 0.1

    # Stretching stiffness (unused)
    k = 0.0

    # Scale down gravity for now
    g = 9.81 * 1e-3

    # Simulation parameters (damping for integration, time step, and number of XPBD steps)
    damping = 0.2
    dt = 0.04
    xpbd_steps = 10

    energies = [Gravity(), Twist(), Bend(), BendTwist()]
    sim = Sim(pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies,
              damping=damping, dt=dt, xpbd_steps=xpbd_steps)

    material_frame = sim.state.material_frame
    create_frame(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, i=0)
    sim.update_analytics(pos, theta)
    update_csv_analytics(sim.analytics, f"output/analytics.csv")

    start = time.time()

    # Straighten the rod (keeping edge lengths constant)
    # edge_lengths = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    # for i in range(1, n_points + 2):
    #     pos[i] = np.array([0, 0, pos[0, 2] - np.sum(edge_lengths[:i])], dtype=np.float64)

    create_frame(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, i=1)

    # Run the simulation
    save_freq = 10
    progress = tqdm(range(2 * save_freq, 10000))
    for i in progress:
        if i % save_freq == 0:
            create_frame(pos=pos, material_frame=sim.state.material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                         ax2_radii=ax2_radii, point_style=point_style, i=i // save_freq)
            progress.set_description(f"Frame {i // save_freq}")
            update_csv_analytics(sim.analytics, f"output/analytics.csv")
        pos, theta = sim.step(pos=pos, theta=theta)
    print(f"Time: {time.time() - start:.2f}s")

    plt.show()
    return


if __name__ == "__main__":
    main()
