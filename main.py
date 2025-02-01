import matplotlib.pyplot as plt
import time
from tqdm import tqdm

from energies.bend_twist import BendTwist
from energies.bend import Bend
from energies.gravity import Gravity
from energies.twist import Twist
from rod.rod_generator import RodGenerator
from solver.sim import Sim
from solver.solver_params import update_csv_analytics
from visualization.visualizer import Visualizer

import jax.numpy as np


def create_frame(pos: np.ndarray, theta: np.ndarray, material_frame: np.ndarray, i: int):
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # Visualizer.draw_nodes(pos=pos, ax=ax)
    # Visualizer.draw_edges(pos=pos, ax=ax)
    # # if bishop_frame is not None:
    # #     Visualizer.draw_material_frame(pos=pos, theta=theta, bishop_frame=bishop_frame, ax=ax)
    # Visualizer.set_lims(pos=pos, ax=ax)
    # plt.title(f"t={i * 0.04:.2f}")
    # plt.savefig(f"output/frames/frame_{i}.png")
    # plt.close()

    # Visualizer.to_obj(pos=pos, output_file=f"output/obj/obj_{i}.obj")
    Visualizer.strand_to_obj(pos=pos, material_frame=material_frame, output_file=f"output/obj/obj_{i}.obj",
                             point_radius=0.1, axis_1_radius=0.2, axis_2_radius=0.1)
    return


def main():
    # Create a rod
    n_points = 30
    pos, theta = RodGenerator.example_rod(n_points)

    # Elastic properties
    n_vertices, n_edges = pos.shape[0], theta.shape[0]
    # For each edge, the bending stiffness matrix
    B = np.zeros((n_edges, 2, 2))
    for i in range(n_edges):
        B = B.at[i, 0, 0].set(4.0)
        B = B.at[i, 1, 1].set(1.0)

    # Twisting stiffness
    beta = 0.1

    # Just set the mass to 1 for now
    mass = np.ones(n_vertices)

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

    create_frame(pos, theta, sim.state.bishop_frame, 0)
    sim.update_analytics(pos, theta)
    update_csv_analytics(sim.analytics, f"output/analytics.csv")

    start = time.time()

    # Straighten the rod (keeping edge lengths constant)
    edge_lengths = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    for i in range(1, n_points + 2):
        pos.at[i].set(np.array([0, 0, pos[0, 2] - np.sum(edge_lengths[:i])], dtype=np.float64))

    create_frame(pos, theta, sim.state.material_frame, 1)

    # Run the simulation
    save_freq = 10
    progress = tqdm(range(2 * save_freq, 10000))
    for i in progress:
        if i % save_freq == 0:
            create_frame(pos, theta, sim.state.material_frame, i // save_freq)
            progress.set_description(f"Frame {i // save_freq}")
            update_csv_analytics(sim.analytics, f"output/analytics.csv")
        pos, theta = sim.step(pos=pos, theta=theta)
    print(f"Time: {time.time() - start:.2f}s")

    plt.show()
    return


if __name__ == "__main__":
    main()
