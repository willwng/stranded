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
from visualization.visualizer import Visualizer


def create_frame(pos, theta, bishop_frame, i):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Visualizer.draw_nodes(pos=pos, ax=ax)
    Visualizer.draw_edges(pos=pos, ax=ax)
    # if bishop_frame is not None:
    #     Visualizer.draw_material_frame(pos=pos, theta=theta, bishop_frame=bishop_frame, ax=ax)
    Visualizer.set_lims(pos=pos, ax=ax)
    plt.title(f"t={i * 0.04:.2f}")
    plt.savefig(f"output/frames/frame_{i}.png")
    plt.close()

    # Visualizer.to_obj(pos=pos, output_file=f"output/obj/obj_{i}.obj")
    Visualizer.strand_to_obj(pos=pos, output_file=f"output/obj/obj_{i}.obj", point_radius=0.3, line_radius=0.4)
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
        B[i, 0, 0] = 1.0
        B[i, 1, 1] = 1.0

    # Twisting stiffness
    beta = 0.1

    # Just set the mass to 1 for now
    mass = np.ones(n_vertices)

    # Stretching stiffness (unused)
    k = 0.0

    # Scale down gravity for now
    g = 9.81 * 1e-5

    # Simulation parameters (damping for integration, time step, and number of XPBD steps)
    damping = 0.2
    dt = 0.02
    xpbd_steps = 10

    create_frame(pos, theta, None, 0)

    energies = [Gravity(), Twist(), Bend(), BendTwist()]
    sim = Sim(pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies,
              damping=damping, dt=dt, xpbd_steps=xpbd_steps)

    start = time.time()

    # Straighten the rod (keeping edge lengths constant)
    edge_lengths = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    for i in range(1, n_points + 2):
        pos[i] = np.array([0, 0, pos[0, 2] - np.sum(edge_lengths[:i])], dtype=np.float64)

    create_frame(pos, theta, sim.state.bishop_frame, 1)

    # Run the simulation
    save_freq = 30
    progress = tqdm(range(2 * save_freq, 10000))
    for i in progress:
        if i % save_freq == 0:
            create_frame(pos, theta, sim.state.bishop_frame, i // save_freq)
            progress.set_description(f"Frame {i // save_freq}")
        pos, theta = sim.step(pos=pos, theta=theta)
    print(f"Time: {time.time() - start:.2f}s")

    plt.show()
    return


if __name__ == "__main__":
    main()
