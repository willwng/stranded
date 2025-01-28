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
    Visualizer.strand_to_obj(pos=pos, output_file=f"output/obj/obj_{i}.obj", point_radius=0.2, line_radius=0.2)
    return


def main():
    # Create a rod
    n_points = 10
    pos, theta = RodGenerator.example_rod(n_points)

    # Elastic properties
    n_vertices, n_edges = pos.shape[0], theta.shape[0]
    B = np.zeros((n_edges, 2, 2))
    for i in range(n_edges):
        B[i] = np.eye(2) * 20
    mass = np.ones(n_vertices) * 0.05
    beta = 1.0
    k = 0.0
    dt = 0.02
    xpbd_steps = 5
    g = 0

    create_frame(pos, theta, None, 0)

    energies = [Gravity(), Twist(), Bend(), BendTwist()]
    sim = Sim(pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies, dt=dt,
              xpbd_steps=xpbd_steps)

    start = time.time()

    # Straighten test (keeping edge lengths constant)
    edge_lengths = np.linalg.norm(pos[1:] - pos[:-1], axis=1)
    for i in range(1, n_points + 2):
        pos[i] = np.array([0, 0, pos[0, 2] - np.sum(edge_lengths[:i])], dtype=np.float64)

    create_frame(pos, theta, sim.state.bishop_frame, 1)

    save_freq = 100
    for i in tqdm(range(10000)):
        if i % save_freq == 0:
            create_frame(pos, theta, sim.state.bishop_frame, i // save_freq)
        pos, theta = sim.step(pos=pos, theta=theta)
    print(f"Time: {time.time() - start:.2f}s")

    plt.show()
    return


if __name__ == "__main__":
    main()
