import matplotlib.pyplot as plt
import numpy as np

from energies.bend_twist import BendTwist
from energies.bending import Bend
from energies.gravity import Gravity
from energies.twisting import Twist
from rod.rod_generator import RodGenerator
from solver.sim import Sim
from visualization.visualizer import Visualizer


def create_frame(rod, i):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    Visualizer.draw_nodes(rod=rod, ax=ax)
    Visualizer.draw_edges(rod=rod, ax=ax)
    # Visualizer.draw_material_frame(rod, ax)
    Visualizer.set_lims(rod=rod, ax=ax)
    plt.savefig(f"frames/frame_{i}.png")
    plt.close()


def main():
    # Create a rod
    n_points = 10
    rod = RodGenerator.example_rod(n_points)

    # Elastic properties
    n_vertices, n_edges = rod.n + 2, rod.n + 1
    B = np.zeros((n_edges, 2, 2))
    for i in range(n_edges):
        B[i] = np.eye(2) * 0.1
    mass = np.ones(n_vertices) / n_vertices
    beta = 0.1
    k = 0.0
    dt = 0.04
    xpbd_steps = 5
    g = 9.81

    # rod.theta *= 0.1
    # rod.pos *= 1.2
    create_frame(rod, 0)

    energies = [Bend(), Twist(), BendTwist(), Gravity()]
    sim = Sim(rod, B, beta, k, g, mass, energies, dt, xpbd_steps)
    for i in range(1000):
        if i % 50 == 0:
            print(f"iteration {i}")
            create_frame(rod, i)
        sim.step()


    plt.show()
    return


if __name__ == "__main__":
    main()
