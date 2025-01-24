import matplotlib.pyplot as plt
import numpy as np

from energies.bend_twist import BendTwist
from energies.bending import Bend
from energies.gravity import Gravity
from energies.stretch import Stretch
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
    ax.set_xlim([-rod.n / 2, rod.n / 2])
    ax.set_ylim([-rod.n / 2, rod.n / 2])
    ax.set_zlim([0, rod.n + 1])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
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
        B[i] = np.eye(2)
    mass = np.ones(n_vertices) / n_vertices * 0.1
    beta = 0.1
    k = 25.0
    dt = 0.05
    g = 9.81

    # rod.theta *= 0.1
    # rod.pos *= 1.2
    create_frame(rod, 0)

    energies = [Gravity()]
    p_top = rod.pos[-1]
    Sim.init(rod, B, beta, k, g, p_top, mass, energies, dt)
    for i in range(500):
        if i % 10 == 0:
            print(f"iteration {i}")
            create_frame(rod, i)
        Sim.step(rod, B, beta, k, g, p_top, mass, energies, dt)


    plt.show()
    return


if __name__ == "__main__":
    main()
