import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm

from energies.bend_twist import BendTwist
from energies.bend import Bend
from energies.gravity import Gravity
from energies.twist import Twist
from rod.helix import Helix
from rod.helix_util import HelixUtil
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


def helix():
    n_pts = 51  # Including index 0
    L = 50
    radius = 10
    tau = np.ones(n_pts) * 0.5
    k_1 = np.ones(n_pts) * 0.1
    k_2 = np.ones(n_pts) * 0.0
    q = np.stack([tau, k_1, k_2], axis=1).ravel()
    s = np.linspace(0, L, n_pts)
    r0 = np.zeros(3)
    n0 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    # Revisit this
    EI = np.ones(3 * n_pts)

    helix = Helix(q=q, q0=q.copy(), n_elems=n_pts, s=s, L=L, r0=r0, n0=n0, EI=EI)
    r, n = HelixUtil.propagate(helix)

    U = HelixUtil.compute_internal_potential_loop(helix)
    U2 = HelixUtil.compute_internal_potential(helix)
    Ug = HelixUtil.compute_gravity_potential_pos(helix, r, g=9.81, rhoS=1.0)
    print(Ug)
    create_frame(pos=r, material_frame=n[:, 1:], point_radii=0.0 * np.ones(n_pts), ax1_radii=0.2 * np.ones(n_pts),
                 ax2_radii=0.1 * np.ones(n_pts), point_style=["sphere"] * n_pts, i=0)
    # print(n)


if __name__ == "__main__":
    # main()
    helix()
