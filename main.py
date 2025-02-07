import matplotlib.pyplot as plt
import time
import numpy as np
from tqdm import tqdm

from energies.bend_twist import BendTwist
from energies.bend import Bend
from energies.gravity import Gravity
from energies.twist import Twist
from rod.RodHelixConverter import RodHelixConverter
from rod.helix import Helix
from rod.helix_util import HelixUtil
from rod.rod_generator import RodGenerator
from rod.rod_util import RodUtil
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
    n_pts = 41  # Including index 0
    L = 50
    s = np.linspace(0, L, n_pts)
    # Generalized coordinates
    tau = np.ones(n_pts) * 0.3
    k_1 = np.ones(n_pts) * 0.01
    k_2 = np.ones(n_pts) * 0.1
    q = np.stack([tau, k_1, k_2], axis=1).ravel()
    # Boundary/initial conditions
    r0 = np.array([0, 0, L])
    n0 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    # Stiffness, mass constants. Revisit this
    EI = np.ones(3 * n_pts) * 1
    rhoS = 0.05
    g = 9.81 * 1e-3

    # Drawing parameters
    point_radii = 0.1 * np.ones(n_pts)
    ax1_radii = 0.2 * np.ones(n_pts)
    ax2_radii = 0.1 * np.ones(n_pts)
    point_style = ["sphere"] * n_pts

    # Draw the helix
    helix = Helix(q=q, q0=q.copy(), n_sites=n_pts, s=s, L=L, r0=r0, n0=n0, EI=EI)
    r, n = HelixUtil.propagate(helix)
    create_frame(pos=r, material_frame=n[:, 1:], point_radii=point_radii, ax1_radii=ax1_radii, ax2_radii=ax2_radii,
                 point_style=point_style, i=0)

    # Compute the rest state
    K = HelixUtil.compute_stiffness_matrix(helix)
    K_inv = HelixUtil.compute_inv_stiffness_matrix(helix)
    B = HelixUtil.compute_gen_gravity_force(helix, g=g, rhoS=rhoS)
    q_target = q.copy()
    q_rest = q_target[3:] - K_inv @ B
    q_rest = np.concatenate([q_target[:3], q_rest])

    # Set and draw the rest state
    # helix.q = q_rest
    helix.q0 = q_rest
    r, n = HelixUtil.propagate(helix)
    create_frame(pos=r, material_frame=n[:, 1:], point_radii=point_radii, ax1_radii=ax1_radii, ax2_radii=ax2_radii,
                 point_style=point_style, i=1)

    # Convert to DER
    pos, theta = RodHelixConverter.helix_to_rod(helix)
    bishop_frame = np.zeros((theta.shape[0], 2, 3))
    bishop_frame = RodUtil.update_bishop_frames(pos=pos, bishop_frame=bishop_frame)
    material_frame = RodUtil.compute_material_frames(theta=theta, bishop_frame=bishop_frame)
    create_frame(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii, ax2_radii=ax2_radii,
                 point_style=point_style, i=2)


    q_back = RodHelixConverter.rod_to_helix(pos, theta)
    q_prev = helix.q.copy()
    helix.q = q_back
    r, n = HelixUtil.propagate(helix)
    create_frame(pos=r, material_frame=n[:, 1:], point_radii=point_radii, ax1_radii=ax1_radii, ax2_radii=ax2_radii,
                 point_style=point_style, i=3)
    helix.q = q_prev
    quit()


    # Simulate the helix
    q = helix.q
    v = np.zeros(3 * n_pts)
    dt = 0.04
    for i in range(3, 500):
        print(f"Frame {i}")
        internal_force = -K @ (q[3:] - q_rest[3:])
        external_force = HelixUtil.compute_gen_gravity_force(helix, g=g, rhoS=rhoS)
        B = internal_force + external_force
        v[3:] += dt * B - 0.1 * v[3:]
        q[3:] += dt * v[3:]
        helix.q = q
        # Print average z value of r
        r, n = HelixUtil.propagate(helix)
        create_frame(pos=r, material_frame=n[:, 1:], point_radii=point_radii, ax1_radii=ax1_radii, ax2_radii=ax2_radii,
                     point_style=point_style, i=i)
    # print(n)


if __name__ == "__main__":
    # main()
    helix()
