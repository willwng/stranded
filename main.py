import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from energies.bend import Bend
from energies.bend_twist import BendTwist
from energies.gravity import Gravity
from energies.random import RandomForce
from energies.twist import Twist
from math_util.rotation import RotationUtil
from rod.RodHelixConverter import RodHelixConverter
from rod.helix import Helix
from rod.helix_util import HelixUtil
from rod.rod_generator import RodGenerator
from rod.rod_util import RodUtil
from solver.sim import Sim
from visualization.visualizer import Visualizer


def create_frame(pos: np.ndarray,
                 material_frame: np.ndarray,
                 point_radii: np.ndarray,
                 ax1_radii: np.ndarray,
                 ax2_radii: np.ndarray,
                 point_style: list[str],
                 frame_idx: int,
                 site_material_frames: np.ndarray = None,
                 draw_arrows: bool = False):
    Visualizer.strand_to_obj(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                             ax2_radii=ax2_radii, point_style=point_style,
                             output_file=f"output/obj/obj_{frame_idx}.obj",
                             site_material_frames=site_material_frames,
                             draw_arrows=draw_arrows)
    return


def create_frame_helix(helix: Helix, point_radii: np.ndarray, ax1_radii: np.ndarray, ax2_radii: np.ndarray,
                       point_style: list[str], frame_idx: int):
    r, n = HelixUtil.propagate(helix)
    # Interpolate the material frame between the two sites
    material_frame = np.zeros((n.shape[0] - 1, 2, 3))
    for i in range(n.shape[0] - 1):
        rotation = RotationUtil.compute_rotation_matrix(n[i], n[i + 1])
        rotation = RotationUtil.interpolate_rotation(rotation, 0.5)
        interpolated_frame = rotation @ n[i]
        material_frame[i] = interpolated_frame[1:]

    create_frame(pos=r, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=frame_idx,
                 site_material_frames=n)
    return


def plot_generalized_coords(helix: Helix):
    plt.figure()
    twist = helix.q[::3]
    bend1 = helix.q[1::3]
    bend2 = helix.q[2::3]
    i = np.arange(0, helix.n_sites)
    plt.plot(i, twist, label="Twist")
    plt.plot(i, bend1, label="Bend 1")
    plt.plot(i, bend2, label="Bend 2")
    plt.legend()
    plt.show()


def main():
    # Import rod
    import_pos, import_theta = RodGenerator.from_obj(file_path="sarah_1.obj", scale=9.75)
    import_bishop_frame = RodUtil.compute_bishop_frames(pos=import_pos)
    import_material_frame = RodUtil.compute_material_frames(theta=import_theta, bishop_frame=import_bishop_frame)

    n_pts = import_pos.shape[0]
    # Drawing parameters
    point_radii = 0.1 * np.ones(n_pts)
    ax1_radii = 0.1 * np.ones(n_pts)
    ax2_radii = 0.1 * np.ones(n_pts)
    point_style = ["sphere"] * n_pts

    # ----- Begin ----- #
    create_frame(pos=import_pos, material_frame=import_material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=0)
    print("Frame 0: DER Imported rod")

    # Convert to helix
    helix = RodHelixConverter.rod_to_helix(import_pos, import_theta)
    create_frame_helix(helix, point_radii, ax1_radii, ax2_radii, point_style, frame_idx=1)
    print("Frame 1: Helix Target")
    plot_generalized_coords(helix)


    # Back to DER (target)
    pos_target, theta_target = RodHelixConverter.helix_to_rod(helix)
    target_bishop_frame = RodUtil.compute_bishop_frames(pos=pos_target)
    target_material_frame = RodUtil.compute_material_frames(theta=theta_target, bishop_frame=target_bishop_frame)
    create_frame(pos=pos_target, material_frame=target_material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=2)
    print("Frame 2: DER Target")

    # Stiffness, mass constants. Revisit this
    mass = np.ones(n_pts) * 1.0
    rhoS = np.sum(mass) / helix.L
    g = 9.81 * 1e-3

    forces = HelixUtil.compute_random_force(pos_target, seed=0) + Gravity().compute_forces(pos_target, mass, g)
    forces = np.tile(forces[:, np.newaxis, :], (1, 2, 1))
    print(forces)
    create_frame(pos=pos_target, material_frame=forces, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=2, draw_arrows=True)

    # Compute the rest shape
    K_inv = HelixUtil.compute_inv_pointwise_stiffness_matrix(helix)
    B_gen = HelixUtil.compute_gen_gravity_force(helix, g=g, rhoS=rhoS)
    q_target = helix.q.copy()
    q_rest = q_target[3:] - K_inv @ B_gen
    q_rest = np.concatenate([q_target[:3], q_rest])

    # Update helix to have rest shape
    helix.q0 = q_rest.copy()
    helix.q = q_rest
    create_frame_helix(helix, point_radii, ax1_radii, ax2_radii, point_style, frame_idx=3)
    print("Frame 3: Helix rest shape")

    plot_generalized_coords(helix)

    # Convert back to DER for simulation
    rest_pos, rest_theta = RodHelixConverter.helix_to_rod(helix)
    bishop_frame = RodUtil.compute_bishop_frames(pos=rest_pos)
    material_frame = RodUtil.compute_material_frames(theta=rest_theta, bishop_frame=bishop_frame)
    create_frame(pos=rest_pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=4)
    print("Frame 4: DER rest shape")

    # Simulate [for now, set current shape as target shape]
    # pos, theta = pos_target, theta_target
    pos, theta = rest_pos, rest_theta
    n_edges = import_theta.shape[0]
    B = np.zeros((n_edges, 2, 2))
    for i in range(n_edges):
        B[i, 0, 0] = 1.0
        B[i, 1, 1] = 1.0

    # Twisting stiffness
    beta = 1.0
    k = 0.0

    # Simulation parameters (damping for integration, time step, and number of XPBD steps)
    damping = 0.05
    dt = 0.1
    xpbd_steps = 10
    frozen_pos_indices = np.array([0], dtype=int)
    frozen_theta_indices = np.array([], dtype=int)

    energies = [Twist(), Bend(), BendTwist(), Gravity(), RandomForce(seed=0)]
    sim = Sim(pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies, damping=damping,
              dt=dt, xpbd_steps=xpbd_steps, frozen_pos_indices=frozen_pos_indices,
              frozen_theta_indices=frozen_theta_indices)
    sim.define_rest_state(rest_pos, rest_theta)
    save_freq = 10
    progress = tqdm(range(5 * save_freq, 10000))
    for i in progress:
        if i % save_freq == 0:
            create_frame(pos=pos, material_frame=sim.state.material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                         ax2_radii=ax2_radii, point_style=point_style, frame_idx=i // save_freq)
            progress.set_description(f"Frame {i // save_freq}")
        pos, theta = sim.step(pos=pos, theta=theta)


    # Save final pos to npy
    np.save("output/final_pos.npy", pos)



if __name__ == "__main__":
    main()
