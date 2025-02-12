import numpy as np
from tqdm import tqdm

from energies.bend import Bend
from energies.bend_twist import BendTwist
from energies.gravity import Gravity
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
                 frame_idx: int):
    Visualizer.strand_to_obj(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                             ax2_radii=ax2_radii, point_style=point_style,
                             output_file=f"output/obj/obj_{frame_idx}.obj")
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
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=frame_idx)
    return


def main():
    # Import rod
    import_pos, import_theta = RodGenerator.from_obj(file_path="../../blender/sarah_1.obj", scale=10)
    import_bishop_frame = RodUtil.compute_bishop_frames(pos=import_pos)
    import_material_frame = RodUtil.compute_material_frames(theta=import_theta, bishop_frame=import_bishop_frame)

    n_pts = import_pos.shape[0]
    # Stiffness, mass constants. Revisit this
    rhoS = 0.05
    g = 9.81 * 1e-3
    # Drawing parameters
    point_radii = 0.1 * np.ones(n_pts)
    ax1_radii = 0.2 * np.ones(n_pts)
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

    # Back to DER (target)
    target_pos, target_theta = RodHelixConverter.helix_to_rod(helix)
    target_bishop_frame = RodUtil.compute_bishop_frames(pos=target_pos)
    target_material_frame = RodUtil.compute_material_frames(theta=target_theta, bishop_frame=target_bishop_frame)
    create_frame(pos=target_pos, material_frame=target_material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=2)
    print("Frame 2: DER Target")

    # Compute the rest shape
    K_inv = HelixUtil.compute_inv_pointwise_stiffness_matrix(helix)
    B = HelixUtil.compute_gen_gravity_force(helix, g=g, rhoS=rhoS)
    q_target = helix.q.copy()
    q_rest = q_target[3:] - K_inv @ B
    q_rest = np.concatenate([q_target[:3], q_rest])

    # Update helix to have rest shape
    helix.q0 = q_rest.copy()
    helix.q = q_rest
    create_frame_helix(helix, point_radii, ax1_radii, ax2_radii, point_style, frame_idx=3)
    print("Frame 3: Helix rest shape")

    # Convert back to DER for simulation
    rest_pos, rest_theta = RodHelixConverter.helix_to_rod(helix)
    bishop_frame = RodUtil.compute_bishop_frames(pos=rest_pos)
    material_frame = RodUtil.compute_material_frames(theta=rest_theta, bishop_frame=bishop_frame)
    create_frame(pos=rest_pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=4)
    print("Frame 4: DER rest shape")

    # Simulate [for now, set current shape as target shape]
    pos, theta = target_pos, target_theta
    n_edges = import_theta.shape[0]
    B = np.zeros((n_edges, 2, 2))
    for i in range(n_edges):
        B[i, 0, 0] = 1.0
        B[i, 1, 1] = 1.0
    mass = np.ones(n_pts)

    # Twisting stiffness
    beta = 0.0
    k = 0.0
    g = 9.81 * 1e-3

    # Simulation parameters (damping for integration, time step, and number of XPBD steps)
    damping = 0.2
    dt = 0.04
    xpbd_steps = 10
    frozen_pos_indices = np.array([0])
    frozen_theta_indices = np.array([0])

    energies = [Twist(), Bend(), BendTwist()]
    sim = Sim(pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies, damping=damping,
              dt=dt, xpbd_steps=xpbd_steps, frozen_pos_indices=frozen_pos_indices,
              frozen_theta_indices=frozen_theta_indices)
    sim.define_rest_state(rest_pos, rest_theta)

    sim.update_analytics(pos, theta)
    der_energy = sim.analytics.potential_energy


if __name__ == "__main__":
    main()
