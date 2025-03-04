import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from scipy.cluster.hierarchy import fclusterdata
from tqdm import tqdm
import concurrent.futures

from energies.bend import Bend
from energies.bend_twist import BendTwist
from energies.gravity import Gravity
from energies.random import RandomForce
from energies.twist import Twist
from math_util.rotation import RotationUtil, Quaternion
from math_util.vectors import Vector
from rod.RodHelixConverter import RodHelixConverter
from rod.helix import Helix
from rod.helix_util import HelixUtil
from rod.preprocess import Preprocess
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
                 output_file: str = None):
    output_file = f"output/obj/obj_{frame_idx}.obj" if output_file is None else output_file
    Visualizer.strand_to_obj(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                             ax2_radii=ax2_radii, point_style=point_style,
                             output_file=output_file)
    return


def strands_to_one_objs(strands: np.ndarray, frame_idx: int, output_file: str = None, y_up: bool = True):
    output_file = f"output/obj/obj_{frame_idx}.obj" if output_file is None else output_file
    Visualizer.clear_output_file(output_file)
    vertex_offset = 1
    for strand in strands:
        pos = strand[:, :3]
        vertex_offset = Visualizer.to_simple_obj(pos=pos, output_file=output_file, init_offset=vertex_offset, y_up=y_up)
    return


def helices_to_one_obj(helices: list[Helix], frame_idx: int, output_file: str = None):
    output_file = f"output/obj/obj_{frame_idx}.obj" if output_file is None else output_file
    Visualizer.clear_output_file(output_file)
    vertex_offset = 1
    for helix in helices:
        r, n = HelixUtil.propagate(helix)
        vertex_offset = Visualizer.to_simple_obj(pos=r, output_file=output_file, init_offset=vertex_offset)
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
    # import scienceplots
    # plt.style.use(["science", "nature"])
    plt.figure()
    twist = helix.q[::3]
    bend1 = helix.q[1::3]
    bend2 = helix.q[2::3]
    # bend1[0] = 0
    i = np.arange(0, helix.n_sites)
    plt.plot(i, twist, label="Twist")
    plt.plot(i, bend1, label="Bend 1")
    plt.plot(i, bend2, label="Bend 2")
    plt.xlabel("Node Index")
    plt.xticks([0, helix.n_sites // 2, helix.n_sites])

    plt.legend()
    plt.show()

def decimate_strands(strands: np.ndarray, n_sites: int):
    decimated_strands = np.zeros((strands.shape[0], n_sites, strands.shape[2]))
    for i in range(strands.shape[0]):
        strand = strands[i]
        n_sites_strand = strand.shape[0]
        idx = np.linspace(0, n_sites_strand - 1, n_sites, dtype=int)
        decimated_strands[i] = strand[idx]
    return decimated_strands

def main():
    scalp_strands = np.load("me.npy") * 2e0
    scalp_strands = scalp_strands[:]
    # fix y up
    scalp_strands = scalp_strands[:, :, [0, 2, 1]]
    scalp_strands[:, :, 1] *= -1
    mean_scalp_pos = np.mean(scalp_strands, axis=(0, 1))
    scalp_strands -= mean_scalp_pos
    new_strands = []
    for strand in scalp_strands:
        strand = RodGenerator.add_vertices_until(strand, 128)
        new_strands.append(strand)
    new_strands = np.array(new_strands)
    scalp_strands = new_strands
    np.save("me_2.npy", new_strands)


    # ind_keep = np.random.choice(scalp_strands.shape[0], scalp_strands.shape[0] // 30, replace=False)
    # scalp_strands = scalp_strands[ind_keep]
    # scalp_strands = decimate_strands(scalp_strands, 128)
    synthetic_strands = np.load("synthetic_strands.npy")
    strands_to_one_objs(scalp_strands, frame_idx=0, y_up=False)
    strands_to_one_objs(synthetic_strands, frame_idx=1)
    synthetic_strands = scalp_strands
    # quit()
    # quit()

    # Make sure the number of strands are the same
    n_strands = min(scalp_strands.shape[0], synthetic_strands.shape[0])
    scalp_strands, synthetic_strands = scalp_strands[:n_strands], synthetic_strands[:n_strands]
    #
    # # Create initial positions and directions of scalp strands
    # r0 = scalp_strands[:, 0]
    # n0 = np.zeros((scalp_strands.shape[0], 3, 3))
    # tangents = scalp_strands[:, 1] - scalp_strands[:, 0]
    # for i in range(n0.shape[0]):
    #     t = tangents[i]
    #     u = Vector.compute_orthogonal_vec(t)
    #     v = np.cross(t, u)
    #     n0[i, 0] = t / np.linalg.norm(t)
    #     n0[i, 1] = u / np.linalg.norm(u)
    #     n0[i, 2] = v / np.linalg.norm(v)
    #
    # # Align the synthetic strands to the scalp strands
    # aligned_synthetic_strands = synthetic_strands.copy()
    # for i in range(n_strands):
    #     strand = aligned_synthetic_strands[i]
    #     # Root position, direction
    #     strand_root = strand[0]
    #     strand_dir = strand[-1] - strand_root
    #     strand_dir = strand_dir / np.linalg.norm(strand_dir)
    #     # Align the strand to the scalp strand
    #     target_dir = n0[i, 0]
    #     rot_axis = np.cross(strand_dir, target_dir)
    #     rot_axis = rot_axis / np.linalg.norm(rot_axis)
    #     rot_angle = np.arccos(np.dot(strand_dir, target_dir))
    #     P_i = Quaternion.from_angle_axis(rot_angle, rot_axis)
    #     P_i.normalize()
    #     # Translate and rotate the strand
    #     strand -= strand_root
    #     for j in range(strand.shape[0]):
    #         strand[j] = P_i.rotate_vec(strand[j])
    #     strand += r0[i]
    # strands_to_one_objs(aligned_synthetic_strands, frame_idx=2)
    aligned_synthetic_strands = synthetic_strands


    # Simulation params
    n_sites = aligned_synthetic_strands.shape[1]
    n_edges = n_sites - 1

    # Make edge lengths on average equal to 1 / n_sites
    # aligned_synthetic_strands /= n_sites

    mass = np.ones(n_sites) * 1
    B = np.zeros((n_edges, 2, 2))
    B[:, 0, 0] = B[:, 1, 1] = 1e-6
    beta = 1e-6
    k = 0.0
    g = 9.81 * 1e-3
    damping = 0.2
    dt = 0.1
    xpbd_steps = 10
    energies = [Gravity()]
    frozen_pos_indices = np.array([0], dtype=int)
    frozen_theta_indices = np.array([], dtype=int)

    poses = aligned_synthetic_strands.copy()
    thetas = np.zeros((n_strands, n_edges))

    sims = []
    for i in range(n_strands):
        pos, theta = poses[i], thetas[i]
        sim = Sim(pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies,
                  damping=damping, dt=dt, xpbd_steps=xpbd_steps, frozen_pos_indices=frozen_pos_indices,
                  frozen_theta_indices=frozen_theta_indices)
        sims.append(sim)

    for i in tqdm(range(1000)):
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(step_wrapper, j, poses[j], thetas[j], sims[j]) for j in range(len(sims))]
            for future in concurrent.futures.as_completed(futures):
                j, pos, theta, sim = future.result()
                poses[j], thetas[j], sims[j] = pos, theta, sim
        strands_to_one_objs(poses, frame_idx=2 + i)
    return


def step_wrapper(i, pos, theta, sim, n_steps = 10):
    for _ in range(n_steps):
        pos, theta = sim.step(pos=pos, theta=theta)
    return i, pos, theta, sim


if __name__ == "__main__":
    main()
