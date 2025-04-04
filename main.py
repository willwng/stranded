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
                 init_offset: int,
                 output_file: str = None):
    output_file = f"output/obj/obj_{frame_idx}.obj" if output_file is None else output_file
    Visualizer.strand_to_obj(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                             ax2_radii=ax2_radii, point_style=point_style, output_file=output_file,
                             init_offset=init_offset)
    return


def strands_to_one_objs_fancy(strands: np.ndarray, thetas: np.ndarray, frame_idx: int, output_file: str = None):
    output_file = f"output/obj/obj_{frame_idx}.obj" if output_file is None else output_file
    Visualizer.clear_output_file(output_file)
    vertex_offset = 1
    for strand, theta in zip(strands, thetas):
        pos = strand[:, :3]
        bishop_frames = RodUtil.compute_bishop_frames(pos)
        material_frame = RodUtil.compute_material_frames(theta, bishop_frames)
        point_radii = np.ones(pos.shape[0]) * 0.1
        ax1_radii = np.ones(pos.shape[0]) * 0.1
        ax2_radii = np.ones(pos.shape[0]) * 0.1
        point_style = ["sphere"] * pos.shape[0]
        vertex_offset = Visualizer.strand_to_obj(pos=pos, material_frame=material_frame, point_radii=point_radii,
                                                 ax1_radii=ax1_radii,
                                                 ax2_radii=ax2_radii, point_style=point_style, output_file=output_file,
                                                 init_offset=vertex_offset)
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
    poses, thetas = [], []
    sims = []
    n_strands = 7
    for i in range(n_strands):
        n = 50
        if i == 0:
            pos, theta = RodGenerator.example_rod(n, curl_radius=1.0, curl_frequency=1.25, height_scale=0.2)
        elif i == 1:
            pos, theta = RodGenerator.example_rod(n, curl_radius=1.5, curl_frequency=1.0, height_scale=0.25)
        elif i == 2:
            pos, theta = RodGenerator.example_rod(n, curl_radius=1.5, curl_frequency=0.75, height_scale=0.3)
        elif i == 3:
            pos, theta = RodGenerator.example_rod(n, curl_radius=1.5, curl_frequency=0.5, height_scale=0.35)
        elif i == 4:
            pos, theta = RodGenerator.example_rod(n, curl_radius=0.5, curl_frequency=0.5, height_scale=0.5)
        elif i == 5:
            pos, theta = RodGenerator.example_rod(n, curl_radius=1.5, curl_frequency=0.1, height_scale=0.5)
        elif i == 6:
            pos, theta = RodGenerator.example_rod(n, curl_radius=1.0, curl_frequency=0.01, height_scale=0.5)
        else:
            pos, theta = RodGenerator.example_rod(n)

        pos[:, 0] += 5 * i
        pos[:, 1] -= pos[0, 1]
        pos[:, 2] -= pos[0, 2]


        n_sites, n_edges = pos.shape[0], theta.shape[0]
        mass = np.ones(n_sites) * 1
        B = np.zeros((n_edges, 2, 2))
        B[:, 0, 0] = 1
        B[:, 1, 1] = 1
        beta = 1
        k = 0.0
        g = 9.81 * 1e-3
        damping = 0.1
        dt = 0.1
        xpbd_steps = 10
        energies = [Gravity(), Bend(), Twist(), BendTwist()]
        frozen_pos_indices = np.array([0], dtype=int)
        frozen_theta_indices = np.array([], dtype=int)

        sim = Sim(pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies,
                  damping=damping, dt=dt, xpbd_steps=xpbd_steps, frozen_pos_indices=frozen_pos_indices,
                  frozen_theta_indices=frozen_theta_indices)

        poses.append(pos)
        thetas.append(theta)
        sims.append(sim)

    save_freq = 5
    progress = tqdm(range(1000 * save_freq))
    for i in progress:
        for j in range(n_strands):
            pos, theta = sims[j].step(pos=poses[j], theta=thetas[j])
            poses[j] = pos
            thetas[j] = theta
        if i % save_freq == 0:
            progress.set_description(f"Frame {i // save_freq}")
            strands_to_one_objs(np.array(poses), i // save_freq)
            # strands_to_one_objs_fancy(np.array(poses), np.array(thetas), i // save_freq)

    return


def scalp():
    # Open OBJ
    pos, edges = [], []
    with open("normals_one_seg.obj", 'r') as f:
        for line in f:
            if line[0] == 'v':
                pos.append(list(map(float, line.split()[1:])))
            elif line[0] == 'l':
                edges.append(list(map(int, line.split()[1:])))
    pos = np.array(pos)
    edges = np.array(edges) - 1

    # Convert y up and center the positions to origin
    pos = pos[:, [0, 2, 1]]
    pos -= np.mean(pos, axis=0)

    # Create strands starts
    start_strands = []
    for i1, i2 in edges:
        start_strands.append((pos[i1], pos[i2]))
    start_strands = np.array(start_strands)
    start_strands = start_strands[:2000]
    strands_to_one_objs(start_strands, frame_idx=0)

    # Create initial positions and directions
    r0 = start_strands[:, 0]
    n0 = np.zeros((start_strands.shape[0], 3, 3))
    tangents = start_strands[:, 1] - start_strands[:, 0]
    for i in range(n0.shape[0]):
        t = tangents[i]
        u = Vector.compute_orthogonal_vec(t)
        v = np.cross(t, u)
        n0[i, 0] = t / np.linalg.norm(t)
        n0[i, 1] = u / np.linalg.norm(u)
        n0[i, 2] = v / np.linalg.norm(v)

    # Convert to helices
    helices = []
    for i in range(1):
        n_sites = 300  # Including index 0
        L = .8
        s = np.linspace(0, L, n_sites)
        # Generalized coordinates
        curl_radius_mean, curl_radius_std = 0.3, 0.00  # 4mm +/- 1mm
        curl_radius = np.random.normal(curl_radius_mean, curl_radius_std, n_sites)
        delta_h = np.random.normal(0.1, 0.00)
        k_1 = 1 / curl_radius
        k_2 = np.random.normal(0, 100, n_sites)
        tau = delta_h / (2 * np.pi * curl_radius_mean ** 2) * np.ones(n_sites)
        avg_num_cm_random = .03 * L / n_sites
        num_cm_random = int(L / avg_num_cm_random)
        # random_idx = np.random.choice(n_sites, num_cm_random, replace=True)
        # delta_tau = np.random.normal(0, 100, num_cm_random)
        # tau[random_idx] += delta_tau
        q = np.stack([tau, k_1, k_2], axis=1).ravel()
        helix = Helix(q=q, q0=q.copy(), n_sites=n_sites, s=s, L=L, r0=r0[i], n0=n0[i], EI=np.ones(3 * n_sites))

        # Align with the given tangent
        r, _ = HelixUtil.propagate(helix)
        centerline = r[-1] - r[0]
        centerline = centerline / np.linalg.norm(centerline)
        hair_axis = n0[i, 0]
        rot_axis = np.cross(centerline, hair_axis)
        rot_axis = rot_axis / np.linalg.norm(rot_axis)
        rot_angle = np.arccos(np.dot(centerline, hair_axis))
        P_i = Quaternion.from_angle_axis(rot_angle, rot_axis)
        P_i.normalize()
        helix.n0 = P_i.rotate_vec(helix.n0)

        helices.append(helix)
    helices_to_one_obj(helices, frame_idx=1)

    return


def step_wrapper(i, pos, theta, sim, n_steps=10):
    for _ in range(n_steps):
        pos, theta = sim.step(pos=pos, theta=theta)
    return i, pos, theta, sim


if __name__ == "__main__":
    # main()
    scalp()
