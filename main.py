import os
import numpy as np
import matplotlib.pyplot as plt
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


def strands_to_one_objs(strands: np.ndarray, frame_idx: int, output_file: str = None):
    output_file = f"output/obj/obj_{frame_idx}.obj" if output_file is None else output_file
    Visualizer.clear_output_file(output_file)
    vertex_offset = 1
    for strand in strands:
        pos = strand[:, :3]
        vertex_offset = Visualizer.to_simple_obj(pos=pos, output_file=output_file, init_offset=vertex_offset)
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
    import scienceplots
    plt.style.use(["science", "nature"])
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

    # plt.xlim([0, helix.n_sites])
    # plt.xticks([0, 35, 70])
    # plt.yticks([-1, 0, 1])
    # plt.yticks([])

    plt.legend()
    plt.show()


def main():
    seed = 1
    output_folder = f"output/seed_{seed}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

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

    forces = HelixUtil.compute_random_force(pos_target, seed=seed) + Gravity().compute_forces(pos_target, mass, g)
    create_frame(pos=pos_target, material_frame=target_material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=2, draw_arrows=True, forces=forces)

    # Compute the rest shape
    K_inv = HelixUtil.compute_inv_pointwise_stiffness_matrix(helix)
    B_gen = HelixUtil.compute_gen_force(helix, g=g, rhoS=rhoS, seed=seed)
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

    energies = [Twist(), Bend(), BendTwist(), Gravity(), RandomForce(seed=seed)]
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

    final_bishop_frame = RodUtil.compute_bishop_frames(pos=pos)
    final_material_frame = RodUtil.compute_material_frames(theta=theta, bishop_frame=final_bishop_frame)

    # Save rest + final shape
    create_frame(pos=rest_pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=0,
                 output_file=f"{output_folder}/rest.obj")
    create_frame(pos=pos_target, material_frame=target_material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=0, draw_arrows=True, forces=forces,
                 output_file=f"{output_folder}/target.obj")
    create_frame(pos=pos, material_frame=final_material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=0,
                 output_file=f"{output_folder}/final.obj")

    # Save final pos to npy
    np.save(f"{output_folder}/init_pos.npy", rest_pos)
    np.save(f"{output_folder}/init_theta.npy", rest_theta)
    np.save(f"{output_folder}/forces.npy", forces)
    np.save(f"{output_folder}/final_pos.npy", pos)
    np.save(f"{output_folder}/final_theta.npy", theta)

    # Compute L2 distance with target shape
    dis = np.linalg.norm(pos - pos_target)
    with open(f"{output_folder}/output.txt", 'w') as f:
        f.write(f"distance: {dis}")


def expt():
    n_pts = 51  # Including index 0
    L = 30
    s = np.linspace(0, L, n_pts)
    # Generalized coordinates
    curl_radius_mean, curl_radius_std = 0.4, 0.0  # 4mm +/- 1mm
    curl_radius = np.random.normal(curl_radius_mean, curl_radius_std, n_pts)
    delta_h = 1.0
    k_1 = 1 / curl_radius
    k_2 = np.random.normal(0, 0.2, n_pts)
    tau = delta_h / (2 * np.pi * curl_radius_mean ** 2) * np.ones(n_pts)

    # Randomize twist
    # avg_num_cm_random = 3 * L / n_pts  # Try to randomize every [avg_num_cm_random] cm
    # num_cm_random = int(L / avg_num_cm_random)
    # random_idx = np.random.choice(n_pts, num_cm_random, replace=True)
    # delta_tau = np.random.normal(0, 2.5, num_cm_random)
    # tau[random_idx] += delta_tau

    q = np.stack([tau, k_1, k_2], axis=1).ravel()
    # Boundary/initial conditions
    r0 = np.array([0, 0, 0])
    n0 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    # Stiffness, mass constants. Revisit this
    EI = np.ones(3 * n_pts) * 1
    g = 9.81 * 1e-3
    helix = Helix(q=q, q0=q.copy(), n_sites=n_pts, s=s, L=L, r0=r0, n0=n0, EI=EI)
    # print(helix.q)
    qtruth = q.copy()
    #
    # helix = HelixUtil.increase_resolution(helix)

    # plot_generalized_coords(helix)

    # Drawing parameters
    point_radii = 0.05 * np.ones(8 * helix.n_sites)
    ax1_radii = 0.03 * np.ones(8 * helix.n_sites)
    ax2_radii = 0.05 * np.ones(8 * helix.n_sites)
    point_style = ["sphere"] * helix.n_sites * 8

    pos, theta = RodHelixConverter.helix_to_rod(helix)

    # Align the helix to point upwards
    centerline = pos[-1] - pos[0]
    centerline = centerline / np.linalg.norm(centerline)
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(centerline, z_axis)
    rot_axis = rot_axis / np.linalg.norm(rot_axis)
    rot_angle = np.arccos(np.dot(centerline, z_axis))
    P_i = Quaternion.from_angle_axis(rot_angle, rot_axis)
    P_i.normalize()
    # for i in range(pos.shape[0]):
    # pos[i] = P_i.rotate_vec(pos[i])
    helix.n0 = P_i.rotate_vec(helix.n0)

    create_frame_helix(helix, point_radii, ax1_radii, ax2_radii, point_style, frame_idx=0)

    # Convert to DER
    pos, theta = RodHelixConverter.helix_to_rod(helix)
    bishop_frame = RodUtil.compute_bishop_frames(pos=pos)
    material_frame = RodUtil.compute_material_frames(theta=theta, bishop_frame=bishop_frame)
    create_frame(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=1)

    helix_draw = RodHelixConverter.rod_to_helix_pos(pos, n0=helix.n0)
    # helix_draw = RodHelixConverter.rod_to_helix(pos, theta)
    create_frame_helix(helix_draw, point_radii, ax1_radii, ax2_radii, point_style, frame_idx=2)
    plot_generalized_coords(helix)
    plot_generalized_coords(helix_draw)

    quit()
    # Reverse pos and theta
    # pos = pos[::-1]
    # theta = theta[::-1]

    # -- simulate
    mass = np.ones(helix.n_sites) * 1.0
    n_edges = theta.shape[0]
    B = np.zeros((n_edges, 2, 2))
    for i in range(n_edges):
        B[i, 0, 0] = 1.0
        B[i, 1, 1] = 1.0

    # Twisting stiffness
    beta = 1.0
    k = 0.0

    # Simulation parameters (damping for integration, time step, and number of XPBD steps)
    damping = 0.1
    dt = 0.1
    xpbd_steps = 10
    frozen_pos_indices = np.array([0, 1, 2], dtype=int)
    frozen_theta_indices = np.array([], dtype=int)

    energies = [Twist(), Bend(), BendTwist(), Gravity()]
    sim = Sim(pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies, damping=damping,
              dt=dt, xpbd_steps=xpbd_steps, frozen_pos_indices=frozen_pos_indices,
              frozen_theta_indices=frozen_theta_indices)
    sim.define_rest_state(pos, theta)
    save_freq = 10
    progress = tqdm(range(3 * save_freq, 10000))
    for i in progress:
        if i % save_freq == 0:
            # helix_draw = RodHelixConverter.rod_to_helix_pos(pos, n0=helix.n0)
            # create_frame_helix(helix_draw, point_radii, ax1_radii, ax2_radii, point_style, frame_idx=i // save_freq)
            # plot_generalized_coords(helix_draw)
            # helix_draw = HelixUtil.increase_resolution(helix_draw)
            # helix_draw = HelixUtil.increase_resolution(helix_draw)
            # plot_generalized_coords(helix_draw)
            # pos_draw, theta_draw = RodHelixConverter.helix_to_rod(helix_draw)
            # bishop_frame_draw = RodUtil.compute_bishop_frames(pos=pos_draw)
            # material_frame_draw = RodUtil.compute_material_frames(theta=theta_draw, bishop_frame=bishop_frame_draw)
            create_frame(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                         ax2_radii=ax2_radii, point_style=point_style, frame_idx=i // save_freq)
            # create_frame_helix(helix_draw, point_radii, ax1_radii, ax2_radii, point_style, frame_idx=i // save_freq)
            progress.set_description(f"Frame {i // save_freq}")
        pos, theta = sim.step(pos=pos, theta=theta)
    return


def convert_to_gen():
    # centerline_data = np.load("centerline_aligned.npy")
    centerline_data = np.load("curl_aligned.npy")
    strand_test = centerline_data[:1]
    n_strands = strand_test.shape[0]
    n_sites = strand_test.shape[1]

    # normalize the strands
    for i in range(strand_test.shape[0]):
        strand = strand_test[i]
        strand = RodGenerator.redistribute_vertices(strand)
        strand = RodHelixConverter.normalize_strand(strand)
        strand_test[i] = strand
    strands_to_one_objs(strand_test, frame_idx=0)

    # Convert to helix
    helices = []
    index_to_gen = np.zeros((n_sites, n_strands, 3))
    for i in tqdm(range(strand_test.shape[0])):
        strand = strand_test[i]
        helix = RodHelixConverter.rod_to_helix(strand, theta=np.zeros(strand.shape[0] - 1))
        helix = RodHelixConverter.rod_to_helix_pos(strand, n0=helix.n0)
        # plot_generalized_coords(helix)
        helices.append(helix)

        for j in range(helix.n_sites):
            index_to_gen[j, i] = helix.q[3 * j:3 * j + 3]
        # plot the fourier transform of the generalized coordinates
        plt.figure()
        labels = ["Twist", "Bend 1", "Bend 2"]
        for j in range(3):
            data = helix.q[j::3]
            plt.plot(data, label=labels[j])
            fft_data = np.fft.fft(data)
            freq = np.fft.fftfreq(len(data))
            plt.plot(freq, np.abs(fft_data), label=labels[j])

        plt.xlabel("Frequency")
        plt.legend()
    helices_to_one_obj(helices, frame_idx=1)
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
    for i in range(start_strands.shape[0]):
        n_sites = 128  # Including index 0
        L = .1
        s = np.linspace(0, L, n_sites)
        # Generalized coordinates
        curl_radius_mean, curl_radius_std = 0.003, 0.0  # 4mm +/- 1mm
        curl_radius = np.random.normal(curl_radius_mean, curl_radius_std, n_sites)
        delta_h = 0.01
        k_1 = 1 / curl_radius
        k_2 = np.random.normal(0, 100, n_sites)
        tau = delta_h / (2 * np.pi * curl_radius_mean ** 2) * np.ones(n_sites)
        avg_num_cm_random = .03 * L / n_sites
        num_cm_random = int(L / avg_num_cm_random)
        random_idx = np.random.choice(n_sites, num_cm_random, replace=True)
        delta_tau = np.random.normal(0, 100, num_cm_random)
        tau[random_idx] += delta_tau

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

    # Solve for the shape under gravity
    finished = 0
    progress = tqdm(total=len(helices))
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(step_wrapper, i, helices[i]) for i in range(len(helices))]
        for future in concurrent.futures.as_completed(futures):
            i, helix = future.result()
            finished += 1
            helices[i] = helix
            progress.set_description(f"Finished {finished}/{len(helices)}")

    helices_to_one_obj(helices, frame_idx=2)

    # Save the positions
    poses = []
    for helix in helices:
        pos, _ = HelixUtil.propagate(helix)
        poses.append(pos)

    poses = np.array(poses)
    np.save("scalp_pos.npy", poses)

    # Convert to DER
    # sims = []
    # poses, thetas = [], []
    # for helix in helices:
    #     pos, theta = RodHelixConverter.helix_to_rod(helix)
    #     pos *= 600
    #     mass = np.ones(pos.shape[0]) * 1.0
    #     n_edges = theta.shape[0]
    #     B = np.zeros((n_edges, 2, 2))
    #     for i in range(n_edges):
    #         B[i, 0, 0] = 1.0
    #         B[i, 1, 1] = 1.0
    #
    #     # Twisting stiffness
    #     beta = 1.0
    #     k = 0.0
    #     g = 9.81 * 1e-3
    #
    #     # Simulation parameters (damping for integration, time step, and number of XPBD steps)
    #     damping = 0.1
    #     dt = 0.04
    #     xpbd_steps = 10
    #     frozen_pos_indices = np.array([0, 1, 2], dtype=int)
    #     frozen_theta_indices = np.array([], dtype=int)
    #
    #     energies = [Twist(), Bend(), BendTwist(), Gravity()]
    #     sim = Sim(pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies, damping=damping,
    #               dt=dt, xpbd_steps=xpbd_steps, frozen_pos_indices=frozen_pos_indices,
    #               frozen_theta_indices=frozen_theta_indices)
    #     sims.append(sim)
    #     poses.append(pos)
    #     thetas.append(theta)
    #
    # poses, thetas = np.array(poses), np.array(thetas)
    # save_freq = 10
    # progress = tqdm(range(2, 10000))
    # for i in progress:
    #     for j in range(len(sims)):
    #         pos, theta = sims[j].step(pos=poses[j], theta=thetas[j])
    #         poses[j] = pos
    #         thetas[j] = theta
    #     # Draw
    #     if i % save_freq == 0:
    #         strands_to_one_objs(poses, frame_idx=i // save_freq)
    #         progress.set_description(f"Frame {i // save_freq}")

    return


def step_wrapper(i, helix):
    # Repeatedly solve for the shape
    for _ in range(3):
        # Compute the stiffness matrix and forces at current shape
        K_inv = HelixUtil.compute_inv_pointwise_stiffness_matrix(helix)
        B_gen = HelixUtil.compute_gen_force(helix, g=9.81, rhoS=1e3, seed=1)
        # Solve for the shape under forces
        q_rest = helix.q0
        q_target = K_inv @ B_gen + q_rest[3:]
        q_target = np.concatenate([q_rest[:3], q_target])
        helix.q = q_target
    return i, helix


if __name__ == "__main__":
    # main()
    # expt()
    convert_to_gen()
    # scalp()
