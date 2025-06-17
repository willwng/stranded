'''
Script to sample curl parameters and produce many curl + centerline pairs for diffusion model training.
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
from tqdm import tqdm
import concurrent.futures
import math

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
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor
from scipy.ndimage import gaussian_filter1d


def strands_to_one_objs(strands: np.ndarray, frame_idx: int, output_file: str = None, y_up: bool = True):
    output_file = f"output/sampling_scratch/obj_{frame_idx}.obj" if output_file is None else output_file
    Visualizer.clear_output_file(output_file)
    vertex_offset = 1
    for strand in strands:
        pos = strand[:, :3]
        vertex_offset = Visualizer.to_simple_obj(pos=pos, output_file=output_file, init_offset=vertex_offset, y_up=y_up)
    return

def add_many_rotation_inextensible(pos, theta, twist_indices=None, twist_values=None, twist_rate=0.8, n_iters=50):
    '''
    Apply twist to a rod and project to maintain segment inextensibility.
    Apply twist at designated twist indices by corresponding twist value. Both twist_indices and twist_values must be arrays of the same length.

    Parameters:
    - pos: (N, 3) array of vertex positions.
    - theta: passed through unchanged.
    - twist_indices: array of indices at which to apply twist. If None, randomly chosen.
    - twist_values: array of twist angles (in radians) to apply at the corresponding twist_indices.
    - twist_rate: fraction of segments to twist, if twist_indices not provided.
    - n_iters: number of projection iterations for enforcing inextensibility.

    Returns:
    - pos_rotated: (N, 3) rotated and projected positions.
    - theta: unchanged.
    '''
    pos_rotated = pos.copy()
    num_vertices = len(pos)

    # Rest lengths of each segment
    rest_lengths = np.linalg.norm(pos[1:] - pos[:-1], axis=1)

    max_twist_points = num_vertices - 1
    total_twists = int(twist_rate * max_twist_points)

    # Generate twist indices/values if not provided
    if twist_indices is None:
        twist_indices = np.random.choice(np.arange(max_twist_points), size=total_twists, replace=False)
    if twist_values is None:
        twist_values = np.random.uniform(-0.05, 0.05, size=len(twist_indices))

    for i, twist_index in enumerate(twist_indices):
        if twist_index >= num_vertices - 1:
            continue

        twist_angle = twist_values[i]
        axis = pos[twist_index + 1] - pos[twist_index]
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-8:
            continue
        axis = axis / axis_norm

        # Rodrigues' rotation formula
        K = np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
        I = np.eye(3)
        R = I + np.sin(twist_angle) * K + (1 - np.cos(twist_angle)) * (K @ K)

        origin = pos[twist_index]
        for j in range(twist_index + 1, num_vertices):
            vec = pos_rotated[j] - origin
            pos_rotated[j] = origin + R @ vec

    # Project to enforce inextensibility
    for _ in range(n_iters):
        for i in range(num_vertices - 1):
            p1, p2 = pos_rotated[i], pos_rotated[i + 1]
            edge = p2 - p1
            current_len = np.linalg.norm(edge)
            if current_len < 1e-8:
                continue
            rest_len = rest_lengths[i]
            correction = 0.5 * (1 - rest_len / current_len) * edge
            pos_rotated[i] += correction
            pos_rotated[i + 1] -= correction

    return pos_rotated, theta

def add_twist_cum(pos, theta, twists_per_curl, angular_curl_freq, indices_per_curl):
    """
    Add cumulative twist to a rod based on desired twists per curl.
    """
    helix = RodHelixConverter.rod_to_helix(pos=pos, theta=theta)
    num_vertices = len(pos)
    num_twist_indices = len(helix.q) // 3

    # compute how many total twist insertions to apply
    twist_rate = twists_per_curl / indices_per_curl
    total_twists = int(twist_rate * num_vertices)

    # randomly select the twist indices to inject twist
    twist_indices = np.random.choice(
        np.arange(num_twist_indices), size=total_twists, replace=False
    )

    # sort twist indices so we apply them in increasing order
    twist_indices.sort()

    # add cumulative twist — later indices accumulate earlier ones
    for idx in twist_indices:
        # delta_twist = np.random.uniform(-0.1, 0.1)  # radians
        delta_twist = np.random.uniform(-0.8*np.pi, 0.8*np.pi)
        helix.q[3 * idx :] += delta_twist  # apply to this index and all later ones

    return RodHelixConverter.helix_to_rod(helix=helix)

def in_to_meters(x):
    inches_to_meters = 0.0254
    return x * inches_to_meters

def smooth_theta(theta, sigma=2.0):
    return gaussian_filter1d(theta, sigma=sigma, mode='nearest')

def main(): # trying to scale from meters to inches
    scaling_factor = 20 # 0.5meters to inches
    # sim parameters
    beta = 0.1 * (1 / scaling_factor**2) # bending stiffness
    # beta = 0.1
    k = 0.0 # inextensibility
    # g = 9.81e-3 * (1 / scaling_factor**3)
    # g = 9.81e-2 * (1 / scaling_factor**3)
    g = 9.81e-4
    damping = 0.5
    # dt = 0.08 * (1 / np.sqrt(scaling_factor))
    dt = 0.08
    xpbd_steps = 10
    energies = [Gravity(), Bend(), Twist(), BendTwist()]

    mass_default = 1.0 #* (1 / scaling_factor**3)#3)      

    # trying to multiply all length scales by 10 for stability, then converting back at the end
    rad_default = 0.5 # inches
    L = 12# inches
    curl_wavelength = 1 # inch

    # twist_prev_default = 10 # twists per curl
    n_points = 256 #300 # dimensionless
    height_scale = L / n_points # vertical displacement per segment
    indices_per_curl = int(curl_wavelength / height_scale)
    print(f'indices per curl: {indices_per_curl}')
    # ang_curl_freq_default = 0.1 * np.pi
    ang_curl_freq_default = (2 * np.pi) / indices_per_curl

    # Define parameter ranges
    param_bounds = np.array([
        #[0.01, 1.5],                     # radius (in)
        # [0.01, 0.99] #[0.01, 1.0],       # curl freq (in)
        [1, indices_per_curl-1] # [0.1, 1.0]          # twists per curl
    ])

    n_params = param_bounds.shape[0]
    n_strands = 5

    sampler = qmc.LatinHypercube(d=n_params)
    lhs_sample = sampler.random(n=n_strands)
    scaled_samples = qmc.scale(lhs_sample, param_bounds[:,0], param_bounds[:,1])

    poses, thetas = [], []
    sims = []

    strand_labels = []

    # rads = [0.5, 0.5] # in inches
    # twists = [0.2, 0.5, 0.8] 
    wavelengths = np.linspace(0.25, 12, 100)
    # for i, sample in enumerate(scaled_samples):
    for i in range(n_strands):
        # tf = twists[math.floor(i / (n_strands / 3))]
        # r = sample.item(0)
        indices_per_curl = wavelengths[i] / height_scale
        ang_curl_freq = (2 * np.pi) / indices_per_curl
        print(wavelengths[i], ang_curl_freq)
        # print(tf)
        tf = 0.8
        tf = int(tf * indices_per_curl)
        r, f = rad_default, ang_curl_freq
        strand_labels.append({'r': r, 'angular_curl_freq': f, 'twists_per_curl': tf})
        pos, theta = RodGenerator.example_rod(n=n_points, curl_radius=in_to_meters(r), curl_frequency=f, height_scale=in_to_meters(height_scale))

        # pos, theta = add_twist_cum(pos, theta, tf, ang_curl_freq_default, indices_per_curl)

        twist_indices = np.linspace(5, 250, 20).astype(int)
        twist_values = np.random.uniform(-np.pi/4, np.pi/4, size=len(twist_indices)) # degrees
        pos, theta = add_many_rotation_inextensible(pos, theta, twist_indices = twist_indices, twist_values=twist_values)
        # theta[1:] = smooth_theta(theta[1:], sigma=2.0)  # or kernel_size=5
        # theta = np.clip(theta, -np.pi, np.pi)

        pos[:, 1] -= pos[0, 1] # normalizing y
        pos[:, 2] -= pos[0, 2] # normalizing z
        pos[:, 0] += 0.1 * i

        n_sites, n_edges = pos.shape[0], theta.shape[0]
        mass = np.ones(n_sites) * mass_default

        rho = mass_default / (L * 0.0254)  # total mass / length in meters
        segment_length = L / n_points * 0.0254  # in meters
        I = rho * (in_to_meters(rad_default) ** 2) * segment_length
        B = np.zeros((n_edges, 2, 2))
        B[:, 0, 0] = I
        B[:, 1, 1] = I

        # B = np.zeros((n_edges, 2, 2))
        # B[:, 0, 0] = #mass_default
        # B[:, 1, 1] = #mass_default
        
        frozen_pos_indices = np.array([0], dtype=int)
        frozen_theta_indices = np.array([], dtype=int)

        sim = Sim(pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies,
                    damping=damping, dt=dt, xpbd_steps=xpbd_steps, frozen_pos_indices=frozen_pos_indices,
                    frozen_theta_indices=frozen_theta_indices)
        sim.define_rest_state(pos=pos, theta=theta)
        strands_to_one_objs(np.array(poses), 2)
        
        poses.append(pos)
        thetas.append(theta)
        sims.append(sim)

    data_to_save = {
        'poses': poses,  # list of numpy arrays
        'labels': strand_labels  # list of dicts
    }
    np.save("100_wavelengths.npy", data_to_save, allow_pickle=True)


    tracking_freq = 20
    progress = tqdm(range(800 * tracking_freq))
    with ProcessPoolExecutor() as executor:
        for i in progress:
            futures = [executor.submit(step_wrapper, j, poses[j], thetas[j], sims[j], n_steps=1) for j in range(n_strands)]
            results = [f.result() for f in futures]
            for j, pos, theta, sim in results:
                poses[j] = pos
                thetas[j] = theta

            if i % tracking_freq == 0:
                progress.set_description(f"Frame {i // tracking_freq}")
                strands_to_one_objs(np.array(poses), i // tracking_freq + 1)

    data_to_save = {
        'poses': poses,  # list of numpy arrays
        'labels': strand_labels  # list of dicts
    }

    np.save("more_twist_samples.npy", data_to_save, allow_pickle=True)
    return

def step_wrapper(i, pos, theta, sim, n_steps=10):
    for _ in range(n_steps):
        pos, theta = sim.step(pos=pos, theta=theta)
    # print(f"Post-{n_steps}-step theta diff:", np.linalg.norm(theta - sim.init_state.theta0))
    return i, pos, theta, sim


if __name__ == "__main__":
    main()