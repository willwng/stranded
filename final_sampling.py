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

def add_twist(pos, theta, twist_indices=None, twist_values=None, twist_rate=0.8,
                angular_curl_freq=2*np.pi, indices_per_curl=20):
    """
    Add twist to a rod by injecting twist at selected indices.

    Parameters:
    - pos: (N, 3) array of vertex positions.
    - theta: twist angles (N-1,) — passed through RodHelixConverter.
    - twist_indices: array of indices to apply cumulative twist (optional).
    - twist_values: array of twist angles in radians (optional, same length as twist_indices).
    - twist_rate: fraction of vertices to twist (used if twist_indices is None).
    - twists_per_curl: desired number of twists per curl (used only for computing default twist_rate).
    - angular_curl_freq: angular frequency of the helix (radians per index).
    - indices_per_curl: number of indices per curl (used only for computing default twist_rate).

    Returns:
    - pos_out, theta_out: positions and twists after applying cumulative twist.
    """
    helix = RodHelixConverter.rod_to_helix(pos=pos, theta=theta)
    num_vertices = len(pos)
    num_twist_indices = len(helix.q) // 3

    # Determine how many total twist applications to apply
    if twist_indices is None:
        twist_rate = twists_per_curl / indices_per_curl if twist_rate is None else twist_rate
        total_twists = int(twist_rate * num_vertices)
        twist_indices = np.random.choice(np.arange(num_twist_indices), size=total_twists, replace=False)
    if twist_values is None:
        twist_values = np.random.uniform(-np.pi, np.pi, size=len(twist_indices))

    twist_indices = np.array(twist_indices)
    twist_values = np.array(twist_values)
    twist_values = np.degrees(twist_values)
    sorted_order = np.argsort(twist_indices)
    twist_indices = twist_indices[sorted_order]
    twist_values = twist_values[sorted_order]

    # Apply  twist: each twist affects this index and all after
    for i, idx in enumerate(twist_indices):
        delta_twist = twist_values[i] # convert to degrees
        helix.q[3 * idx] += delta_twist  # apply to this edge

    return RodHelixConverter.helix_to_rod(helix=helix)

def in_to_meters(x):
    inches_to_meters = 0.0254
    return x * inches_to_meters

def step_wrapper(i, pos, theta, sim, n_steps=10):
    for _ in range(n_steps):
        pos, theta = sim.step(pos=pos, theta=theta)
    # print(f"Post-{n_steps}-step theta diff:", np.linalg.norm(theta - sim.init_state.theta0))
    return i, pos, theta, sim

#################### Sampling ################################
def main():

    # Simulation Parameters
    scaling_factor = 20 # 0.5 meters --> inches
    beta = 0.1 * (1 / scaling_factor**2) # bending stiffness
    k = 0.0 # inextensibility
    g = 9.81e-4
    damping = 0.5
    dt = 0.08
    xpbd_steps = 10
    energies = [Gravity(), Bend(), Twist(), BendTwist()]
    mass_default = 1.0 

    # Procedural Strand Generation
    L = 6 # 12 inches
    n_points = 256
    curl_wavelength_default = 1 # inch
    curl_radius_default = 0.5 # inch
    twist_freq_default = 0.8

    height_scale = L / n_points # vertical displacement per segment
    indices_per_curl = curl_wavelength_default / height_scale
    ang_curl_freq_default = (2 * np.pi) / indices_per_curl


    param_bounds = np.array([
        [0.1, 0.8],        # radius (in)
        # [0.01, 0.99].       # curl freq (in)
        # [0.1, 1.0]          # twist prevalence
    ])

    n_strands = 5
    n_params = param_bounds.shape[0]
    sampler = qmc.LatinHypercube(d=n_params)
    lhs_sample = sampler.random(n=n_strands)
    scaled_samples = qmc.scale(lhs_sample, param_bounds[:,0], param_bounds[:,1])

    poses, thetas = [], []
    sims = []
    strand_labels = []

    for i, sample in enumerate(scaled_samples):
        r = sample.item(0)
        f, tf = ang_curl_freq_default, twist_freq_default
        strand_labels.append({'r': r, 'angular_curl_freq': f, 'twists_per_curl': tf})
        pos, theta = RodGenerator.example_rod(n=n_points, curl_radius=in_to_meters(r), curl_frequency=f, height_scale=in_to_meters(height_scale))
        pos, theta = add_twist(pos, theta, twist_rate=tf)

        pos[:, 1] -= pos[0, 1] # normalizing y
        pos[:, 2] -= pos[0, 2] # normalizing z
        pos[:, 0] += 0.1 * i

        n_sites, n_edges = pos.shape[0], theta.shape[0]
        mass = np.ones(n_sites) * mass_default

        rho = mass_default / (L * 0.0254)  # total mass / length in meters
        segment_length = L / n_points * 0.0254  # in meters
        I = rho * (in_to_meters(r) ** 2) * segment_length
        B = np.zeros((n_edges, 2, 2))
        B[:, 0, 0] = I
        B[:, 1, 1] = I

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

    np.save("sampling_test.npy", data_to_save, allow_pickle=True)

    # Running Simulation

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

    # np.save("more_twist_samples.npy", data_to_save, allow_pickle=True)
    return


if __name__ == "__main__":
    main()


    

