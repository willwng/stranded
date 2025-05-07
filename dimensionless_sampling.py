'''
Script to sample curl parameters and produce many curl + centerline pairs for diffusion model training.
All quantities re-scaled to be dimensionless, to be compatible with segment masses = 1
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc
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

# def create_frame(pos: np.ndarray,
#                  material_frame: np.ndarray,
#                  point_radii: np.ndarray,
#                  ax1_radii: np.ndarray,
#                  ax2_radii: np.ndarray,
#                  point_style: list[str],
#                  frame_idx: int,
#                  init_offset: int,
#                  output_file: str = None):
#     output_file = f"output/obj/obj_{frame_idx}.obj" if output_file is None else output_file
#     Visualizer.strand_to_obj(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
#                              ax2_radii=ax2_radii, point_style=point_style, output_file=output_file,
#                              init_offset=init_offset)
#     return

def strands_to_one_objs(strands: np.ndarray, frame_idx: int, output_file: str = None, y_up: bool = True):
    output_file = f"output/sampling_scratch/obj_{frame_idx}.obj" if output_file is None else output_file
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

# applying twist in gen. coord space, and changing scalp normal
def add_twist_tan(pos, theta, twist_freq): # twist_freq in units 1/length
    helix = RodHelixConverter.rod_to_helix(pos=pos, theta=theta)
    # q = [twist_{1}, curvature_{1, 1}, curvature_{1, 2}, ..., twist_{n}, curvature_{n, 1}, curvature_{n, 2}]
    # index into every 3rd element of q randomly and add a twist
    num_twists = len(helix.q) // 3
    total_twists = int(twist_freq * num_twists)
    twist_indices = np.random.choice(np.arange(num_twists), size=total_twists, replace=False)
    q_indices = 3 * twist_indices
    helix.q[q_indices] += np.random.uniform(-np.pi, np.pi, size=total_twists)
    
    return RodHelixConverter.helix_to_rod(helix=helix) # returns pos, theta for DER
    
def main(): 
    height_scale = 0.5
    n=100

    # Choosing characteristic scales
    L0 = height_scale  # length scale = full strand height
    M0 = 1.0           # segment mass already = 1
    B0 = 0.01           # choose 1 or average of B1, B2
    T0 = np.sqrt(M0 * L0**4 / B0)

    # Define parameter ranges
    param_bounds = np.array([
        [0.01, 1.5]                # radius (m)
        #[0.2, 0.2], #[0.01, 1.0],         # frequency (m^-1)
        #[0.5, 0.5] # [0.1, 1.0]          # twist frequency (m^-1)
    ])

    n_params = param_bounds.shape[0]
    n_strands = 100 

    sampler = qmc.LatinHypercube(d=n_params)
    lhs_sample = sampler.random(n=n_strands)
    scaled_samples = qmc.scale(lhs_sample, param_bounds[:,0], param_bounds[:,1])

    # moments for elliptical cross-sections
    a = 80e-6  # meters
    b = 40e-6  # meters
    E = 4.2e9  # Pascals = N/m^2
    I1 = (np.pi * a * b**3) / 4
    I2 = (np.pi * a**3 * b) / 4
    B1 = E * I1  # bending modulus along one principal axis
    B2 = E * I2

    B0 = 4.2e9 * (80e-6)**4  # scaling bending modulus

    B1_nd = B1 / B0
    B2_nd = B2 / B0

    beta = 0.1
    k = 0.0
    # g = 9.81
    g = 0.01 * (B0 / L0**2) # around 1% of elastic forces
    damping = 0.2
    dt = 0.002
    xpbd_steps = 10
    energies = [Gravity(), Bend(), Twist(), BendTwist()]

    # Rescaling
    # g = 9.81 * T0**2 / L0 # nondimensionalized gravity
    g=9.81e-3
    print(f'g: {g}')
    dt = 0.04 / T0
    B1_scaled = B1 / B0
    B2_scaled = B2 / B0 
    print(B1_scaled, B2_scaled)

    poses, thetas = [], []
    sims = []

    strand_labels = []

    for i, sample in enumerate(scaled_samples):
        r = sample.item(0)
        print(r)
        f, tf = 0.7, 0.5
        strand_labels.append({'r': r, 'f': f, 'tf': tf})
        pos, theta = RodGenerator.example_rod(n, r / L0, f, height_scale / L0)
        pos, theta = add_twist_tan(pos, theta, tf)

        pos[:, 1] -= pos[0, 1] # normalizing y
        pos[:, 2] -= pos[0, 2] # normalizing z
        pos[:, 0] += 10 * i

        n_sites, n_edges = pos.shape[0], theta.shape[0]
        mass = np.ones(n_sites) * 1

        B = np.zeros((n_edges, 2, 2))
        B[:, 0, 0] = B1_nd
        B[:, 1, 1] = B2_nd
        
        frozen_pos_indices = np.array([0], dtype=int)
        frozen_theta_indices = np.array([], dtype=int)

        sim = Sim(pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies,
                    damping=damping, dt=dt, xpbd_steps=xpbd_steps, frozen_pos_indices=frozen_pos_indices,
                    frozen_theta_indices=frozen_theta_indices)
        sim.define_rest_state(pos=pos, theta=theta)
        
        poses.append(pos)
        thetas.append(theta)
        sims.append(sim)


    tracking_freq = 20
    progress = tqdm(range(600 * tracking_freq))
    for i in progress:
        for j in range(n_strands):
            pos, theta = sims[j].step(pos=poses[j], theta=thetas[j])
            poses[j] = pos
            thetas[j] = theta
        if i % tracking_freq == 0:
            progress.set_description(f"Frame {i // tracking_freq}")
            strands_to_one_objs(np.array(poses), i // tracking_freq)

    strands_to_one_objs(np.array(poses), 1)
    data_to_save = {
        'poses': poses,  # list of numpy arrays
        'labels': strand_labels  # list of dicts
    }

    np.save("100_sample2_with_labels.npy", data_to_save, allow_pickle=True)
    return

def step_wrapper(i, pos, theta, sim, n_steps=10):
    for _ in range(n_steps):
        pos, theta = sim.step(pos=pos, theta=theta)
    return i, pos, theta, sim


if __name__ == "__main__":
    main()