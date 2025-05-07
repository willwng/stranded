import os
import cupy as cp
import numpy as np
from scipy.stats import qmc  # remains CPU
from tqdm import tqdm

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


def strands_to_one_objs(strands: cp.ndarray, frame_idx: int, output_file: str = None, y_up: bool = True):
    output_file = f"output/sampling_scratch/obj_{frame_idx}.obj" if output_file is None else output_file
    Visualizer.clear_output_file(output_file)
    vertex_offset = 1
    for strand in strands:
        pos = cp.asnumpy(strand[:, :3])  # convert back to numpy for visualization
        vertex_offset = Visualizer.to_simple_obj(pos=pos, output_file=output_file, init_offset=vertex_offset, y_up=y_up)
    return


def add_twist_tan(pos, theta, twist_freq):  # remains NumPy
    helix = RodHelixConverter.rod_to_helix(pos=cp.asnumpy(pos), theta=cp.asnumpy(theta))
    num_twists = len(helix.q) // 3
    total_twists = int(twist_freq * num_twists)
    twist_indices = np.random.choice(np.arange(num_twists), size=total_twists, replace=False)
    q_indices = 3 * twist_indices
    helix.q[q_indices] += np.random.uniform(-np.pi, np.pi, size=total_twists)
    pos, theta = RodHelixConverter.helix_to_rod(helix=helix)
    return cp.array(pos), cp.array(theta)


def main():
    height_scale = 0.5
    n = 100

    L0 = height_scale
    M0 = 1.0
    B0 = 0.01
    T0 = np.sqrt(M0 * L0**4 / B0)

    param_bounds = np.array([[0.01, 1.5]])
    n_params = param_bounds.shape[0]
    n_strands = 100

    sampler = qmc.LatinHypercube(d=n_params)
    lhs_sample = sampler.random(n=n_strands)
    scaled_samples = qmc.scale(lhs_sample, param_bounds[:, 0], param_bounds[:, 1])

    # Elastic moduli
    a, b = 80e-6, 40e-6
    E = 4.2e9
    I1 = (np.pi * a * b**3) / 4
    I2 = (np.pi * a**3 * b) / 4
    B1, B2 = E * I1, E * I2
    B0 = E * (a**4)

    B1_nd = B1 / B0
    B2_nd = B2 / B0

    beta, k, g, damping = 0.1, 0.0, 9.81e-3, 0.2
    dt = 0.04 / T0
    xpbd_steps = 10
    energies = [Gravity(), Bend(), Twist(), BendTwist()]

    poses, thetas, sims = [], [], []
    strand_labels = []

    for i, sample in enumerate(scaled_samples):
        r = sample.item(0)
        f, tf = 0.7, 0.5
        strand_labels.append({'r': r, 'f': f, 'tf': tf})

        # Generate rod, convert to GPU
        pos, theta = RodGenerator.example_rod(n, r / L0, f, height_scale / L0)
        pos, theta = cp.array(pos), cp.array(theta)
        pos, theta = add_twist_tan(pos, theta, tf)

        pos[:, 1] -= pos[0, 1]
        pos[:, 2] -= pos[0, 2]
        pos[:, 0] += 10 * i

        n_sites, n_edges = pos.shape[0], theta.shape[0]
        mass = cp.ones(n_sites)
        B = cp.zeros((n_edges, 2, 2))
        B[:, 0, 0] = B1_nd
        B[:, 1, 1] = B2_nd

        frozen_pos_indices = cp.array([0])
        frozen_theta_indices = cp.array([], dtype=int)

        sim = Sim(
            pos=pos, theta=theta, B=B, beta=beta, k=k, g=g, mass=mass, energies=energies,
            damping=damping, dt=dt, xpbd_steps=xpbd_steps,
            frozen_pos_indices=frozen_pos_indices, frozen_theta_indices=frozen_theta_indices
        )
        sim.define_rest_state(pos=pos, theta=theta)
        poses.append(pos)
        thetas.append(theta)
        sims.append(sim)

    tracking_freq = 20
    progress = tqdm(range(600 * tracking_freq))
    for i in progress:
        for j in range(n_strands):
            pos, theta = sims[j].step(pos=poses[j], theta=thetas[j])
            poses[j], thetas[j] = pos, theta
        if i % tracking_freq == 0:
            progress.set_description(f"Frame {i // tracking_freq}")
            strands_to_one_objs(cp.stack(poses), i // tracking_freq)

    strands_to_one_objs(cp.stack(poses), 1)
    data_to_save = {
        'poses': [cp.asnumpy(p) for p in poses],
        'labels': strand_labels
    }
    np.save("gpu_sample1.npy", data_to_save, allow_pickle=True)


if __name__ == "__main__":
    main()
