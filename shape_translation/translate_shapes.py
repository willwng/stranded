import numpy as np
from scipy.interpolate import interp1d
from math_util.rotation import RotationUtil, Quaternion
from math_util.vectors import Vector
from tqdm import tqdm
import concurrent.futures
from rod.helix_util import HelixUtil
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from rod.helix import Helix
from rod.helix_util import HelixUtil
import pickle

from rod.helix import Helix
from rod.helix_util import HelixUtil

def create_target_centerlines(num_centerlines, n_points_list, L_list, spacing=5.0):
    target_centerlines = []
    
    for i in range(num_centerlines):
        n_sites = int(n_points_list[i])  # Number of sites along the helix
        L = int(L_list[i])  # Total length
        s = np.linspace(0, L, n_sites)
        r0 = np.array([i * spacing, 0.0, 0.0])  # Evenly spaced along x-axis
        n0 = np.eye(3)  # Standard basis
        twist = np.full(n_sites, np.pi / 12)
        curvature_x = 1.0 * np.cos(s * np.pi / 6) + np.random.random_sample()
        curvature_y = 0  # Keeping y-curvature at 0
        q = np.zeros((3 * n_sites,))
        q[0::3] = twist
        q[1::3] = curvature_x
        q[2::3] = curvature_y
        EI = np.ones((3 * n_sites,)) * 0.1  # Uniform stiffness
        
        helix_instance = Helix(q=q, q0=q, n_sites=n_sites, s=s, L=L, r0=r0, n0=n0, EI=EI)
        r, n = HelixUtil.propagate(helix_instance)
        
        # Set the top z-coordinate to 0
        r[0, 2] = 0
        target_centerlines.append(r)
    
    return target_centerlines

def compute_L(r):
    diff = np.diff(r, axis=0) 
    distances = np.linalg.norm(diff, axis=1)
    arc_length = np.sum(distances)
    return arc_length

def create_curls(L_list, site_list, curl_amount):
    helices = []
    for i in range(len(L_list)):
        n_sites = site_list[i]  # Number of sites along the helix
        L = L_list[i] # Total length
        s = np.linspace(0, L, n_sites)  # Arc length array
        r0 = np.array([0.0, 0.0, 0.0])
        n0 = np.eye(3)  # Standard basis (modify if needed)

        twist = np.full(n_sites, np.pi)
        curvature_x = curl_amount * np.cos(s * np.pi) + np.random.random_sample() # Varies to form a curl
        curvature_y = 0 #3.0 * np.sin(s * np.pi) + np.random.random_sample()  # Varies to form a curl

        q = np.zeros((3 * n_sites,))
        q[0::3] = twist
        q[1::3] = curvature_x
        q[2::3] = curvature_y

        EI = np.ones((3 * n_sites,)) * 0.1  # Uniform stiffness

        # r0 and n0 are the position and material frame of the clamped top node
        helix_instance = Helix(q=q, q0=q, n_sites=n_sites, s=s, L=L, r0=r0, n0=n0, EI=EI)

        # r, _ = HelixUtil.propagate(helix_instance)
        helices.append(helix_instance)
    return helices

def compute_centerlines(helices, group_size):
    real_centerlines = []
    for helix in helices:
        r, _ = HelixUtil.propagate(helix)
        real_centerline = np.zeros_like(r[:-group_size, :])
        
        # moving average of positions
        for i in range(group_size):
            real_centerline += r[i:-group_size+i, :]
        real_centerline /= group_size

        real_centerline = np.concatenate((r[0:1, :], real_centerline, r[-1:, :]), axis=0) # adding top/bottom clamped point

        #redistributing points along centerline to be uniformly spaced
        distances = np.cumsum(np.linalg.norm(np.diff(real_centerline, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)

        n_new = len(r)  # Same # points as helix
        new_distances = np.linspace(0, distances[-1], n_new)
        interp_func = interp1d(distances, real_centerline, axis=0, kind='linear')  # Linear interpolation
        real_centerline = interp_func(new_distances)
        real_centerlines.append(real_centerline)
    return real_centerlines

def compute_shape(helices, real_centerline_list):
    deltas = []
    for i in range(len(real_centerline_list)):
        r, _ = HelixUtil.propagate(helices[i])
        deltas_vec = r - real_centerline_list[i]
        deltas.append(deltas_vec)
    return deltas

def step_wrapper(i, helix):
    for _ in range(3):
        K_inv = HelixUtil.compute_inv_pointwise_stiffness_matrix(helix)
        B_gen = HelixUtil.compute_gen_force(helix, g=9.81, rhoS=1e3, seed=1)
        q_rest = helix.q0
        q_target = K_inv @ B_gen + q_rest[3:]
        q_target = np.concatenate([q_rest[:3], q_target])
        helix.q = q_target
    return i, helix
    

if __name__ == "__main__":

    # Modify to take inputs from a file
    # target_centerlines is an array of helix position indices
    # TODO: convert to np array, try Will's shape translation approach
    target_centerlines = create_target_centerlines(4, [15, 20, 25, 40], [15, 20, 25, 40])

    # First, get n_sites and length of target centerlines
    n_sites = [len(target_centerlines[i]) for i in range(len(target_centerlines))]
    L = [compute_L(target_centerlines[i]) for i in range(len(target_centerlines))]

    # Instantiate curls to simulate
    helices = create_curls(L, n_sites, 8)

    # Evolve them under gravity
    progress = tqdm(total=len(helices))
    finished = 0
    evolved_helices = [None] * len(helices)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(step_wrapper, i, helices[i]) for i in range(len(helices))] # len(helices)
        for future in concurrent.futures.as_completed(futures):
            i, helix = future.result()
            finished += 1
            evolved_helices[i] = helix
            progress.set_description(f"Finished {finished}/{len(helices)}")
    # evolved helices is a np array of helix instances

    # Compute centerlines
    real_centerlines = compute_centerlines(evolved_helices, group_size=10)
    # Compute shape deltas
    deltas = compute_shape(evolved_helices, real_centerlines)

    # Applying simulated shapes onto target centerlines
    print(f"Lengths - target_centerlines: {len(target_centerlines)}, deltas: {len(deltas)}")
    new_curls = [target_centerlines[i] + deltas[i] for i in range(len(target_centerlines))]

    with open("shaped_curls", "wb") as f:
        pickle.dump(new_curls, f)
    with open("target_centerlines", "wb") as f:
        pickle.dump(target_centerlines, f)