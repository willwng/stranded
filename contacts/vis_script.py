from math_util.rotation import RotationUtil, Quaternion
from math_util.vectors import Vector
from tqdm import tqdm
import concurrent.futures
from rod.helix_util import HelixUtil
import multiprocessing
multiprocessing.set_start_method('spawn', force=True)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import plotly.graph_objects as go
from rod.helix import Helix
from rod.helix_util import HelixUtil
import pickle
from scipy.interpolate import RBFInterpolator

# def step_wrapper(i, helix):
    
#     for _ in range(3):
#         K_inv = HelixUtil.compute_inv_pointwise_stiffness_matrix(helix)
#         B_gen = HelixUtil.compute_gen_force(helix, g=9.81, rhoS=1e3, seed=1)
#         q_rest = helix.q0
#         q_target = K_inv @ B_gen + q_rest[3:]
#         q_target = np.concatenate([q_rest[:3], q_target])
#         helix.q = q_target
#     return i, helix

# Getting strand starts from obj file
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
start_strands = start_strands[:2000] # 2000 strands. start & end pts per strand. 3 coords per point

# Create initial positions and directions
print("Creating initial positions and directions")
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

# ----------------------------------------
# Scaling Points & Fitting Ellipsoid for SDF
print("Fitting Ellipsoid")
r0_mean = np.mean(r0, axis=0)
r0_std = np.std(r0, axis=0)  # Avoids large numerical values
r0_scaled = (r0 - r0_mean) / r0_std  # Normalize
r0_scaled += 1e-6 * np.random.randn(*r0_scaled.shape) # jitter to avoid singular matrix

center = np.mean(r0_scaled, axis=0)  # Compute the mean
cov = np.cov(r0_scaled.T)            # Compute the covariance matrix
U, S, _ = np.linalg.svd(cov)          # SVD to get principal axes

radii = np.sqrt(S)  # Approximate radii of the ellipsoid

num_points = 1000
phi = np.arccos(1 - 2 * np.linspace(0, 1, num_points))
theta = np.pi * (1 + 5**0.5) * np.arange(num_points)

x = np.sin(phi) * np.cos(theta)
y = np.sin(phi) * np.sin(theta)
z = np.cos(phi)
unit_sphere_points = np.vstack((x, y, z)).T

ellipsoid_points = unit_sphere_points * radii*1.7  # Scale by radii
ellipsoid_points = ellipsoid_points @ U.T  # Rotate to match original orientation
ellipsoid_points += center  # Translate to match original data
ellipsoid_points[:, 1] - 0.25
ellipsoid_points[:, 2] - 0.1

# Fitting SDF
print("Fitting SDF")
# Creating training data
offset = 0.2
ellipsoid_center = np.mean(ellipsoid_points, axis=0)
inner_points = ellipsoid_points * (1 - offset) + ellipsoid_center * offset  # Move inward
outer_points = ellipsoid_points * (1 + offset) - ellipsoid_center * offset  # Move outward

sdf_values = np.concatenate([
    np.zeros(len(ellipsoid_points)),  # Surface
    -offset * np.ones(len(inner_points)),  # Inside
    offset * np.ones(len(outer_points))  # Outside
])

# Combine all points
sdf_points = np.vstack((ellipsoid_points, inner_points, outer_points))

rbf = RBFInterpolator(
    sdf_points, sdf_values, 
    kernel='multiquadric',
    epsilon=0.1,  # regularization
)
# Define SDF function
def scalp_sdf(p):
    p_scaled = (p - r0_mean) / r0_std  # Apply same scaling
    return rbf(p_scaled)

def sdf_gradient(p, eps=1e-4):
    grad = np.zeros((p.shape[0], 3))
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = eps
        grad = (scalp_sdf(p + dp) - scalp_sdf(p - dp)) / (2 * eps)
    return grad / np.linalg.norm(grad)  # Normalize

# Rewriting gen_forces for step_wrapper
def compute_gen_force(helix: Helix, g: float, rhoS: float, seed: int, sdf_func) -> np.ndarray:
    """
    Computes the generalized force including gravity and scalp contact forces using SDF.
    """
    grad = np.zeros(3 * (helix.n_sites - 1))
    eps = 1e-6
    k_collision = 1e4  # Stiffness for head-hair contact force

    q_free = helix.q.copy()[3:]
    for i in range(3 * (helix.n_sites - 1)):
        q_plus = q_free.copy()
        q_plus[i] += eps
        helix.q = np.concatenate([helix.q[:3], q_plus])
        r_plus, _ = HelixUtil.propagate(helix)
        U_g_plus = HelixUtil.compute_gen_potential_pos(helix, r_plus, g, rhoS, seed)
        q_minus = q_free.copy()
        q_minus[i] -= eps
        helix.q = np.concatenate([helix.q[:3], q_minus])
        r_minus, _ = HelixUtil.propagate(helix)
        U_g_minus = HelixUtil.compute_gen_potential_pos(helix, r_minus, g, rhoS, seed)

        # Finite difference gravity force
        grad[i] = (U_g_plus - U_g_minus) / (2 * eps)

    # Compute contact forces from SDF
    r, _ = HelixUtil.propagate(helix)
    d = scalp_sdf(r)
    grad_sdf = sdf_gradient(r)

    for j in range(helix.n_sites - 1):  # Iterate over sites (except base)
        if d[j] < 0:  # If inside scalp, apply repulsive force
            contact_force = -k_collision * d[j] * grad_sdf[j]  # (3,)
            # Map force to `grad` (flattened array)
            grad[3 * j : 3 * j + 3] += contact_force

    helix.q = np.concatenate([helix.q[:3], q_free])  # Reset q
    return -grad

def step_wrapper(i, helix):
    for _ in range(3):
        K_inv = HelixUtil.compute_inv_pointwise_stiffness_matrix(helix)
        B_gen = compute_gen_force(helix, g=9.81, rhoS=1e3, seed=1, sdf_func=scalp_sdf)
        q_rest = helix.q0
        q_target = K_inv @ B_gen + q_rest[3:]
        q_target = np.concatenate([q_rest[:3], q_target])
        helix.q = q_target
    return i, helix

# ----------------------------------------
# Convert to helices
print("Converting to Helices")
helices = []
for i in range(start_strands.shape[0]):
    n_sites = 128  # Including index 0
    #L = .1
    L = .5
    s = np.linspace(0, L, n_sites)
    # Generalized coordinates
    #curl_radius_mean, curl_radius_std = 0.003, 0.0  # 4mm +/- 1mm #0.4 cm +/- 0.1 cm
    curl_radius_mean, curl_radius_std = 0.01, 0.0 # 1 cm +/- 0 cm
    curl_radius = np.random.normal(curl_radius_mean, curl_radius_std, n_sites)
    # delta_h = 0.01 #1 cm
    delta_h = 0.007 #0.7 cm
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

# Solve for the shape under gravity
if __name__ == "__main__":  # Multiprocessing needs this guard
    progress = tqdm(total=len(helices))
    finished = 0
    helices_small = helices[::10]
    evolved_helices = np.zeros_like(helices_small)

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(step_wrapper, i, helices_small[i]) for i in range(len(helices_small))] # len(helices)
        for future in concurrent.futures.as_completed(futures):
            i, helix = future.result()
            finished += 1
            evolved_helices[i] = helix
            progress.set_description(f"Finished {finished}/{len(helices_small)}")

    with open("evolved_helices_contacts_smallo_bige.pkl", "wb") as f:
        pickle.dump(evolved_helices, f)
