import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from main import strands_to_one_objs, helices_to_one_obj
from rod.RodHelixConverter import RodHelixConverter
from rod.helix_util import HelixUtil


def main():
    # Load strand data
    strands = np.load("curl.npy")
    avg_strands_pos = np.mean(strands, axis=(0, 1))
    strands = strands[:]
    strands -= avg_strands_pos
    avg_edge_length = 0
    for i in range(strands.shape[0]):
        strand = strands[i]
        e = strand[1:] - strand[:-1]
        edge_lengths = np.linalg.norm(e, axis=1)
        avg_edge_length += np.mean(edge_lengths)
    avg_edge_length /= strands.shape[0]
    strands /= avg_edge_length

    # Count the number of strands that point in the negative z-direction
    count = 0
    for strand in strands:
        if strand[0, 2] > strand[-1, 2]:
            count += 1
    if count > strands.shape[0] // 2:
        strands[:, :, 2] = -strands[:, :, 2]

    strands_to_one_objs(strands, frame_idx=0)

    print(f"Loaded {strands.shape[0]} strands")

    # Look at the generalized coordinates
    labels = ["Twist", "Bend 2", "Bend 2"]
    ind = np.array([1000])
    target_strands = strands[ind]
    helices = []
    fig, ax = plt.subplots(3, 1)
    for strand in tqdm(target_strands):
        helix = RodHelixConverter.rod_to_helix(strand, np.zeros(strand.shape[0] - 1))
        helix = RodHelixConverter.rod_to_helix_pos(strand, helix.n0)
        for j in range(3):
            ax[j].plot(helix.q[j::3], color=f"C{j}")
            ax[j].set_xticks([0, helix.n_sites // 2, helix.n_sites])
            ax[j].set_ylabel(labels[j])
            ax[j].axhline(0, color="black", linestyle="--")
        helices.append(helix)
    fig.supxlabel("Node Index")

    fig, ax = plt.subplots(3, 1)
    helices_to_one_obj(helices, frame_idx=1)
    for helix in helices:
        # --- Compute Rest Shape
        # helix.EI[0::3] = 1
        # helix.EI[1::3] = 10
        # helix.EI[2::3] = 10
        K_inv = HelixUtil.compute_inv_pointwise_stiffness_matrix(helix)
        B_gen = HelixUtil.compute_gen_force(helix, g=9.81 * 1e-4, rhoS=1.0, seed=1)
        q_target = helix.q.copy()
        q_rest = q_target[3:] - K_inv @ B_gen
        q_rest = np.concatenate([q_target[:3], q_rest])
        helix.q0 = q_rest.copy()
        helix.q = q_rest
        for j in range(3):
            ax[j].plot(helix.q[j::3], color=f"C{j}")
            ax[j].set_xticks([0, helix.n_sites // 2, helix.n_sites])
            ax[j].set_ylabel(labels[j])
            ax[j].axhline(0, color="black", linestyle="--")
    # fig xlabel
    fig.supxlabel("Node Index")
    helices_to_one_obj(helices, frame_idx=2)

    plt.show()


if __name__ == "__main__":
    main()
