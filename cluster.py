import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from main import strands_to_one_objs
from rod.RodHelixConverter import RodHelixConverter
from rod.helix_util import HelixUtil
from rod.preprocess import Preprocess


def feature_extraction(strands):
    """
    Extract features from strands for clustering
    Input: strands - np.array of shape (N, 128, 3)
    Returns: features - np.array of shape (N, num_features)
    """
    N = strands.shape[0]
    features = []

    for i in range(N):
        strand = strands[i]

        # Take fft of each dimension
        fft_x = fft(strand[:, 0])[:20]
        fft_y = fft(strand[:, 1])[:20]
        fft_z = fft(strand[:, 2])[:20]

        # Combine the ffts to make a feature vector
        strand_features = np.concatenate((fft_x, fft_y, fft_z))
        strand_features = np.abs(strand_features)

        features.append(strand_features)

    return np.array(features)


def find_optimal_k(features, start_k, max_k):
    """
    Find optimal k for KMeans using silhouette score
    """
    silhouette_scores = []
    for k in range(start_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        score = silhouette_score(features, labels)
        silhouette_scores.append(score)
        print(f"k = {k}, silhouette score = {score:.3f}")

    optimal_k = np.argmax(silhouette_scores) + start_k

    plt.plot(range(start_k, max_k + 1), silhouette_scores)
    plt.xlabel("Number of clusters")
    plt.ylabel("Silhouette score")
    plt.show()
    return optimal_k


def cluster_strands(strands):
    # Extract features and standardize
    features = feature_extraction(strands)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Find optimal number of clusters
    # n_clusters = find_optimal_k(scaled_features, start_k=3, max_k=15)
    # print(f"Optimal number of clusters: {n_clusters}")

    n_clusters = 20
    # Apply KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(scaled_features)

    return labels


def visualize_clusters(strands, labels):
    """
    Visualize clustered strands in 3D
    """
    # Get unique labels
    unique_labels = np.unique(labels)

    # Plot each strand
    new_strands = strands.copy()
    for i, strand in enumerate(strands):
        label = labels[i]
        color_idx = np.where(unique_labels == label)[0][0]
        x_modifier = 30 * color_idx
        new_strands[i] += np.array([x_modifier, 0, 0])

    strands_to_one_objs(new_strands, frame_idx=0)
    return


def main():
    # Load strand data
    strands = np.load("curl.npy")
    strands = strands[::5]
    strands, centroids, directions = Preprocess.align_data(strands)
    for i in range(strands.shape[0]):
        strand = strands[i]
        strand = RodHelixConverter.normalize_strand(strand, normalize_positions=False, normalize_direction=False)
        # If z is decreasing, reverse
        if strand[0, 2] > strand[-1, 2]:
            strand = strand[::-1]
        strands[i] = strand

    print(f"Loaded {strands.shape[0]} strands")

    # Cluster strands using feature-based approach
    labels = cluster_strands(strands)
    visualize_clusters(strands, labels)

    # Get all strands with label
    target_label = 5
    target_strands = strands[labels == target_label]
    target_strands = np.array([target_strands[0]])
    print(f"Found {target_strands.shape[0]} strands with label {target_label}")

    fig, ax = plt.subplots(3, 1)
    for i, strand in enumerate(target_strands):
        alpha = 0.3 if i < target_strands.shape[0] - 1 else 1
        ax[0].plot(strand[:, 0], color="C0", alpha=alpha)
        ax[1].plot(strand[:, 1], color="C1", alpha=alpha)
        ax[2].plot(strand[:, 2], color="C2", alpha=alpha)
        labels = ["x", "y", "z"]
        for j in range(3):
            ax[j].set_xticks([0, strand.shape[0] // 2, strand.shape[0]])
            ax[j].set_ylabel(labels[j])
        ax[2].set_xlabel("Node index")
    max_v = np.max(np.abs(target_strands))
    ax[0].set_ylim([-max_v, max_v])
    ax[1].set_ylim([-max_v, max_v])
    ax[2].set_ylim([-max_v, max_v])
    fig.suptitle("Original Strands")

    # Look at the fft of the strands
    fig, ax = plt.subplots(3, 1)
    target_fft = np.zeros((len(target_strands), target_strands[0].shape[0], 3))
    for i, strand in enumerate(target_strands):
        alpha = 0.3 if i != 0 else 1
        fft_x, fft_y, fft_z = fft(strand[:, 0]), fft(strand[:, 1]), fft(strand[:, 2])
        target_fft[i, :, 0] = fft_x
        target_fft[i, :, 1] = fft_y
        target_fft[i, :, 2] = fft_z
        ax[0].plot(np.abs(fft_x), color="C0", alpha=alpha)
        ax[1].plot(np.abs(fft_y), color="C1", alpha=alpha)
        ax[2].plot(np.abs(fft_z), color="C2", alpha=alpha)
        labels = ["x", "y", "z"]
        for j in range(3):
            ax[j].set_xticks([0, strand.shape[0] // 2, strand.shape[0]])
            ax[j].set_ylabel(labels[j])
    # Set ax limits to maximum

    # Look at the generalized coordinates
    helices = []
    for strand in target_strands[:1]:
        helix = RodHelixConverter.rod_to_helix(strand, np.zeros(strand.shape[0] - 1))
        # helix = RodHelixConverter.rod_to_helix_pos(strand, helix.n0, helix.q)
        helices.append(helix)
    # target_strands = [target_strands[0]]
    fig, ax = plt.subplots(3, 1)
    for i, helix in enumerate(helices):
        alpha = 0.3 if i != 0 else 1
        ax[0].plot(helix.q[0::3], color="C0", alpha=alpha)
        ax[1].plot(helix.q[1::3], color="C1", alpha=alpha)
        ax[2].plot(helix.q[2::3], color="C2", alpha=alpha)
        labels = ["twist", "bend1", "bend2"]
        for j in range(3):
            ax[j].set_xticks([0, strand.shape[0] // 2, strand.shape[0]])
            ax[j].set_ylabel(labels[j])

    fig, ax = plt.subplots(3, 1)
    all_fft = np.zeros((len(helices), helices[0].n_sites, 3))
    for i, helix in enumerate(helices):
        all_fft[i, :, 0] = (helix.q[0::3])
        all_fft[i, :, 1] = (helix.q[1::3])
        all_fft[i, :, 2] = (helix.q[2::3])
        alpha = 0.3 if i != 0 else 1
        ax[0].plot(helix.q[0::3], color="C0", alpha=alpha)
        ax[1].plot(helix.q[1::3], color="C1", alpha=alpha)
        ax[2].plot(helix.q[2::3], color="C2", alpha=alpha)
        labels = ["twist", "bend1", "bend2"]
        for j in range(3):
            ax[j].set_xticks([0, strand.shape[0] // 2, strand.shape[0]])
            ax[j].set_ylabel(labels[j])



    # From one strand, create new strands by perturbing
    new_strands = []

    # for j in range(15):
    #     base_strand = target_strands[0].copy()
    #     base_x, base_y, base_z = fft(base_strand[:, 0]), fft(base_strand[:, 1]), fft(base_strand[:, 2])
    #     window_size = 2
    #     n_windows = base_strand.shape[0] // window_size
    #     for i in range(n_windows):
    #         start = i * window_size
    #         end = (i + 1) * window_size
    #         std_x, std_y, std_z = np.std(target_fft[:, start:end, 0]), np.std(target_fft[:, start:end, 1]), np.std(
    #             target_fft[:, start:end, 2])
    #         if i > 10:
    #             std_x, std_y, std_z = 0.0, 0.0, 0.0
    #         base_x[start:end] += np.random.normal(0, std_x, base_x[start:end].shape)
    #         base_y[start:end] += np.random.normal(0, std_y, base_y[start:end].shape)
    #         base_z[start:end] += np.random.normal(0, std_z, base_z[start:end].shape)
    #     base_strand[:, 0] = ifft(base_x).real
    #     base_strand[:, 1] = ifft(base_y).real
    #     base_strand[:, 2] = ifft(base_z).real
    #     new_strands.append(base_strand)

    new_helices = []
    base_helix = helices[0]
    n_new_strands = 2000
    for j in range(0, n_new_strands):
        helix_new = HelixUtil.copy_helix(base_helix)
        q = helix_new.q
        # Take fft of each dimension
        fft_t, fft_k1, fft_k2 = (q[0::3]), (q[1::3]), (q[2::3])
        # Perturb each
        window_size = 16
        n_windows = helix_new.n_sites // window_size
        if j != 0:
            for i in range(n_windows):
                start = i * window_size
                end = (i + 1) * window_size
                std_t, std_k1, std_k2 = np.std(all_fft[:, start:end, 0]), np.std(all_fft[:, start:end, 1]), np.std(
                    all_fft[:, start:end, 2])
                fft_t[start:end] += np.random.normal(0, std_t, fft_t[start:end].shape)
                fft_k1[start:end] += np.random.normal(0, std_k1, fft_k1[start:end].shape)
                fft_k2[start:end] += np.random.normal(0, std_k2, fft_k2[start:end].shape)
        # Inverse fft
        q[0::3] = fft_t.real
        q[1::3] = fft_k1.real
        q[2::3] = fft_k2.real

        new_helices.append(helix_new)

    for helix in new_helices:
        pos, theta = RodHelixConverter.helix_to_rod(helix)
        new_strands.append(pos)

    # --- post processing of new strands ---
    new_strands = np.array(new_strands)
    new_strands, centroids, directions = Preprocess.align_data(new_strands)
    for i in range(new_strands.shape[0]):
        strand = new_strands[i]
        if strand[0, 2] > strand[-1, 2]:
            strand[:, 2] = -strand[:, 2]
        # strand[:, 0] += 20 * i + 10
        new_strands[i] = strand
    new_strands[:, :, 0] += 50

    for i, strand in enumerate(target_strands):
        strand[:, 1] -= 20 * i
    for i, strand in enumerate(new_strands):
        strand[:, 1] -= 20 * i


    # Add all target strands
    strands_vis = np.concatenate((target_strands, new_strands))
    strands_to_one_objs(strands_vis, frame_idx=1)

    np.save("synthetic_strands.npy", strands_vis)

    plt.show()


if __name__ == "__main__":
    main()
