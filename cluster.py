import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from main import strands_to_one_objs, plot_generalized_coords
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


def main(): # Works with strands w/ different number of points
    # Load strand data
    # curl_strands = np.load("curl.npy")[:5]
    # centerline_strands = np.load("centerline.npy")[:5]
    # n_curl_strands = curl_strands.shape[0]
    # strands = np.concatenate((curl_strands, centerline_strands))

    ### i change
    # keys are: strands, centerlines, straightened_strands, straightened_cls
    strands = np.load('apple_afro_full.npy', allow_pickle=True).item()
    # print(strands['straightened_strands'])
    curl_strands = strands['straightened_strands']
    centerline_strands = strands['straightened_cls']
    strands = np.concatenate((curl_strands, centerline_strands))
    # strands = curl_strands + centerline_strands
    print(f"Strands shape: {curl_strands[1].shape}")
    strands_to_one_objs(strands, frame_idx=1, y_up=True)

    new_curl_strands = curl_strands
    

    ################ - Code to redistribute strands in virtual space
    # lines original strands up in a line
    # strands, centroids, directions = Preprocess.align_data(strands)
    # for i in range(strands.shape[0]):
    #     strand = strands[i]
    #     # strand = RodHelixConverter.normalize_strand(strand, normalize_positions=False, normalize_direction=False, )
    #     strand[:, 0] += 0.02 * (i % n_curl_strands)
    #     strand[:, 2] *= -1
    #     # reverse order
    #     strand = strand[::-1]
    #     strands[i] = strand
    # curl_strands, centerline_strands = strands[:n_curl_strands], strands[n_curl_strands:]
    # strands_to_one_objs(strands, frame_idx=1, y_up=True)

    # diffs = curl_strands - centerline_strands
    # new_centerline_strands = centerline_strands.copy()
    # # For each centerline strand, make a vertical copy
    # for i in range(new_centerline_strands.shape[0]):
    #     strand = new_centerline_strands[i]
    #     # Center at mean
    #     mean_x, mean_y = np.mean(strand[:, 0]), np.mean(strand[:, 1])
    #     strand[:, 0] = mean_x
    #     strand[:, 1] = mean_y
    #     # Make min z 0
    #     strand[:, 2] -= np.min(strand[:, 2])
    #     new_centerline_strands[i] = strand
    # new_curl_strands = new_centerline_strands + diffs
    # new_strands = np.concatenate((new_curl_strands, new_centerline_strands))
    # strands_to_one_objs(new_strands, frame_idx=2, y_up=True)
    ###############

    helices = []
    qs = []
    fig, ax = plt.subplots(3, 1)
    ax = ax.ravel()
    for curl_strand in new_curl_strands:
        # Solve for the first material frame
        helix_init = RodHelixConverter.rod_to_helix(curl_strand, np.zeros(curl_strand.shape[0] - 1))
        # Then solve for the arc lengths
        s = RodHelixConverter.rod_to_helix_pos(curl_strand, n0=helix_init.n0, q_guess=helix_init.q).s
        # Then, resolve the generalized coordinates
        helix = RodHelixConverter.rod_to_helix(curl_strand, np.zeros(curl_strand.shape[0] - 1), s=s)
        # Plot
        alpha = 0.3
        ax[0].plot(helix.q[0::3], color="C0", alpha=alpha)
        ax[1].plot(helix.q[1::3], color="C1", alpha=alpha)
        ax[2].plot(helix.q[2::3], color="C2", alpha=alpha)
        qs.append(helix.q)
        helices.append(helix)
    qs = np.array(qs) #- keeping qs as a sequence instead of converting to an array
    print(qs.shape)

    # Make some perturbations
    perturbed_helices = [[] for _ in range(len(helices))]
    n_perturbations = 10

    # finding global stats when have strands of different point #'s
    # tau_vals   = np.concatenate([q[0::3] for q in qs])
    # bend1_vals = np.concatenate([q[1::3] for q in qs])
    # bend2_vals = np.concatenate([q[2::3] for q in qs])
    for i, helix in enumerate(helices):
        for _ in range(n_perturbations):
            tau_std, bend1_std, bend2_std = (np.std(qs[:, 0::3], axis=0),
                                             np.std(qs[:, 1::3], axis=0),
                                             np.std(qs[:, 2::3], axis=0))

            # alternatives for strands of different point #'s
            # tau_std = np.nanstd(tau_vals)
            # bend1_std = np.nanstd(bend1_vals)
            # bend2_std = np.nanstd(bend2_vals)
            perturbed_helix = HelixUtil.copy_helix(helix)
            perturbed_q = perturbed_helix.q
            perturbed_q[0::3] += np.random.normal(0, tau_std)
            perturbed_q[1::3] += np.random.normal(0, bend1_std)
            perturbed_q[2::3] += np.random.normal(0, bend2_std)
            perturbed_helix.q = perturbed_q
            perturbed_helices[i].append(perturbed_helix)
    print(f'stds: {tau_std, bend1_std, bend2_std}')

    converted_strands = []
    for i in range(len(helices)):
        perturbed_helices_i = perturbed_helices[i] # n_perturbations helices per original helix
        print(f"perturbed helices {i}: {len(perturbed_helices_i)}")
        # orig_helix = helices[i]
        for j, helix in enumerate(perturbed_helices_i): # + [orig_helix]):
            # helix.r0[1] += 0.01 * j
            pos, theta = RodHelixConverter.helix_to_rod(helix)
            converted_strands.append(pos)
    converted_strands = np.array(converted_strands)
    print(f"Converted strands shape: {converted_strands.shape}")
    # print(f"Z coords:{converted_strands[:, 0, 2]}")
    strands_to_one_objs(converted_strands, frame_idx=3, y_up=True)

    min_twist, min_bend1, min_bend2 = np.min(qs[:, 0::3]), np.min(qs[:, 1::3]), np.min(qs[:, 2::3])
    max_twist, max_bend1, max_bend2 = np.max(qs[:, 0::3]), np.max(qs[:, 1::3]), np.max(qs[:, 2::3])
    # min_twist, min_bend1, min_bend2 = np.min(tau_vals), np.min(bend1_vals), np.min(bend2_vals)
    # max_twist, max_bend1, max_bend2 = np.max(tau_vals), np.max(bend1_vals), np.max(bend2_vals)
    min_q, max_q = np.min([min_twist, min_bend1, min_bend2]), np.max([max_twist, max_bend1, max_bend2])
    print(min_twist, min_bend1, min_bend2)
    ax[0].set_ylim([min_q, max_q])
    ax[1].set_ylim([min_q, max_q])
    ax[2].set_ylim([min_q, max_q])
    ax[0].set_ylabel("Twist")
    ax[1].set_ylabel("Bend 1")
    ax[2].set_ylabel("Bend 2")
    plt.show()

    print("saving perturbations...")
    np.save("apple_full_just_perturb.npy", np.array(converted_strands, dtype=object), allow_pickle=True) #allow pickle=True
    return


if __name__ == "__main__":
    main()
