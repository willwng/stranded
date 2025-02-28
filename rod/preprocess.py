import numpy as np


class Preprocess:
    @staticmethod
    def batch_pca_fit_lines(points):
        # Calculate centroids (mean points) for each polyline
        centroids = np.mean(points, axis=1)  # Shape: (batch_size, 3)
        # Center the points by subtracting centroids
        centered_points = points - centroids[:, np.newaxis, :]  # Shape: (batch_size, num_samples, 3)
        # Initialize an array to store direction vectors
        directions = np.zeros((points.shape[0], 3))
        # Perform PCA on each polyline
        for i in range(points.shape[0]):
            # Get the centered points for the current polyline
            A = centered_points[i]  # Shape: (num_samples, 3)
            # Apply SVD to find the principal components
            _, _, vh = np.linalg.svd(A)
            # The first principal component (largest variance) is the best-fit line direction
            directions[i] = vh[0]  # The first row in vh is the principal direction vector
        return centroids, directions

    @staticmethod
    def rotation_matrix_from_vectors(vec1, vec2):
        """ Find the rotation matrix that aligns vec1 to vec2
        :param vec1: A 3d "source" vector
        :param vec2: A 3d "destination" vector
        :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
        """
        a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        k_mat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + k_mat + k_mat.dot(k_mat) * ((1 - c) / (s ** 2))
        return rotation_matrix

    @staticmethod
    def write_obj_file(vertices: np.ndarray, filename="output.obj"):
        """ Write a strand (list of vertices) to an obj file """
        with open(filename, 'w') as f:
            for v in vertices.T:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for i in range(1, vertices.shape[1]):
                f.write(f"l {i} {i + 1}\n")
        return

    @staticmethod
    def align(all_strands: np.ndarray, all_centroids: np.ndarray, all_directions: np.ndarray):
        """ Aligns all strand centroid to origin and direction facing z-axis """
        all_strands_rot = np.zeros_like(all_strands)
        for i, (strand, centroid, direction) in enumerate(zip(all_strands, all_centroids, all_directions)):
            strand_centered = strand - centroid
            rot_mat = Preprocess.rotation_matrix_from_vectors(direction, np.array([0, 0, 1.]))
            all_strands_rot[i] = rot_mat.dot(strand_centered.T).T

        return all_strands_rot

    @staticmethod
    def parse_obj(file_path: str, max_num_points: int):
        """ Parse an obj file and return an array of strand (shape num_strands, num_points, 3) """
        vertex_positions = []
        edges = []
        with open(file_path, 'r') as file:
            for line in file:
                if line.startswith('v '):
                    parts = line.split()
                    vertex_pos = list(map(float, parts[1:4]))
                    vertex_positions.append(vertex_pos)
                elif line.startswith('l '):
                    parts = line.split()
                    edge = list(map(int, parts[1:]))
                    for i in range(0, len(edge) - 1):
                        edges.append((edge[i] - 1, edge[i + 1] - 1))  # -1 to convert to 0-based index
                pass

        # Build directed adjacency list
        adjacency_list = {v: [] for v in range(len(vertex_positions))}
        vertex_to_num_parents = {v: 0 for v in range(len(vertex_positions))}
        for edge in edges:
            adjacency_list[edge[0]].append(edge[1])
            vertex_to_num_parents[edge[1]] += 1

        # Get all starting vertices
        starting_vertices = [v for v in vertex_to_num_parents if vertex_to_num_parents[v] == 0]

        # Convert adjacency list to strands
        visited = set()
        strands = []
        vertex_positions = np.array(vertex_positions)
        for vertex_idx in starting_vertices:
            strand = []
            # Traverse the strand
            while True:
                strand.append(vertex_positions[vertex_idx])
                visited.add(vertex_idx)
                if not adjacency_list[vertex_idx]:
                    break
                vertex_idx = adjacency_list[vertex_idx][0]
            strands.append(strand)

        # Cut strands to be same length
        min_strand_length = min(min([len(strand) for strand in strands]), max_num_points)
        strands = [strand[:min_strand_length] for strand in strands]

        strand_data = np.array(strands)
        return strand_data

    @staticmethod
    def align_data(data):
        """ Aligns the centerlines and curls centroids to origin and directions facing z-axis """
        # Centroids and directions are computed using centerline (not curl)
        centroids, directions = Preprocess.batch_pca_fit_lines(data)
        data_aligned = Preprocess.align(data, centroids, directions)
        return data_aligned, centroids, directions
