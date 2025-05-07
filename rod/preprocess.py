import cupy as cp
import numpy as np  # Still used for file I/O and string formatting

class Preprocess:
    @staticmethod
    def batch_pca_fit_lines(points):
        centroids = cp.mean(points, axis=1)
        centered_points = points - centroids[:, cp.newaxis, :]
        directions = cp.zeros((points.shape[0], 3))
        for i in range(points.shape[0]):
            A = centered_points[i]
            _, _, vh = cp.linalg.svd(A)
            directions[i] = vh[0]
        return centroids, directions

    @staticmethod
    def rotation_matrix_from_vectors(vec1, vec2):
        a = (vec1 / cp.linalg.norm(vec1)).reshape(3)
        b = (vec2 / cp.linalg.norm(vec2)).reshape(3)
        v = cp.cross(a, b)
        c = cp.dot(a, b)
        s = cp.linalg.norm(v)
        k_mat = cp.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])
        rotation_matrix = cp.eye(3) + k_mat + k_mat @ k_mat * ((1 - c) / (s ** 2))
        return rotation_matrix

    @staticmethod
    def write_obj_file(vertices, filename="output.obj"):
        vertices = cp.asnumpy(vertices)
        with open(filename, 'w') as f:
            for v in vertices.T:
                f.write(f"v {v[0]} {v[1]} {v[2]}\n")
            for i in range(1, vertices.shape[1]):
                f.write(f"l {i} {i + 1}\n")

    @staticmethod
    def align(all_strands, all_centroids, all_directions):
        all_strands_rot = cp.zeros_like(all_strands)
        for i, (strand, centroid, direction) in enumerate(zip(all_strands, all_centroids, all_directions)):
            strand_centered = strand - centroid
            rot_mat = Preprocess.rotation_matrix_from_vectors(direction, cp.array([0, 0, 1.]))
            all_strands_rot[i] = (rot_mat @ strand_centered.T).T
        return all_strands_rot

    @staticmethod
    def parse_obj(file_path: str, max_num_points: int):
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
                        edges.append((edge[i] - 1, edge[i + 1] - 1))

        adjacency_list = {v: [] for v in range(len(vertex_positions))}
        vertex_to_num_parents = {v: 0 for v in range(len(vertex_positions))}
        for edge in edges:
            adjacency_list[edge[0]].append(edge[1])
            vertex_to_num_parents[edge[1]] += 1

        starting_vertices = [v for v in vertex_to_num_parents if vertex_to_num_parents[v] == 0]
        visited = set()
        strands = []
        vertex_positions = np.array(vertex_positions)

        for vertex_idx in starting_vertices:
            strand = []
            while True:
                strand.append(vertex_positions[vertex_idx])
                visited.add(vertex_idx)
                if not adjacency_list[vertex_idx]:
                    break
                vertex_idx = adjacency_list[vertex_idx][0]
            strands.append(strand)

        min_strand_length = min(min([len(strand) for strand in strands]), max_num_points)
        strands = [strand[:min_strand_length] for strand in strands]
        strand_data = cp.array(strands)
        return strand_data

    @staticmethod
    def align_data(data):
        centroids, directions = Preprocess.batch_pca_fit_lines(data)
        data_aligned = Preprocess.align(data, centroids, directions)
        return data_aligned, centroids, directions
