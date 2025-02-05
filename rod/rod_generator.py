import numpy as np


class RodGenerator:
    """ A class for generating rods """

    @staticmethod
    def example_rod(n: int):
        # n + 2 vertices
        vertices = []
        curl_radius = 1.5
        curl_frequency = 0.3
        height_scale = 0.5
        for i in range(n + 2):
            pos = np.array(
                [0 * np.cos(curl_frequency * i), curl_radius * np.sin(curl_frequency * i), height_scale * i],
                dtype=np.float64)
            vertices.append(pos)

        # Reverse so that the last node is at the top
        vertices.reverse()
        vertices = np.stack(vertices)

        # Translate so that pos[0] x and y are 0
        vertices[:, 0] -= vertices[0, 0]
        vertices[:, 1] -= vertices[0, 1]

        # n + 1 edges
        thetas = []
        for i in range(n + 1):
            thetas.append(0.0)
        return np.array(vertices), np.array(thetas)

    @staticmethod
    def straight_rod(n_points: int):
        vertices = []
        thetas = []
        for i in range(n_points + 2):
            pos = np.array([0, 0, i], dtype=np.float64)
            thetas.append(0.0)
            vertices.append(pos)
        vertices.reverse()
        thetas = thetas[1:]
        return np.array(vertices), np.array(thetas)

    @staticmethod
    def jittery_rod(n_points: int):
        vertices = []
        thetas = []
        for i in range(n_points + 2):
            pos = np.array([0, 0, i], dtype=np.float64) + np.random.normal(0, 0.1, 3)
            thetas.append(np.random.rand())
            vertices.append(pos)
        vertices.reverse()
        thetas = thetas[1:]
        return np.array(vertices), np.array(thetas)

    @staticmethod
    def diagonal_rod(n_points: int):
        vertices = []
        thetas = []
        for i in range(n_points + 2):
            pos = np.array([i, 0, i], dtype=np.float64)
            thetas.append(0.0)
            vertices.append(pos)
        vertices.reverse()
        thetas = thetas[1:]
        return np.array(vertices), np.array(thetas)

    @staticmethod
    def evenly_spaced_helix(num_points: int, total_length: float, n_curls: float, curl_radius: float, offset: float):
        """Generate points on a helix with equal arc length spacing."""
        L = total_length
        spacing = total_length / (num_points - 1)

        # Compute phi (height increment)
        if L / (2 * np.pi * n_curls) < curl_radius:
            raise ValueError("Curl radius is too large for the given total length and number of curls.")
        phi = 2 * np.pi * np.sqrt((L / (2 * np.pi * n_curls)) ** 2 - curl_radius ** 2)

        # Differential arc length
        d_s = np.sqrt(curl_radius ** 2 + (phi / (2 * np.pi)) ** 2)

        # Generate points
        points = []
        for i in range(num_points):
            distance = i * spacing
            t_i = distance / d_s
            x = curl_radius * np.cos(t_i + offset)
            y = curl_radius * np.sin(t_i + offset)
            z = (phi / (2 * np.pi)) * t_i

            points.append([x, y, z])
        points.reverse()

        pos = np.array(points)
        thetas = np.zeros(num_points - 1)
        return pos, thetas

    @staticmethod
    def from_obj(file_path: str, scale: float, y_up: bool = True):
        # Load in the vertices and the edges (required for ordering) from the obj file
        pos = []
        edges = []
        with open(file_path, 'r') as f:
            for line in f:
                if line.startswith('v '):
                    pos.append(list(map(float, line.strip().split()[1:])))
                elif line.startswith('l '):
                    edges.append(list(map(int, line.strip().split()[1:])))
        pos = np.array(pos)
        edges = np.array(edges) - 1

        # Fix y-up vs z-up
        if y_up:
            pos = pos[:, [0, 2, 1]]

        # Scale the positions
        pos *= scale
        # Re-order so that connected points are adjacent in the listing
        pos = RodGenerator.reorder_vertices(pos, edges)
        # Redistribute the vertex positions to be evenly spaced along the connected path
        for _ in range(5):
            pos = RodGenerator.redistribute_vertices(pos)

        # Set thetas to zero (hard to determine twist)
        thetas = np.zeros(pos.shape[0] - 1)
        return np.array(pos), thetas

    @staticmethod
    def reorder_vertices(vertices, lines):
        """ Reorder vertices so that connected points are adjacent in the listing. """
        # Create adjacency list representation
        adj_list = {}
        for i in range(len(vertices)):
            adj_list[i] = set()

        for line in lines:
            v1, v2 = line
            adj_list[v1].add(v2)
            adj_list[v2].add(v1)

        # The starting vertex has one connection and the highest z value
        endpoints = [v for v in adj_list if len(adj_list[v]) == 1]
        start = max(endpoints, key=lambda x: vertices[x][2])

        # Traverse
        visited = set()
        ordered_indices = []
        current = start
        while len(visited) < len(vertices):
            ordered_indices.append(current)
            visited.add(current)

            # Find unvisited neighbor with fewest remaining connections
            next_vertex = None
            min_connections = float('inf')

            for neighbor in adj_list[current]:
                if neighbor not in visited and len(adj_list[neighbor]) < min_connections:
                    next_vertex = neighbor
                    min_connections = len(adj_list[neighbor])

            # If no unvisited neighbors, find nearest unvisited vertex
            if next_vertex is None and len(visited) < len(vertices):
                for v in range(len(vertices)):
                    if v not in visited:
                        next_vertex = v
                        break

            current = next_vertex

        new_vertices = [vertices[i] for i in ordered_indices]

        return np.array(new_vertices)

    @staticmethod
    def redistribute_vertices(vertices):
        """ Redistribute vertices to be evenly spaced along the connected path. """
        # Calculate current cumulative distances from one endpoint to another
        cum_distances = [0]
        total_distance = 0
        for i in range(1, len(vertices)):
            d = np.linalg.norm(vertices[i] - vertices[i - 1])
            total_distance += d
            cum_distances.append(total_distance)

        # Create new points at even intervals while following the original path
        n_points = len(vertices)
        new_vertices = []
        segment_length = total_distance / (n_points - 1)
        for i in range(n_points):
            target_distance = i * segment_length

            # Find which segment this distance falls on
            segment_idx = np.searchsorted(cum_distances, target_distance) - 1
            segment_idx = max(0, min(segment_idx, len(vertices) - 2))

            # Calculate interpolation factor
            segment_start = cum_distances[segment_idx]
            segment_end = cum_distances[segment_idx + 1]
            segment_fraction = (target_distance - segment_start) / (segment_end - segment_start)
            segment_fraction = max(0, min(segment_fraction, 1))  # Clamp to [0,1]

            # Interpolate position
            start_pos = vertices[segment_idx]
            end_pos = vertices[segment_idx + 1]
            new_pos = start_pos + segment_fraction * (end_pos - start_pos)
            new_vertices.append(new_pos)
        new_vertices = np.array(new_vertices)

        return new_vertices
