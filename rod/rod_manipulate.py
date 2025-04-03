import numpy as np

class RodManipulator:
    @staticmethod
    def compress_x(pos, compression_factor):
        # DOES NOT PRESERVE EDGE LENGTHS
        compressed_pos = pos.copy()
        compressed_pos[:, 0] *= (1 + compression_factor)
        return compressed_pos
    
    @staticmethod
    def scrunch_rod(vertices: np.ndarray, compression_factor: float, iterations: int = 100):
        # Step 1: Store original edge lengths
        edges = np.linalg.norm(np.diff(vertices, axis=0), axis=1)

        # Step 2: Apply vertical compression
        vertices[:, 1] *= compression_factor

        # Step 3: Iteratively project onto length constraints
        for _ in range(iterations):
            for i in range(1, len(vertices)):
                diff = vertices[i] - vertices[i - 1]
                dist = np.linalg.norm(diff)
                correction = (dist - edges[i - 1]) / dist * diff
                vertices[i] -= correction / 2
                vertices[i - 1] += correction / 2

        return vertices

    @staticmethod
    def compress_y(pos, compression_factor):
        """
        Compresses the rod in the y-directsion while preserving edge inextensibility.
        Adjusts x and z coordinates (radial displacement) to compensate.

        Parameters:
            pos (np.ndarray): (N, 3) array of vertex positions [x, y, z].
            compression_factor (float): Factor by which the y-coordinates are compressed.

        Returns:
            np.ndarray: Transformed positions maintaining edge inextensibility.
        """
        n_points = pos.shape[0]
        
        # Compute original edge lengths (segment distances)
        edge_vectors = np.diff(pos, axis=0)
        edge_lengths = np.linalg.norm(edge_vectors, axis=1)  # Edge lengths should remain unchanged
        total_length = np.sum(edge_lengths)

        # Compute new y-coordinates (compressed)
        compressed_total_length = total_length / compression_factor
        cumulative_lengths = np.hstack(([0], np.cumsum(edge_lengths)))  # Arc-length parameterization
        new_y = (cumulative_lengths / total_length) * compressed_total_length  # Rescaled y

        # Compute radial displacement in (x, z)
        r_original = np.sqrt(edge_vectors[:, 0]**2 + edge_vectors[:, 2]**2) + 1e-8  # Original radial distance, never zero
        y_mag_original = np.abs(np.diff(pos, axis=0)[:, 1]) # Magnitude of y-displacement
        term_1 = (edge_lengths[:] ** 2) / (r_original[:] ** 2)
        term_2 = (y_mag_original ** 2) / ((r_original[:] ** 2) * compression_factor ** 2)
        r_scale_factor = np.sqrt(term_1 - term_2)  # Scale factor for radial displacement
        r_new = r_original * r_scale_factor  # Shape: (N,)
        r_new = r_original * r_scale_factor  # Scale r to preserve arc-length

        # Compute new (x, z) positions
        zeta = np.arctan2(pos[1:, 2], pos[1:, 0])  # Angle in the x-z plane, not changing first point
        zeta = np.insert(zeta, -1, 0)  # Add the angle of the last point (clamped)
        r_new = np.insert(r_new, -1, r_original[0]) # Add radius of last point (clamped)
        new_x = r_new * np.cos(zeta)
        new_z = r_new * np.sin(zeta)

        # Apply transformations
        compressed_pos = pos.copy()
        compressed_pos[:, 1] = new_y  # Update y-coordinates
        compressed_pos[:, 0] = new_x  # Update x-coordinates
        compressed_pos[:, 2] = new_z  # Update z-coordinates

        return compressed_pos
