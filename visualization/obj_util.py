import numpy as np


class ObjUtil:
    """ Tools for creating 3d meshes in the Wavefront OBJ format """

    @staticmethod
    def create_sphere(center, radius, segments=8):
        """Create a sphere mesh centered at the given point"""
        vertices = []
        faces = []

        # Generate vertices
        for i in range(segments + 1):
            lat = np.pi * (-0.5 + float(i) / segments)
            for j in range(segments):
                lon = 2 * np.pi * float(j) / segments
                x = center[0] + radius * np.cos(lat) * np.cos(lon)
                y = center[1] + radius * np.cos(lat) * np.sin(lon)
                z = center[2] + radius * np.sin(lat)
                vertices.append([x, y, z])

        # Generate faces
        for i in range(segments):
            for j in range(segments):
                v1 = i * segments + j
                v2 = i * segments + (j + 1) % segments
                v3 = (i + 1) * segments + (j + 1) % segments
                v4 = (i + 1) * segments + j
                faces.append([v1, v2, v3])
                faces.append([v1, v3, v4])

        return vertices, faces

    @staticmethod
    def create_cylinder(start, end, radius, segments=8):
        """Create a cylinder mesh between two points"""
        vertices = []
        faces = []

        # Calculate cylinder direction and length
        direction = end - start
        length = np.linalg.norm(direction)
        if length < 1e-6:
            return [], []

        direction = direction / length

        # Create an orthonormal basis
        if abs(direction[0]) < abs(direction[1]):
            right = np.cross(direction, [1, 0, 0])
        else:
            right = np.cross(direction, [0, 1, 0])
        right = right / np.linalg.norm(right)
        up = np.cross(direction, right)

        # Generate vertices
        for i in range(2):  # Two ends of cylinder
            for j in range(segments):
                angle = 2 * np.pi * j / segments
                normal = right * np.cos(angle) + up * np.sin(angle)
                point = (start if i == 0 else end) + normal * radius
                vertices.append(point)

        # Generate faces
        for i in range(segments):
            i1 = i
            i2 = (i + 1) % segments
            i3 = i + segments
            i4 = ((i + 1) % segments) + segments

            # Side faces
            faces.append([i1, i2, i4])
            faces.append([i1, i4, i3])

        return vertices, faces

    @staticmethod
    def create_elliptical_cylinder(start: np.ndarray, end: np.ndarray, a_dir: np.ndarray,
                                   a: float, b: float, segments: int):
        """
        Create an elliptical cylinder mesh between two points with one axis pointed in a_dir
            with length a and the other axis length b
        """
        vertices = []
        faces = []

        # Calculate cylinder direction and length
        cyl_dir = end - start
        cyl_dir = cyl_dir / np.linalg.norm(cyl_dir)

        # Project major_axis_dir onto plane perpendicular to cylinder axis
        a_dir = a_dir - np.dot(a_dir, cyl_dir) * cyl_dir

        # Check if major_axis_dir is not zero after projection
        major_norm = np.linalg.norm(a_dir)
        if major_norm < 1e-6:
            raise ValueError("Major axis direction cannot be parallel to cylinder axis")

        a_dir = a_dir / major_norm

        # Create minor axis direction perpendicular to both cylinder axis and major axis
        minor_axis_dir = np.cross(cyl_dir, a_dir)
        minor_axis_dir = minor_axis_dir / np.linalg.norm(minor_axis_dir)

        # Generate vertices
        for i in range(2):  # Two ends of cylinder
            for j in range(segments):
                angle = 2 * np.pi * j / segments
                # Create elliptical profile using parametric equation
                normal = (a_dir * (a * np.cos(angle)) +
                          minor_axis_dir * (b * np.sin(angle)))
                point = (start if i == 0 else end) + normal
                vertices.append(point)

        # Generate faces
        for i in range(segments):
            i1 = i
            i2 = (i + 1) % segments
            i3 = i + segments
            i4 = ((i + 1) % segments) + segments

            # Side faces
            faces.append([i1, i2, i4])
            faces.append([i1, i4, i3])

        return vertices, faces
