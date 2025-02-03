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
    def create_cube(center, size, segments=1):
        """ Create a cube mesh centered at the given point """
        vertices = []
        faces = []

        # Half-length of the cube
        h = size / 2

        # Generate vertices for each segment
        for face in range(6):  # 6 faces of the cube
            # Determine the primary axis for this face
            axis = face // 2  # 0=x, 1=y, 2=z
            sign = -1 if face % 2 == 0 else 1

            # Generate grid of vertices for this face
            for i in range(segments + 1):
                for j in range(segments + 1):
                    # Convert grid coordinates to [-h, h] range
                    u = -h + (2 * h * i / segments)
                    v = -h + (2 * h * j / segments)

                    # Create vertex based on face orientation
                    vertex = [0, 0, 0]
                    vertex[axis] = sign * h  # Primary axis
                    vertex[(axis + 1) % 3] = u  # Secondary axis
                    vertex[(axis + 2) % 3] = v  # Tertiary axis

                    # Offset by center position
                    vertex = [vertex[i] + center[i] for i in range(3)]
                    vertices.append(vertex)

        # Generate faces (triangles)
        verts_per_face = (segments + 1) * (segments + 1)
        for face in range(6):
            face_offset = face * verts_per_face

            # Generate grid of triangles
            for i in range(segments):
                for j in range(segments):
                    # Get vertex indices for this grid cell
                    v1 = face_offset + i * (segments + 1) + j
                    v2 = v1 + 1
                    v3 = v1 + (segments + 1)
                    v4 = v3 + 1

                    # Create two triangles for this grid cell
                    if face % 2 == 0:
                        faces.append([v1, v2, v3])
                        faces.append([v2, v4, v3])
                    else:
                        faces.append([v1, v3, v2])
                        faces.append([v2, v3, v4])

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
        major_norm = np.linalg.norm(a_dir)
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
