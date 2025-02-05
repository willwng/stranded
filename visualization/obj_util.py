import numpy as np


class ObjUtil:
    """ Tools for creating 3d meshes in the Wavefront OBJ format """

    @staticmethod
    def create_sphere(center, radius, segments=16):
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
    def create_cylinder(start, end, radius, segments=16):
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

    @staticmethod
    def create_cone(base_center, tip, radius, segments=16):
        """ Create a cone mesh from base center to tip. """
        base_center = np.array(base_center)
        tip = np.array(tip)

        # Calculate cone axis and height
        axis = tip - base_center
        height = np.linalg.norm(axis)
        if height < 1e-6:
            raise ValueError("Cone height must be non-zero")

        # Normalize axis
        axis = axis / height

        # Find perpendicular vectors to create circle
        if abs(axis[0]) < abs(axis[1]):
            perpendicular = np.array([1., 0., 0.])
        else:
            perpendicular = np.array([0., 1., 0.])

        # Create orthonormal basis
        u = np.cross(axis, perpendicular)
        u = u / np.linalg.norm(u)
        v = np.cross(axis, u)

        # Generate vertices
        vertices = [tip.tolist()]  # First vertex is the tip

        # Create base vertices
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            circle_point = (u * np.cos(angle) + v * np.sin(angle)) * radius
            vertex = base_center + circle_point
            vertices.append(vertex.tolist())

        # Create faces
        faces = []

        # Side faces
        for i in range(segments):
            v1 = 0  # Tip vertex
            v2 = i + 1
            v3 = ((i + 1) % segments) + 1
            faces.append([v1, v2, v3])

        # Base face (triangulate the base)
        base_center_idx = len(vertices)
        vertices.append(base_center.tolist())

        for i in range(segments):
            v1 = base_center_idx
            v2 = i + 1
            v3 = ((i + 1) % segments) + 1
            faces.append([v1, v3, v2])  # Note: reversed order for correct normal

        return vertices, faces

    @staticmethod
    def create_arrow(start_point, direction, length=1.0, radius=0.1):
        """ Create an arrow mesh consisting of a cylinder and cone. """
        # Normalize direction vector
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)

        # Create shaft (cylinder)
        shaft_length = length * 0.8  # Shaft takes 80% of total length
        cylinder_vertices, cylinder_faces = ObjUtil.create_cylinder(
            start_point,
            start_point + direction * shaft_length,
            radius
        )

        # Create arrowhead (cone)
        cone_height = length * 0.2  # Cone takes 20% of total length
        cone_radius = radius * 2  # Cone is wider than shaft
        cone_start = start_point + direction * shaft_length
        cone_vertices, cone_faces = ObjUtil.create_cone(
            cone_start,
            cone_start + direction * cone_height,
            cone_radius
        )

        # Combine vertices and update face indices for cone
        all_vertices = cylinder_vertices + cone_vertices
        cone_faces = [[f + len(cylinder_vertices) for f in face] for face in cone_faces]
        all_faces = cylinder_faces + cone_faces

        return all_vertices, all_faces
