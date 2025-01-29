import numpy as np

from visualization.obj_util import ObjUtil


class Visualizer:
    @staticmethod
    def draw_nodes(pos, ax):
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='k', marker='o', s=0.1)
        # Pinned node, last node
        ax.scatter(pos[0, 0], pos[0, 1], pos[0, 2], c='b', marker='o', s=10)
        return

    @staticmethod
    def draw_edges(pos: np.ndarray, ax):
        for i in range(pos.shape[0] - 1):
            ax.plot([pos[i, 0], pos[i + 1, 0]],
                    [pos[i, 1], pos[i + 1, 1]],
                    [pos[i, 2], pos[i + 1, 2]],
                    c='k')
        return

    @staticmethod
    def draw_bishop_frame(pos: np.ndarray, bishop_frame: np.ndarray, ax):
        for i in range(pos.shape[0] - 1):
            e_pos = (pos[i] + pos[i + 1]) / 2
            u = bishop_frame[i, 0]
            v = bishop_frame[i, 1]
            ax.quiver(e_pos[0], e_pos[1], e_pos[2], u[0], u[1], u[2], color='r')
            ax.quiver(e_pos[0], e_pos[1], e_pos[2], v[0], v[1], v[2], color='g')
        return

    @staticmethod
    def draw_material_frame(pos: np.ndarray, theta: np.ndarray, bishop_frame: np.ndarray, ax):
        for i in range(pos.shape[0] - 1):
            e_pos = (pos[i] + pos[i + 1]) / 2
            u = bishop_frame[i, 0]
            v = bishop_frame[i, 1]
            theta_i = theta[i]
            m1, m2 = np.cos(theta_i) * u + np.sin(theta_i) * v, -np.sin(theta_i) * u + np.cos(theta_i) * v
            ax.quiver(e_pos[0], e_pos[1], e_pos[2], m1[0], m1[1], m1[2], color='r')
            ax.quiver(e_pos[0], e_pos[1], e_pos[2], m2[0], m2[1], m2[2], color='g')
        return

    @staticmethod
    def set_lims(pos: np.ndarray, ax):
        max_height = np.max(pos[:, 2])
        ax.set_xlim([-max_height / 2, max_height / 2])
        ax.set_ylim([-max_height / 2, max_height / 2])
        ax.set_zlim([0, max_height + 1])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        return

    @staticmethod
    def to_simple_obj(pos: np.ndarray, output_file: str):
        """ OBJ output with just points and edges """
        with open(output_file, 'w') as f:
            # Write header
            f.write("# Point cloud converted to OBJ\n")

            # Write vertices
            for i, point in enumerate(pos):
                # Write vertex position
                f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}")
                f.write("\n")

            # Write edges between consecutive points
            f.write("\n# Edges (lines)\n")
            for i in range(len(pos) - 1):
                # OBJ indices start at 1, so we add 1 to our zero-based indices
                f.write(f"l {i + 1} {i + 2}\n")
        return

    @staticmethod
    def strand_to_obj(pos: np.ndarray,
                      material_frame: np.ndarray,
                      output_file: str,
                      point_radius: float,
                      major_radius: float,
                      minor_radius: float,
                      y_up: bool = True
                      ):
        """ OBJ output with spheres for points and cylinders for lines """
        # Objs use the convention of y-up, but our simulation uses z-up
        if y_up:
            pos = pos[:, [0, 2, 1]]
            material_frame = material_frame[:, :, [0, 2, 1]]

        with open(output_file, 'w') as f:
            f.write("# Point cloud with 3D points and lines\n")

            vertex_offset = 1  # OBJ uses 1-based indexing

            # Create spheres for points
            for point in pos:
                sphere_vertices, sphere_faces = ObjUtil.create_sphere(point, point_radius)

                # Write sphere vertices
                for v in sphere_vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                # Write sphere faces
                for face in sphere_faces:
                    f.write(f"f {face[0] + vertex_offset} {face[1] + vertex_offset} {face[2] + vertex_offset}\n")

                vertex_offset += len(sphere_vertices)

            # Create cylinders for each edge
            for i in range(pos.shape[0] - 1):
                start, end = pos[i], pos[i + 1]

                # cyl_vertices, cyl_faces = ObjUtil.create_cylinder(start, end, line_radius)
                a_dir = material_frame[i, 0]
                cyl_vertices, cyl_faces = ObjUtil.create_elliptical_cylinder(
                    start=start, end=end, a_dir=a_dir, a=major_radius, b=minor_radius, segments=16)

                # Write cylinder vertices
                for v in cyl_vertices:
                    f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

                # Write cylinder faces
                for face in cyl_faces:
                    f.write(f"f {face[0] + vertex_offset} {face[1] + vertex_offset} {face[2] + vertex_offset}\n")

                vertex_offset += len(cyl_vertices)
