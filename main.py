import numpy as np

from math_util.rotation import RotationUtil
from rod.RodHelixConverter import RodHelixConverter
from rod.helix import Helix
from rod.helix_util import HelixUtil
from rod.rod_generator import RodGenerator
from rod.rod_util import RodUtil
from visualization.visualizer import Visualizer


def create_frame(pos: np.ndarray,
                 material_frame: np.ndarray,
                 point_radii: np.ndarray,
                 ax1_radii: np.ndarray,
                 ax2_radii: np.ndarray,
                 point_style: list[str],
                 frame_idx: int):
    print(f"Creating frame {frame_idx}")
    Visualizer.strand_to_obj(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                             ax2_radii=ax2_radii, point_style=point_style,
                             output_file=f"output/obj/obj_{frame_idx}.obj")
    return


def create_frame_helix(helix: Helix, point_radii: np.ndarray, ax1_radii: np.ndarray, ax2_radii: np.ndarray,
                       point_style: list[str], frame_idx: int):
    r, n = HelixUtil.propagate(helix)
    # Interpolate the material frame between the two sites
    material_frame = np.zeros((n.shape[0] - 1, 2, 3))
    for i in range(n.shape[0] - 1):
        rotation = RotationUtil.compute_rotation_matrix(n[i], n[i + 1])
        rotation = RotationUtil.interpolate_rotation(rotation, 0.5)
        interpolated_frame = rotation @ n[i]
        material_frame[i] = interpolated_frame[1:]

    create_frame(pos=r, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=frame_idx)
    return


def main():
    # Import rod
    pos, theta = RodGenerator.from_obj(file_path="../../blender/sarah_1.obj", scale=10)
    bishop_frame = RodUtil.compute_bishop_frames(pos=pos, m0=None)
    material_frame = RodUtil.compute_material_frames(theta=theta, bishop_frame=bishop_frame)

    n_pts = pos.shape[0]
    # Stiffness, mass constants. Revisit this
    EI = np.ones(3 * n_pts) * 1
    rhoS = 0.05
    g = 9.81 * 1e-3

    # Drawing parameters
    point_radii = 0.1 * np.ones(n_pts)
    ax1_radii = 0.2 * np.ones(n_pts)
    ax2_radii = 0.1 * np.ones(n_pts)
    point_style = ["sphere"] * n_pts

    create_frame(pos=pos, material_frame=material_frame, point_radii=point_radii, ax1_radii=ax1_radii,
                 ax2_radii=ax2_radii, point_style=point_style, frame_idx=0)

    # Convert to helix
    init_bishop_frame = bishop_frame[0]
    e0 = pos[1] - pos[0]
    t0 = e0 / np.linalg.norm(e0)
    # n0 = np.array([t0, init_bishop_frame[0], init_bishop_frame[1]])
    helix_import = RodHelixConverter.rod_to_helix(pos, theta)
    create_frame_helix(helix_import, point_radii, ax1_radii, ax2_radii, point_style, frame_idx=1)


if __name__ == "__main__":
    main()
