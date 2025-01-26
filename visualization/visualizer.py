import numpy as np


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
