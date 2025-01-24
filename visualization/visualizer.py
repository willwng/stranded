import numpy as np

from rod.rod import Rod


class Visualizer:
    @staticmethod
    def draw_nodes(rod: Rod, ax):
        pos = rod.pos
        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], c='k', marker='o', s=0.1)
        # Pinned node, last node
        ax.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2], c='b', marker='o', s=10)
        return

    @staticmethod
    def draw_edges(rod: Rod, ax):
        pos = rod.pos
        for i in range(rod.n + 1):
            ax.plot([pos[i, 0], pos[i + 1, 0]],
                    [pos[i, 1], pos[i + 1, 1]],
                    [pos[i, 2], pos[i + 1, 2]],
                    c='k')
        return

    @staticmethod
    def draw_bishop_frame(rod: Rod, ax):
        bishop_frame = rod.bishop_frame
        pos = rod.pos
        for i in range(rod.n + 1):
            e_pos = (pos[i] + pos[i + 1]) / 2
            u = bishop_frame[i, 0]
            v = bishop_frame[i, 1]
            ax.quiver(e_pos[0], e_pos[1], e_pos[2], u[0], u[1], u[2], color='r')
            ax.quiver(e_pos[0], e_pos[1], e_pos[2], v[0], v[1], v[2], color='g')
        return

    @staticmethod
    def draw_material_frame(rod: Rod, ax):
        pos = rod.pos
        theta = rod.theta
        bishop_frame = rod.bishop_frame

        for i in range(rod.n + 1):
            e_pos = (pos[i] + pos[i + 1]) / 2
            u = bishop_frame[i, 0]
            v = bishop_frame[i, 1]
            theta_i = theta[i]
            m1, m2 = np.cos(theta_i) * u + np.sin(theta_i) * v, -np.sin(theta_i) * u + np.cos(theta_i) * v
            ax.quiver(e_pos[0], e_pos[1], e_pos[2], m1[0], m1[1], m1[2], color='r')
            ax.quiver(e_pos[0], e_pos[1], e_pos[2], m2[0], m2[1], m2[2], color='g')
        return
