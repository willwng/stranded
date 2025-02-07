import numpy as np


class RodGenerator:
    """ A class for generating rods """

    @staticmethod
    def example_rod(n: int):
        # n + 2 vertices
        vertices = []
        curl_radius = 1.0
        curl_frequency = 1.0
        height_scale = 0.2
        for i in range(n + 2):
            pos = np.array(
                [curl_radius * np.cos(curl_frequency * i), curl_radius * np.sin(curl_frequency * i), height_scale * i],
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
