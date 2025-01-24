import numpy as np
from rod.rod import Rod


class RodGenerator:
    @staticmethod
    def example_rod(n: int):
        # n + 2 vertices
        vertices = []
        for i in range(n + 2):
            pos = np.array([np.cos(0.8 * i), np.sin(0.8 * i), i], dtype=np.float64)
            vertices.append(pos)

        # n + 1 edges
        thetas = []
        for i in range(n + 1):
            thetas.append(np.random.rand() * 0.2)
        rod = Rod(pos=np.array(vertices), thetas=np.array(thetas))
        return rod

    @staticmethod
    def straight_rod(n_points: int):
        vertices = []
        thetas = []
        for i in range(n_points + 1):
            pos = np.array([0, 0, i], dtype=np.float64)
            thetas.append(0.0)
            vertices.append(pos)
        thetas = thetas[1:]
        rod = Rod(pos=np.array(vertices), thetas=np.array(thetas))
        return rod

    @staticmethod
    def jittery_rod(n_points: int):
        vertices = []
        thetas = []
        for i in range(n_points + 1):
            pos = np.array([0, 0, i], dtype=np.float64) + np.random.normal(0, 0.1, 3)
            thetas.append(0.0)
            vertices.append(pos)
        thetas = thetas[1:]
        rod = Rod(pos=np.array(vertices), thetas=np.array(thetas))
        return rod
