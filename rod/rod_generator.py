import numpy as np
from rod.rod import Rod


class RodGenerator:
    """ A class for generating rods """

    @staticmethod
    def example_rod(n: int):
        # n + 2 vertices
        vertices = []
        d = 0.75
        for i in range(n + 2):
            pos = np.array([np.cos(d * i), np.sin(d * i), i], dtype=np.float64)
            vertices.append(pos)

        # Reverse so that the last node is at the top
        vertices.reverse()

        # n + 1 edges
        thetas = []
        for i in range(n + 1):
            thetas.append(np.random.rand() * 0.0)
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
        vertices.reverse()
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
