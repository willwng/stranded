import cupy as cp

class RodGenerator:
    """ A class for generating rods on GPU using CuPy """

    @staticmethod
    def example_rod(n: int, curl_radius=1.0, curl_frequency=1.0, height_scale=0.2):
        vertices = []
        for i in range(n + 2):
            pos = cp.array([
                curl_radius * cp.cos(curl_frequency * i),
                curl_radius * cp.sin(curl_frequency * i),
                height_scale * i
            ], dtype=cp.float64)
            vertices.append(pos)

        # Reverse and stack
        vertices = cp.stack(vertices[::-1])  # reverse in one line

        # Translate so that pos[0] x and y are 0
        vertices[:, 0] -= vertices[0, 0]
        vertices[:, 1] -= vertices[0, 1]

        thetas = cp.zeros(n + 1, dtype=cp.float64)
        return vertices, thetas

    @staticmethod
    def straight_rod(n_points: int):
        vertices = [cp.array([0, 0, i], dtype=cp.float64) for i in range(n_points + 2)]
        vertices = cp.stack(vertices[::-1])
        thetas = cp.zeros(n_points + 1, dtype=cp.float64)
        return vertices, thetas

    @staticmethod
    def jittery_rod(n_points: int):
        vertices = [cp.array([0, 0, i], dtype=cp.float64) + cp.random.normal(0, 0.1, 3) for i in range(n_points + 2)]
        thetas = cp.array([cp.random.rand() for _ in range(n_points + 2)], dtype=cp.float64)[1:]
        vertices = cp.stack(vertices[::-1])
        return vertices, thetas

    @staticmethod
    def diagonal_rod(n_points: int):
        vertices = [cp.array([i, 0, i], dtype=cp.float64) for i in range(n_points + 2)]
        vertices = cp.stack(vertices[::-1])
        thetas = cp.zeros(n_points + 1, dtype=cp.float64)
        return vertices, thetas

