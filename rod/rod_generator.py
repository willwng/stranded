import jax.numpy as jnp


class RodGenerator:
    """ A class for generating rods """

    @staticmethod
    def example_rod(n: int):
        # n + 2 vertices
        vertices = []
        d = 1.
        for i in range(n + 2):
            pos = jnp.array([jnp.cos(d * i), jnp.sin(d * i), 0.2 * i], dtype=jnp.float64)
            vertices.append(pos)

        # Reverse so that the last node is at the top
        vertices.reverse()
        vertices = jnp.stack(vertices)

        # Translate so that pos[0] x and y are 0
        vertices.at[:, 0].add(-vertices[0, 0])
        vertices.at[:, 1].add(-vertices[0, 1])

        # n + 1 edges
        thetas = []
        for i in range(n + 1):
            thetas.append(0.0)
        return jnp.array(vertices), jnp.array(thetas)

    @staticmethod
    def straight_rod(n_points: int):
        vertices = []
        thetas = []
        for i in range(n_points + 2):
            pos = jnp.array([0, 0, i], dtype=jnp.float64)
            thetas.append(0.0)
            vertices.append(pos)
        vertices.reverse()
        thetas = thetas[1:]
        return jnp.array(vertices), jnp.array(thetas)

    @staticmethod
    def jittery_rod(n_points: int):
        vertices = []
        thetas = []
        for i in range(n_points + 2):
            pos = jnp.array([0, 0, i], dtype=jnp.float64) + jnp.random.normal(0, 0.1, 3)
            thetas.append(jnp.random.rand())
            vertices.append(pos)
        vertices.reverse()
        thetas = thetas[1:]
        return jnp.array(vertices), jnp.array(thetas)

    @staticmethod
    def diagonal_rod(n_points: int):
        vertices = []
        thetas = []
        for i in range(n_points + 2):
            pos = jnp.array([i, 0, i], dtype=jnp.float64)
            thetas.append(0.0)
            vertices.append(pos)
        vertices.reverse()
        thetas = thetas[1:]
        return jnp.array(vertices), jnp.array(thetas)
