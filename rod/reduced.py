import numpy as np


class Reduced:
    @staticmethod
    def to_polar_coordinates(pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Converts 3D positions to polar coordinates """
        e = pos[1:] - pos[:-1]
        r = np.linalg.norm(e, axis=1)
        theta = np.arccos(e[:, 2] / r)
        phi = np.arctan2(e[:, 1], e[:, 0])
        return r, np.stack([theta, phi], axis=1)

    @staticmethod
    def to_cartesian_coordinates(r: np.ndarray, polar_coords: np.ndarray, pos0: np.ndarray) -> np.ndarray:
        """ Converts the polar coordinates to 3D positions. Starting with pos0,
        the first edge is in the direction of the first polar coordinate """
        theta, phi = polar_coords[:, 0], polar_coords[:, 1]
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        e = np.stack([x, y, z], axis=1)
        return np.concatenate([pos0[None], np.cumsum(e, axis=0) + pos0])

    @staticmethod
    def to_reduced_coordinates(pos: np.ndarray) -> np.ndarray:
        """ Converts 3D positions to reduced coordinates (assuming inextensible rod) """
        polar_coords = Reduced.to_polar_coordinates(pos)
        _, theta, phi = polar_coords[:, 0], polar_coords[:, 1], polar_coords[:, 2]
        return np.stack([theta, phi], axis=1)
