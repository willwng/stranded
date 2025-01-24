from dataclasses import dataclass

import numpy as np


@dataclass
class SolverParams:
    # Bending, twisting, stretching
    B: np.ndarray
    beta: float
    k: float

    # Gravity
    mass: np.ndarray
    g: float

    # Rod parameters
    n: int
    pos0: np.ndarray
    vel: np.ndarray
    bishop_frame: np.ndarray
    l_bar: np.ndarray
    l_bar_edge: np.ndarray
    omega_bar: np.ndarray

    dt: float
