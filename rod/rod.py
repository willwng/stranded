from dataclasses import dataclass

import numpy as np

@dataclass
class RodParams:
    # Bending, twisting, stretching
    B: np.ndarray
    beta: float
    k: float

    # Gravity
    mass: np.ndarray
    g: float


@dataclass
class RodState:
    # Computed or updated at every step
    vel: np.ndarray
    bishop_frame: np.ndarray
    kb: np.ndarray


@dataclass
class InitialRodState:
    # Computed only once
    pos0: np.ndarray
    theta0: np.ndarray
    l_bar: np.ndarray
    l_bar_edge: np.ndarray
    omega_bar: np.ndarray
