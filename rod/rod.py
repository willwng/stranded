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
    vel: np.ndarray
    # These can be recovered from pos and theta, but useful to precompute
    # Updated after every centerline update
    bishop_frame: np.ndarray
    kb: np.ndarray
    kb_den: np.ndarray  # denominator of kb (to compute nabla (kb)_i)
    nabla_kb: np.ndarray
    nabla_psi: np.ndarray
    # Updated after every quasistatic (theta) update
    material_frame: np.ndarray

    # For shape-matching, freezing indices of these nodes/edges
    frozen_pos_indices: np.ndarray
    frozen_theta_indices: np.ndarray


@dataclass
class InitialRodState:
    # Computed only once
    pos0: np.ndarray
    theta0: np.ndarray
    l_bar: np.ndarray
    l_bar_edge: np.ndarray
    omega_bar: np.ndarray
