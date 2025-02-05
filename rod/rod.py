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


def empty_rod_state(n_vertices: int, frozen_pos_indices: np.ndarray, frozen_theta_indices: np.ndarray) -> RodState:
    n_edges = n_vertices - 1
    return RodState(
        vel=np.zeros((n_vertices, 3)),
        bishop_frame=np.zeros((n_edges, 2, 3)),
        kb=np.zeros((n_edges, 3)),
        kb_den=np.zeros(n_edges),
        nabla_kb=np.zeros((n_edges, 3, 3, 3)),
        nabla_psi=np.zeros((n_edges, 3, 3)),
        material_frame=np.zeros((n_edges, 2, 3)),
        frozen_pos_indices=frozen_pos_indices,
        frozen_theta_indices=frozen_theta_indices,
    )


def copy_rod_state(state: RodState) -> RodState:
    return RodState(
        vel=state.vel.copy(),
        bishop_frame=state.bishop_frame.copy(),
        kb=state.kb.copy(),
        kb_den=state.kb_den.copy(),
        nabla_kb=state.nabla_kb.copy(),
        nabla_psi=state.nabla_psi.copy(),
        material_frame=state.material_frame.copy(),
        frozen_pos_indices=state.frozen_pos_indices.copy(),
        frozen_theta_indices=state.frozen_theta_indices.copy(),
    )


@dataclass
class InitialRodState:
    # Computed only once
    pos0: np.ndarray
    theta0: np.ndarray
    l_bar: np.ndarray
    l_bar_edge: np.ndarray
    omega_bar: np.ndarray
