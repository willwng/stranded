from dataclasses import dataclass

import numpy as np


@dataclass
class Helix:
    # Generalized coordinates (size 3n)
    # q = [twist_{1}, curvature_{1, 1}, curvature_{1, 2}, ..., twist_{n}, curvature_{n, 1}, curvature_{n, 2}]
    q: np.ndarray
    q0: np.ndarray
    # Number of elements (including clamped index 0)
    n_elems: int

    # Arc length. s[i] is the arc length from the start to the ith element
    s: np.ndarray

    # Total length of the helix
    L: float

    # Clamped position (size 3)
    r0: np.ndarray

    # Clamped material frame (size 3x3)
    n0: np.ndarray

    # Stiffness (size 3n)
    EI: np.ndarray
