from dataclasses import dataclass
import cupy as cp


@dataclass
class Helix:
    # Generalized coordinates (size 3n)
    q: cp.ndarray
    q0: cp.ndarray

    # Number of sites (including clamped index 0)
    n_sites: int

    # Arc length. s[i] is the arc length from the start to the ith element
    s: cp.ndarray

    # Total length of the helix
    L: float

    # Clamped position (size 3)
    r0: cp.ndarray

    # Clamped material frame (size 3x3)
    n0: cp.ndarray

    # Stiffness (size 3n)
    EI: cp.ndarray
