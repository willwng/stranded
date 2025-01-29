from dataclasses import dataclass

import numpy as np


@dataclass
class SolverParams:
    # Simulation parameters
    damping: float
    dt: float
    xpbd_steps: int