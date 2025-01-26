from dataclasses import dataclass

import numpy as np


@dataclass
class SolverParams:
    # Simulation parameters
    dt: float
    xpbd_steps: int