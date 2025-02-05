import csv
import os
from dataclasses import dataclass

import numpy as np


@dataclass
class SolverParams:
    # Simulation parameters
    damping: float
    dt: float
    xpbd_steps: int


@dataclass
class SolverAnalytics:
    # Time taken for each step
    time_taken: float
    quasistatic_time: float
    integration_time: float
    xpbd_time: float

    # Energy values
    total_energy: float
    kinetic_energy: float
    potential_energy: float
    force: np.ndarray
    mag_force: float


def init_solver_analytics():
    return SolverAnalytics(
        time_taken=0.0,
        quasistatic_time=0.0,
        integration_time=0.0,
        xpbd_time=0.0,
        total_energy=0.0,
        kinetic_energy=0.0,
        potential_energy=0.0,
        force=np.zeros(0),
        mag_force=0.0,
    )


def pretty_print_analytics(analytics: SolverAnalytics) -> str:
    return f"Time taken: {analytics.time_taken:.3f}s\n" \
           f"Quasistatic time: {analytics.quasistatic_time:.3f}s\n" \
           f"Integration time: {analytics.integration_time:.3f}s\n" \
           f"XPBD time: {analytics.xpbd_time:.3f}s\n" \
           f"Total energy: {analytics.total_energy:.3f}\n" \
           f"Kinetic energy: {analytics.kinetic_energy:.3f}\n" \
           f"Potential energy: {analytics.potential_energy:.3f}\n" \
           f"Force Magnitude: {analytics.mag_force:.3f}\n"


def update_csv_analytics(analytics: SolverAnalytics, csv_file: str):
    # If the file doesn't exist, write the header
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["TimeTaken", "QuasistaticTime", "IntegrationTime", "XPBDTime", "TotalEnergy",
                             "KineticEnergy", "PotentialEnergy", "MagForce"])
    # Append the new row
    with open(csv_file, "a+") as f:
        writer = csv.writer(f)
        row = [analytics.time_taken, analytics.quasistatic_time, analytics.integration_time, analytics.xpbd_time,
               analytics.total_energy, analytics.kinetic_energy, analytics.potential_energy, analytics.mag_force]
        writer.writerow(row)
    return
