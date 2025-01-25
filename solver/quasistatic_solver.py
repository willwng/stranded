import numpy as np

from energies.energy import Energy
from solver.solver_params import SolverParams

import scipy.optimize as opt


class QuasistaticSolver:

    @staticmethod
    def quasistatic_update(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams,
                           energies: list[Energy]) -> np.ndarray:
        # Clamp/fix theta for the first edge
        theta_clamped = theta[0]
        def total_energy(t: np.ndarray):
            energy_tot = 0.0
            full_theta = np.concatenate(([theta_clamped], t))
            for energy in energies:
                energy_tot += energy.compute_energy(pos, full_theta, solver_params)
            return energy_tot

        def total_grad_energy(t: np.ndarray):
            grad_energy = np.zeros_like(t)
            full_theta = np.concatenate(([theta_clamped], t))
            for energy in energies:
                grad_energy += energy.d_energy_d_theta(pos, full_theta, solver_params)[1:]
            return grad_energy

        # Minimize total energy wrst the free theta values
        theta_free = theta[1:]
        res = opt.minimize(total_energy, theta_free, jac=total_grad_energy, method='L-BFGS-B')
        theta_relaxed = np.concatenate(([theta_clamped], res.x))
        return theta_relaxed
