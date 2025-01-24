import numpy as np

from energies.energy import Energy
from solver.solver_params import SolverParams

import scipy.optimize as opt


class QuasistaticSolver:

    @staticmethod
    def quasistatic_update(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams,
                           energies: list[Energy]) -> np.ndarray:
        theta0 = theta.copy()

        def total_energy(t: np.ndarray):
            energy_tot = 0.0
            for energy in energies:
                energy_tot += energy.compute_energy(pos, t, solver_params)
            return energy_tot

        def total_grad_energy(t: np.ndarray):
            grad_energy = np.zeros_like(t)
            for energy in energies:
                grad_energy += energy.d_energy_d_theta(pos, t, solver_params)
            # TODO: Clamped edge
            grad_energy[-1] = 0.0
            return grad_energy

        # Minimize total energy wrst theta
        # TODO: maybe don't minimize the clamped theta?
        res = opt.minimize(total_energy, theta0, jac=total_grad_energy, method='L-BFGS-B')

        # Perform finite difference check
        # from scipy.optimize import check_grad
        # print("Quasistatic gradient check", check_grad(total_energy, total_grad_energy, theta0))
        return res.x
