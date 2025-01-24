import numpy as np

from energies.energy import Energy
from solver.solver_params import SolverParams


class CenterlineIntegrator:

    @staticmethod
    def integrate_centerline(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams,
                             energies: list[Energy]):
        forces = np.zeros_like(pos)
        for energy in energies:
            forces -= energy.d_energy_d_pos(pos, theta, solver_params)

        # Zero out forces on root
        forces[-1] = 0.0

        # Finite diff check
        from scipy.optimize import check_grad
        def total_energy(p: np.ndarray):
            energy_tot = 0.0
            for e in energies:
                energy_tot += e.compute_energy(p.reshape(-1, 3), theta, solver_params)
            return energy_tot
        def total_grad_energy(p: np.ndarray):
            grad_energy = np.zeros_like(p)
            for energy in energies:
                grad_energy += energy.d_energy_d_pos(p.reshape(-1, 3), theta, solver_params).ravel()
            return grad_energy
        #
        # print("Gradient check", check_grad(total_energy, total_grad_energy, pos.ravel()))
        # # finite diff
        # eps = 1e-6
        # grad_est = np.zeros_like(forces)
        # for i in range(len(pos)):
        #     pos[i] += eps
        #     e_plus = total_energy(pos.ravel())
        #     pos[i] -= 2 * eps
        #     e_minus = total_energy(pos.ravel())
        #     pos[i] += eps
        #     grad_est[i] = (e_plus - e_minus) / (2 * eps)
        #
        # print(" Finite diff:", grad_est)
        # print(" Analytical:", total_grad_energy(pos.ravel()).reshape(-1, 3))
        # quit()
        #

        return pos + solver_params.dt * forces
