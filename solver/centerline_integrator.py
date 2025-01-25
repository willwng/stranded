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

        def total_energy(p):
            p = p.reshape(-1, 3)
            energy = 0
            for e in energies:
                energy += e.compute_energy(p, theta, solver_params)
            return energy

        def total_grad_energy(p):
            p = p.reshape(-1, 3)
            grad_energy = np.zeros_like(p)
            for e in energies:
                grad_energy += e.d_energy_d_pos(p, theta, solver_params)
            return grad_energy.flatten()
        #
        # print("analytical gradient: ", total_grad_energy(pos.flatten()))
        # # Compute finite diff
        # h = 1e-6
        # grad_fd = np.zeros_like(pos)
        # for i in range(pos.shape[0]):
        #     for j in range(pos.shape[1]):
        #         pos[i, j] += h
        #         energy_plus = total_energy(pos)
        #         pos[i, j] -= 2 * h
        #         energy_minus = total_energy(pos)
        #         pos[i, j] += h
        #         grad_fd[i, j] = (energy_plus - energy_minus) / (2 * h)
        # print("finite difference gradient: ", grad_fd.ravel())
        # quit()

        # Predicted position, with no constraints
        M_inv = np.diag(1 / solver_params.mass)
        pred_pos = pos + solver_params.dt * solver_params.vel + 0.5 * solver_params.dt ** 2 * M_inv @ forces

        # Use XPBD to solve for the new position considering the constraints
        solved_pos = CenterlineIntegrator.xpbd(pred_pos, solver_params)
        solver_params.vel = (solved_pos - pos) / solver_params.dt
        return solved_pos

    @staticmethod
    def xpbd(pos, solver_params):
        lambdas = np.zeros(pos.shape[0])

        solver_iterations = 10
        for _ in range(solver_iterations):
            # Fixed node constraint (just clamp the top node)
            inv_mass = 1 / solver_params.mass[-1]
            sum_mass = inv_mass
            p1, p2 = pos[-1], solver_params.pos0
            p1_minus_p2 = p1 - p2
            distance = np.linalg.norm(p1_minus_p2)
            constraint = distance
            if constraint > 0:
                compliance = 1e-12 / (solver_params.dt ** 2)
                d_lambda = (-constraint - compliance * lambdas[-1]) / (sum_mass + compliance)
                correction_vector = d_lambda * p1_minus_p2 / (distance + 1e-8)
                lambdas[-1] += d_lambda
                pos[-1] += inv_mass * correction_vector

            # Inextensibility constraint for each edge
            for i in range(solver_params.n + 1):
                inv_mass_i1 = 1 / solver_params.mass[i]
                inv_mass_i2 = 1 / solver_params.mass[i + 1]
                sum_mass = inv_mass_i1 + inv_mass_i2
                if sum_mass == 0:
                    continue
                p1_minus_p2 = pos[i] - pos[i + 1]
                distance = np.linalg.norm(p1_minus_p2)
                constraint = distance - solver_params.l_bar_edge[i]
                compliance = 1e-12 / (solver_params.dt ** 2)
                d_lambda = (-constraint - compliance * lambdas[i]) / (sum_mass + compliance)
                correction_vector = d_lambda * p1_minus_p2 / (distance + 1e-8)
                lambdas[i] += d_lambda

                pos[i] += inv_mass_i1 * correction_vector
                pos[i + 1] -= inv_mass_i2 * correction_vector
                pass

        return pos.reshape(-1, 3)
