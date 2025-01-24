import numpy as np

from constraints.clamp import Clamp
from constraints.inextensibility import Inextensibility
from energies.energy import Energy
from solver.solver_params import SolverParams


class CenterlineIntegrator:

    @staticmethod
    def integrate_centerline(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams,
                             energies: list[Energy]):
        forces = np.zeros_like(pos)
        for energy in energies:
            forces -= energy.d_energy_d_pos(pos, theta, solver_params)

        # Fixed node constraint
        # pos[-1] = solver_params.pos0
        # solver_params.vel[-1] = 0.0
        # forces[-1] = 0.0

        # Use XPBD to solve for the new position of the other nodes
        M_inv = np.diag(1 / solver_params.mass)
        pred_pos = pos + solver_params.dt * solver_params.vel + 0.5 * solver_params.dt ** 2 * M_inv @ forces
        solved_pos = CenterlineIntegrator.xpbd(pred_pos, solver_params)
        solver_params.vel = (solved_pos - pos) / solver_params.dt
        return solved_pos

    @staticmethod
    def xpbd(pos, solver_params):
        lambdas = np.zeros(pos.shape[0])

        solver_iterations = 100
        for _ in range(solver_iterations):
            # Fixed node constraint
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


            # Inextensibility constraint
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
                # c_j = c.constraint(pos, solver_params)
                # grad = c.d_constraint_d_pos(pos, solver_params).ravel()
                # grad = grad.reshape(1, -1)
                # delta_lambdas[i] = (-c_j - alpha[i] * lambdas[i]) / (grad @ M_inv @ grad.T + alpha[i])
                # delta_x = M_inv @ grad.T * delta_lambdas[i]
                #
                # lambdas[i] += delta_lambdas[i]
                # pos += delta_x.ravel()
                pass

        return pos.reshape(-1, 3)
