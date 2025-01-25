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

        # Predicted position, with no constraints
        M_inv = np.diag(1 / solver_params.mass)
        pred_pos = pos + solver_params.dt * solver_params.vel + 0.5 * solver_params.dt ** 2 * M_inv @ forces

        # Use XPBD to solve for the new position considering the constraints
        solved_pos = CenterlineIntegrator.xpbd(pred_pos, solver_params)

        # Disabled velocity (somehow it makes the simulation less stable)
        # solver_params.vel = (solved_pos - pos) / solver_params.dt
        return solved_pos

    @staticmethod
    def xpbd(pos, solver_params):
        """ Extended Position Based Dynamics (XPBD) for solving constraints """
        lambdas = np.zeros(pos.shape[0])

        for _ in range(solver_params.xpbd_steps):
            # Fixed node constraint (just clamp the top node)
            inv_mass = 1 / solver_params.mass[0]
            sum_mass = inv_mass
            p1, p2 = pos[0], solver_params.pos0[0]
            p1_minus_p2 = p1 - p2
            distance = np.linalg.norm(p1_minus_p2)
            constraint = distance
            compliance = 1e-12 / (solver_params.dt ** 2)
            d_lambda = (-constraint - compliance * lambdas[0]) / (sum_mass + compliance)
            correction_vector = d_lambda * p1_minus_p2 / (distance + 1e-8)
            lambdas[0] += d_lambda
            pos[0] += inv_mass * correction_vector

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
