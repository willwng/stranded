import numpy as np

from energies.energy import Energy
from solver.solver_params import SolverParams


class CenterlineIntegrator:

    @staticmethod
    def integrate_centerline(pos: np.ndarray, theta: np.ndarray, solver_params: SolverParams,
                             energies: list[Energy]):
        grad = np.zeros_like(pos)
        for energy in energies:
            energy.d_energy_d_pos(grad, pos, theta, solver_params)

        # Predicted position, with no constraints
        forces = -grad
        M_inv = np.diag(1 / solver_params.mass)
        pred_pos = pos + solver_params.dt * solver_params.vel + 0.5 * solver_params.dt ** 2 * M_inv @ forces

        # Use XPBD to solve for the new position considering the constraints
        solved_pos = CenterlineIntegrator.xpbd(pred_pos, solver_params)

        # Disabled velocity (somehow this makes the simulation less stable)
        # solver_params.vel = (solved_pos - pos) / solver_params.dt
        return solved_pos

    @staticmethod
    def xpbd(pos, solver_params):
        """ Extended Position Based Dynamics (XPBD) for solving constraints """
        # Prepare for constraint solve: inverse mass and Lagrange multipliers
        i = np.arange(solver_params.n + 1)
        inv_mass = 1 / solver_params.mass[i]
        inv_mass2 = 1 / solver_params.mass[i + 1]
        sum_mass = inv_mass + inv_mass2
        lambdas = np.zeros(solver_params.n + 1, dtype=np.float64)
        compliance = 1e-12 / (solver_params.dt ** 2)

        for _ in range(solver_params.xpbd_steps):
            # Fixed node constraint (just clamp the top node)
            pos[0] = solver_params.pos0[0]

            # Inextensibility constraint for each edge
            # Constraint is difference between length and rest length
            p1_minus_p2 = pos[i] - pos[i + 1]
            distance = np.linalg.norm(p1_minus_p2, axis=1)
            constraint = distance - solver_params.l_bar_edge[i]
            # Update lambda and position
            d_lambda = (-constraint - compliance * lambdas[i]) / (sum_mass + compliance)
            correction_vector = d_lambda[:, None] * p1_minus_p2 / (distance[:, None] + 1e-8)
            lambdas[i] += d_lambda
            pos[i] += inv_mass[:, None] * correction_vector
            pos[i + 1] -= inv_mass2[:, None] * correction_vector

        return pos.reshape(-1, 3)
