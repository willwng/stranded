import numpy as np
from scipy.optimize import minimize

from rod.reduced import Reduced
from solver.sim import Sim


class ShapeMatch:
    @staticmethod
    def project_out_force(forces: np.ndarray, p: np.ndarray) -> np.ndarray:
        """ Projects out the force in irrelevant directions (e.g., in direction of edges) """
        # Zero out force of first node (fixed)
        forces[0] = 0.0
        # Project out the force in direction of the edges (inextensibility)
        edges = p[1:] - p[:-1]
        edges /= np.linalg.norm(edges, axis=1)[:, None]
        forces[1:] -= np.sum(forces[1:] * edges, axis=1)[:, None] * edges
        return forces

    @staticmethod
    def force_obj(pos_rest: np.ndarray, theta: np.ndarray, target_pos: np.ndarray, target_theta: np.ndarray,
                  sim: Sim) -> float:
        """ Objective function for shape matching (minimize the magnitude of forces) """
        sim.define_rest_state(pos=pos_rest, theta=theta)
        forces = sim.compute_force(pos_test=target_pos, theta_test=target_theta)
        forces = ShapeMatch.project_out_force(forces, target_pos)
        return 0.5 * np.linalg.norm(forces) ** 2

    @staticmethod
    def optimize_positions(target_pos: np.ndarray, target_theta: np.ndarray, sim: Sim,
                           callback: callable) -> np.ndarray:
        """ Finds the rest position that best matches the target position """
        # Start with target pos as initial guess
        pos_rest, theta_rest = target_pos.copy(), target_theta.copy()
        # Solve the minimization in reduced coordinates of positions
        r, polar = Reduced.to_polar_coordinates(pos_rest)

        def pos_obj(p: np.ndarray, t: np.ndarray):
            pos_rest_k = Reduced.to_cartesian_coordinates(r, p.reshape(-1, 2), target_pos[0])
            return ShapeMatch.force_obj(pos_rest=pos_rest_k, theta=t, target_pos=target_pos, target_theta=target_theta,
                                        sim=sim)

        def pos_callback(p: np.ndarray):
            print(f"Objective: {pos_obj(p, target_theta)}")
            pos_k = Reduced.to_cartesian_coordinates(r, p.reshape(-1, 2), target_pos[0])
            callback(pos_k)

        res = minimize(pos_obj, x0=polar.ravel(), method='bfgs', tol=1e-8, args=(target_theta,), options={'disp': True},
                       callback=pos_callback)
        polar_optimized = res.x
        pos_optimized = Reduced.to_cartesian_coordinates(r, polar_optimized.reshape(-1, 2), target_pos[0])
        return pos_optimized
