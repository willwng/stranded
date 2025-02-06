import numpy as np
from scipy.optimize import minimize

from rod.reduced import Reduced
from rod.rod_generator import RodGenerator
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

    @staticmethod
    def optimize_stiffness(target_pos: np.ndarray, target_theta: np.ndarray, pos_rest: np.ndarray,
                           theta_rest: np.ndarray, sim: Sim, callback: callable) -> np.ndarray:
        init_rod_params = sim.rod_params
        beta, k, g, mass = init_rod_params.beta, init_rod_params.k, init_rod_params.g, init_rod_params.mass

        # Update the rod parameters based on the input
        def update_rod_params(b_vals: np.ndarray):
            B = np.zeros((b_vals.size // 2, 2, 2))
            # To ensure positivity of the stiffness values, we use the square of the input values
            b_vals = np.square(b_vals)
            B[:, 0, 0] = b_vals[0::2]
            B[:, 1, 1] = b_vals[1::2]
            sim.define_rod_params(B=B, beta=beta, k=k, g=g, mass=mass)
            return

        # Solve the minimization wrst bending stiffness
        def stiffness_obj(b_vals: np.ndarray):
            update_rod_params(b_vals)
            return ShapeMatch.force_obj(pos_rest=pos_rest, theta=theta_rest, target_pos=target_pos,
                                        target_theta=target_theta, sim=sim)

        def stiffness_callback(b_vals: np.ndarray):
            print(f"Objective: {stiffness_obj(b_vals)}")
            update_rod_params(b_vals)
            callback(pos_rest)

        # Get diagonals of B matrix and flatten
        init_b_vals = init_rod_params.B[:, 0, 0].tolist() + init_rod_params.B[:, 1, 1].tolist()
        init_b_vals = np.array(init_b_vals)
        # add constraint that the stiffness values are positive
        res = minimize(stiffness_obj, x0=init_b_vals, method='bfgs', tol=1e-8, options={'disp': True},
                       callback=stiffness_callback)
        print(res.x)
        b_vals = np.square(res.x)
        B_optimized = np.zeros_like(init_rod_params.B)
        B_optimized[:, 0, 0] = b_vals[0::2]
        B_optimized[:, 1, 1] = b_vals[1::2]
        return B_optimized

    @staticmethod
    def optimize_curvature(target_pos: np.ndarray, target_theta: np.ndarray, sim: Sim,
                           callback: callable) -> np.ndarray:
        total_length = np.sum(np.linalg.norm(target_pos[1:] - target_pos[:-1], axis=1))

        # Solve the minimization wrst bending stiffness
        def stiffness_obj(x: np.ndarray):
            n_curls, curl_radius, offset = x[-3], x[-2], x[-1]
            print(f"n_curls: {n_curls}, curl_radius: {curl_radius}, offset: {offset}")
            pos_rest, theta_rest = RodGenerator.evenly_spaced_helix(
                num_points=len(target_pos), n_curls=n_curls, curl_radius=curl_radius, total_length=total_length,
                offset=offset)
            correction = target_pos[0] - pos_rest[0]
            pos_rest += correction
            callback(pos_rest)
            return ShapeMatch.force_obj(pos_rest=pos_rest, theta=theta_rest, target_pos=target_pos,
                                        target_theta=target_theta, sim=sim)

        init_x = np.array([0.01 * total_length, 0.9, 0.0])
        best_x = None
        best_obj = np.inf
        for i in range(10):
            for j in range(10):
                for k in range(10):
                    # offset from 0 to 2pi
                    x = init_x + np.array([i * 0.05 * total_length, j * 0.05, 2 * np.pi * k / 10])
                    try:
                        obj = stiffness_obj(x)
                        print(f"Objective: {obj}")
                        if obj < best_obj:
                            best_obj = obj
                            best_x = x
                    except ValueError:
                        n_curls, curl_radius, offset = best_x
                        pos_rest, theta_rest = RodGenerator.evenly_spaced_helix(
                            num_points=len(target_pos), n_curls=n_curls, curl_radius=curl_radius,
                            total_length=total_length,
                            offset=offset)
                        return pos_rest

        n_curls, curl_radius, offset = best_x
        pos_rest, theta_rest = RodGenerator.evenly_spaced_helix(
            num_points=len(target_pos), n_curls=n_curls, curl_radius=curl_radius, total_length=total_length,
            offset=offset)
        return pos_rest
