import numpy as np

from energies.energy import Energy
from rod.rod import Rod
from solver.centerline_integrator import CenterlineIntegrator
from solver.quasistatic_solver import QuasistaticSolver
from solver.solver_params import SolverParams


class Sim:
    @staticmethod
    def step(rod: Rod, B: np.ndarray, beta: float, k: float, g: float, p_top: np.ndarray, mass: np.ndarray, energies: list[Energy], dt):
        # Set up the solver
        solver_params = SolverParams(
            B=B,
            beta=beta,
            k=k,
            mass=mass,
            g=g,
            n=rod.n,
            pos0=p_top,
            vel=rod.vel,
            bishop_frame=rod.bishop_frame,
            l_bar=rod.l_bar,
            l_bar_edge=rod.l_bar_edge,
            omega_bar=rod.omega_bar,
            dt=dt
        )
        pos, theta = rod.pos, rod.theta
        pos = CenterlineIntegrator.integrate_centerline(pos, theta, solver_params, energies)
        rod.update_bishop_frame()
        theta = QuasistaticSolver.quasistatic_update(pos, theta, solver_params, energies)
        # Copy back
        rod.pos, rod.theta = pos, theta
        return

    @staticmethod
    def init(rod: Rod, B: np.ndarray, beta: float, k: float, g: float, p_top, mass: np.ndarray, energies: list[Energy], dt):
        # Set up the solver
        solver_params = SolverParams(
            B=B,
            beta=beta,
            k=k,
            mass=mass,
            g=g,
            n=rod.n,
            pos0=p_top,
            vel=rod.vel,
            bishop_frame=rod.bishop_frame,
            l_bar=rod.l_bar,
            l_bar_edge=rod.l_bar_edge,
            omega_bar=rod.omega_bar,
            dt=dt
        )
        pos, theta = rod.pos, rod.theta
        rod.theta = QuasistaticSolver.quasistatic_update(pos, theta, solver_params, energies)
        return
