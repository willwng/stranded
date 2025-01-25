import numpy as np

from energies.energy import Energy
from rod.rod import Rod
from solver.centerline_integrator import CenterlineIntegrator
from solver.quasistatic_solver import QuasistaticSolver
from solver.solver_params import SolverParams


class Sim:

    def __init__(self, rod: Rod, B: np.ndarray, beta: float, k: float, g: float, mass: np.ndarray,
                 energies: list[Energy], dt, xpbd_steps):
        self.solver_params = SolverParams(
            B=B,
            beta=beta,
            k=k,
            mass=mass,
            g=g,
            n=rod.n,
            pos0=rod.pos0,
            vel=rod.vel,
            bishop_frame=rod.bishop_frame,
            l_bar=rod.l_bar,
            l_bar_edge=rod.l_bar_edge,
            omega_bar=rod.omega_bar,
            dt=dt,
            xpbd_steps=xpbd_steps
        )
        self.energies = energies
        self.rod = rod

        self.init()
        return

    def step(self):
        rod = self.rod
        pos, theta = rod.pos, rod.theta
        pos = CenterlineIntegrator.integrate_centerline(pos, theta, self.solver_params, self.energies)
        rod.update_bishop_frame()
        theta = QuasistaticSolver.quasistatic_update(pos, theta, self.solver_params, self.energies)
        # Copy back
        rod.pos, rod.theta = pos, theta
        return

    def init(self):
        """ Initialization steps of simulation """
        rod = self.rod
        pos, theta = rod.pos, rod.theta
        rod.theta = QuasistaticSolver.quasistatic_update(pos, theta, self.solver_params, self.energies)
        return
