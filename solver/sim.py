import jax.numpy as jnp
import scipy.optimize as opt
import time

from energies.energy import Energy
from rod.rod import RodState, InitialRodState, RodParams
from rod.rod_util import RodUtil
from solver.solver_params import SolverParams, init_solver_analytics


class Sim:
    """ Handles all the simulation logic """

    def __init__(self, pos: jnp.ndarray, theta: jnp.ndarray, B: jnp.ndarray, beta: float, k: float, g: float,
                 mass: jnp.ndarray, energies: list[Energy], damping: float, dt: float, xpbd_steps: int):
        # Pre-compute the initial state
        edge_lengths = RodUtil.compute_edge_lengths(pos=pos)
        node_lengths = RodUtil.compute_node_lengths(edge_lengths=edge_lengths)
        kb, kb_den = RodUtil.compute_curvature_binormal(pos=pos, rest_edge_lengths=edge_lengths)
        bishop_frame = jnp.zeros((theta.shape[0], 2, 3))
        bishop_frame = RodUtil.update_bishop_frames(pos=pos, bishop_frame=bishop_frame)
        material_frame = RodUtil.compute_material_frames(theta=theta, bishop_frame=bishop_frame)
        omega = RodUtil.compute_omega(theta=theta, kb=kb, bishop_frame=bishop_frame)
        # Derivatives
        nabla_kb = RodUtil.compute_nabla_kb(pos=pos, kb=kb, kb_den=kb_den)
        nabla_psi = RodUtil.compute_nabla_psi(kb=kb, rest_edge_lengths=edge_lengths)
        self.init_state = InitialRodState(
            pos0=pos.copy(),
            theta0=theta.copy(),
            l_bar=node_lengths.copy(),
            l_bar_edge=edge_lengths.copy(),
            omega_bar=omega.copy(),
        )
        self.state = RodState(
            vel=jnp.zeros_like(pos),
            bishop_frame=bishop_frame,
            kb=kb,
            kb_den=kb_den,
            nabla_kb=nabla_kb,
            nabla_psi=nabla_psi,
            material_frame=material_frame,
        )
        self.energies = energies
        self.rod_params = RodParams(B=B, beta=beta, k=k, mass=mass, g=g)
        self.solver_params = SolverParams(damping=damping, dt=dt, xpbd_steps=xpbd_steps)

        # Analytics
        self.analytics = init_solver_analytics()
        return

    def step(self, pos: jnp.ndarray, theta: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        """ Perform a simulation step """
        start_step_time = time.time()

        # Integrate centerline, update anything that depends on position
        pos = self.integrate_centerline(pos=pos, theta=theta)
        self.update_curvature_binormal(pos=pos)
        self.update_bishop_frames(pos=pos)

        # Quasistatic update, updating anything that depends on theta
        start_quasistatic_time = time.time()
        # theta = self.quasistatic_update(pos=pos, theta=theta)
        self.update_material_frames(theta=theta)

        # Update analytics
        end_time = time.time()
        self.analytics.integration_time = start_quasistatic_time - start_step_time
        self.analytics.quasistatic_time = end_time - start_quasistatic_time
        self.analytics.time_taken = end_time - start_step_time
        self.update_analytics(pos=pos, theta=theta)
        return pos, theta

    def update_analytics(self, pos: jnp.ndarray, theta: jnp.ndarray):
        """ Update the analytics """
        kinetic_energy = 0.5 * jnp.sum(self.rod_params.mass * jnp.linalg.norm(self.state.vel, axis=1) ** 2)
        potential_energy = sum([e.compute_energy(pos=pos, theta=theta, rod_state=self.state,
                                                 init_rod_state=self.init_state, rod_params=self.rod_params)
                                for e in self.energies])
        grad = jnp.zeros_like(pos)
        for energy in self.energies:
            grad = energy.d_energy_d_pos(grad=grad, pos=pos, theta=theta, rod_state=self.state,
                                         init_rod_state=self.init_state,
                                         rod_params=self.rod_params)
        self.analytics.kinetic_energy = kinetic_energy
        self.analytics.potential_energy = potential_energy
        self.analytics.total_energy = kinetic_energy + potential_energy
        self.analytics.mag_force = jnp.linalg.norm(grad)
        return

    def update_bishop_frames(self, pos: jnp.ndarray):
        """ Update the bishop frames of the rod """
        self.state.bishop_frame = RodUtil.update_bishop_frames(pos, self.state.bishop_frame)
        return

    def update_material_frames(self, theta: jnp.ndarray):
        """ Update the material frames of the rod """
        self.state.material_frame = RodUtil.compute_material_frames(theta=theta, bishop_frame=self.state.bishop_frame)
        return

    def update_curvature_binormal(self, pos: jnp.ndarray):
        """ Update the curvature binormal of the rod and any dependent values """
        self.state.kb, self.state.kb_den = RodUtil.compute_curvature_binormal(
            pos=pos, rest_edge_lengths=self.init_state.l_bar_edge)
        self.state.nabla_kb = RodUtil.compute_nabla_kb(pos=pos, kb=self.state.kb, kb_den=self.state.kb_den)
        self.state.nabla_psi = RodUtil.compute_nabla_psi(kb=self.state.kb, rest_edge_lengths=self.init_state.l_bar_edge)
        return

    def quasistatic_update(self, pos: jnp.ndarray, theta: jnp.ndarray):
        """ Minimizes the energy with respect to theta (except theta of the first edge) """
        # Clamp/fix theta for the first edge
        theta_clamped = theta[0]

        def total_energy(t: jnp.ndarray):
            energy_tot = 0.0
            full_theta = jnp.concatenate(([theta_clamped], t))
            for energy in self.energies:
                energy_tot += energy.compute_energy(pos=pos, theta=full_theta, rod_state=self.state,
                                                    init_rod_state=self.init_state, rod_params=self.rod_params)
            return energy_tot

        def total_grad_energy(t: jnp.ndarray):
            grad = jnp.zeros(t.size + 1)
            full_theta = jnp.concatenate(([theta_clamped], t))
            for energy in self.energies:
                energy.d_energy_d_theta(grad=grad, pos=pos, theta=full_theta, rod_state=self.state,
                                        init_rod_state=self.init_state, rod_params=self.rod_params)
            return grad[1:]

        # Minimize total energy wrst the free theta values
        theta_free = theta[1:]
        res = opt.minimize(total_energy, theta_free, jac=total_grad_energy, method='L-BFGS-B')
        relaxed_theta = jnp.concatenate(([theta_clamped], res.x))
        return relaxed_theta

    def integrate_centerline(self, pos: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
        """ Update the position of the rod according to the force with constraint solve """
        solver_params = self.solver_params
        state = self.state

        # Compute forces
        grad = jnp.zeros_like(pos)
        for energy in self.energies:
            grad = energy.d_energy_d_pos(grad=grad, pos=pos, theta=theta, rod_state=state,
                                         init_rod_state=self.init_state,
                                         rod_params=self.rod_params)

        # Predicted position, with no constraints
        forces = -grad
        forces -= solver_params.damping * state.vel  # Damping

        M_inv = 1 / self.rod_params.mass
        pred_pos = pos + solver_params.dt * state.vel + 0.5 * solver_params.dt ** 2 * M_inv @ forces

        # Use XPBD to solve for the new position considering the constraints
        start_xpbd_time = time.time()
        solved_pos = self.xpbd(pred_pos)
        self.analytics.xpbd_time = time.time() - start_xpbd_time

        # Disabled velocity (seems unstable)
        state.vel = (solved_pos - pos) / solver_params.dt
        return solved_pos

    def xpbd(self, pred_pos):
        """ Extended Position Based Dynamics (XPBD) for solving constraints """
        solver_params = self.solver_params
        n_edges = pred_pos.shape[0] - 1

        # Prepare for constraint solve: inverse mass and Lagrange multipliers
        i = jnp.arange(n_edges)
        inv_mass = 1 / self.rod_params.mass[i]
        inv_mass2 = 1 / self.rod_params.mass[i + 1]
        inv_mass = inv_mass.at[0].set(0.0)  # Fixed node
        sum_mass = inv_mass + inv_mass2
        lambdas = jnp.zeros(n_edges, dtype=jnp.float64)
        compliance = 1e-12 / (solver_params.dt ** 2)

        for _ in range(solver_params.xpbd_steps):
            # Inextensibility constraint for each edge
            # Constraint is difference between length and rest length
            p1_minus_p2 = pred_pos[i] - pred_pos[i + 1]
            distance = jnp.linalg.norm(p1_minus_p2, axis=1)
            constraint = distance - self.init_state.l_bar_edge[i]
            # Update lambda and position
            d_lambda = (-constraint - compliance * lambdas) / (sum_mass + compliance)
            correction_vector = d_lambda[:, None] * p1_minus_p2 / (distance[:, None] + 1e-8)
            lambdas = lambdas.at[i].add(d_lambda)
            pred_pos = pred_pos.at[i].add(inv_mass[:, None] * correction_vector)
            pred_pos = pred_pos.at[i + 1].add(-inv_mass2[:, None] * correction_vector)

            # Fixed node constraint (just clamp the top node)
            pred_pos = pred_pos.at[0].set(self.init_state.pos0[0])

        return pred_pos.reshape(-1, 3)
