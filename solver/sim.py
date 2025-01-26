import numpy as np
import scipy.optimize as opt
import scipy.sparse as sp

from energies.energy import Energy
from rod.rod import RodState, InitialRodState, RodParams
from rod.rod_util import RodUtil
from solver.solver_params import SolverParams


class Sim:
    """ Handles all the simulation logic """

    def __init__(self, pos: np.ndarray, theta: np.ndarray, B: np.ndarray, beta: float, k: float, g: float,
                 mass: np.ndarray, energies: list[Energy], dt, xpbd_steps):
        self.energies = energies
        self.rod_params = RodParams(B=B, beta=beta, k=k, mass=mass, g=g)
        self.solver_params = SolverParams(dt=dt, xpbd_steps=xpbd_steps)
        # Pre-compute the initial state
        edge_lengths = RodUtil.compute_edge_lengths(pos=pos)
        node_lengths = RodUtil.compute_node_lengths(edge_lengths=edge_lengths)
        kb, kb_den = RodUtil.compute_curvature_binormal(pos=pos, rest_edge_lengths=edge_lengths)
        bishop_frame = np.zeros((theta.shape[0], 2, 3))
        bishop_frame = RodUtil.update_bishop_frames(pos=pos, theta=theta, bishop_frame=bishop_frame)
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
            vel=np.zeros_like(pos),
            bishop_frame=bishop_frame,
            kb=kb,
            kb_den=kb_den,
            nabla_kb=nabla_kb,
            nabla_psi=nabla_psi,
            material_frame=material_frame,
        )
        self.init(pos=pos, theta=theta)
        return

    def init(self, pos: np.ndarray, theta: np.ndarray):
        """
        Initialization steps of simulation: Update the bishop frames and perform a quasistatic update
        """
        self.update_bishop_frames(pos=pos, theta=theta)
        self.quasistatic_update(pos=pos, theta=theta)
        self.update_material_frames(theta=theta)
        self.update_curvature_binormal(pos=pos)
        return

    def step(self, pos: np.ndarray, theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ Perform a simulation step """
        # Pre-compute curvature binormal and material curvature
        pos = self.integrate_centerline(pos=pos, theta=theta)
        self.update_curvature_binormal(pos=pos)
        self.update_bishop_frames(pos=pos, theta=theta)

        theta = self.quasistatic_update(pos=pos, theta=theta)
        self.update_material_frames(theta=theta)
        return pos, theta

    def update_bishop_frames(self, pos: np.ndarray, theta: np.ndarray):
        """ Update the bishop frames of the rod """
        self.state.bishop_frame = RodUtil.update_bishop_frames(pos, theta, self.state.bishop_frame)
        return

    def update_material_frames(self, theta: np.ndarray):
        """ Update the material frames of the rod """
        self.state.material_frame = RodUtil.compute_material_frames(theta=theta, bishop_frame=self.state.bishop_frame)
        return

    def update_curvature_binormal(self, pos: np.ndarray):
        """ Update the curvature binormal of the rod and any dependent values """
        self.state.kb, self.state.kb_den = RodUtil.compute_curvature_binormal(
            pos=pos, rest_edge_lengths=self.init_state.l_bar_edge)
        self.state.nabla_kb = RodUtil.compute_nabla_kb(pos=pos, kb=self.state.kb, kb_den=self.state.kb_den)
        self.state.nabla_psi = RodUtil.compute_nabla_psi(kb=self.state.kb, rest_edge_lengths=self.init_state.l_bar_edge)
        return

    def quasistatic_update(self, pos: np.ndarray, theta: np.ndarray):
        """ Minimizes the energy with respect to theta (except theta of the first edge) """
        # Clamp/fix theta for the first edge
        theta_clamped = theta[0]

        def total_energy(t: np.ndarray):
            energy_tot = 0.0
            full_theta = np.concatenate(([theta_clamped], t))
            for energy in self.energies:
                energy_tot += energy.compute_energy(pos=pos, theta=full_theta, rod_state=self.state,
                                                    init_rod_state=self.init_state, rod_params=self.rod_params)
            return energy_tot

        def total_grad_energy(t: np.ndarray):
            grad = np.zeros(t.size + 1)
            full_theta = np.concatenate(([theta_clamped], t))
            for energy in self.energies:
                energy.d_energy_d_theta(grad=grad, pos=pos, theta=full_theta, rod_state=self.state,
                                        init_rod_state=self.init_state, rod_params=self.rod_params)
            return grad[1:]

        # Minimize total energy wrst the free theta values
        theta_free = theta[1:]
        res = opt.minimize(total_energy, theta_free, jac=total_grad_energy, method='L-BFGS-B')
        relaxed_theta = np.concatenate(([theta_clamped], res.x))
        return relaxed_theta

    def integrate_centerline(self, pos: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """ Update the position of the rod according to the force with constraint solve """
        solver_params = self.solver_params
        state = self.state

        # Compute forces
        grad = np.zeros_like(pos)
        for energy in self.energies:
            energy.d_energy_d_pos(grad=grad, pos=pos, theta=theta, rod_state=state, init_rod_state=self.init_state,
                                  rod_params=self.rod_params)

        # Predicted position, with no constraints
        forces = -grad
        M_inv = sp.diags(1 / self.rod_params.mass)
        pred_pos = pos + solver_params.dt * state.vel + 0.5 * solver_params.dt ** 2 * M_inv @ forces

        # Use XPBD to solve for the new position considering the constraints
        solved_pos = self.xpbd(pred_pos)
        return solved_pos

    def xpbd(self, pred_pos):
        """ Extended Position Based Dynamics (XPBD) for solving constraints """
        solver_params = self.solver_params
        n_edges = pred_pos.shape[0] - 1

        # Prepare for constraint solve: inverse mass and Lagrange multipliers
        i = np.arange(n_edges)
        inv_mass = 1 / self.rod_params.mass[i]
        inv_mass2 = 1 / self.rod_params.mass[i + 1]
        sum_mass = inv_mass + inv_mass2
        lambdas = np.zeros(n_edges, dtype=np.float64)
        compliance = 1e-12 / (solver_params.dt ** 2)

        for _ in range(solver_params.xpbd_steps):
            # Fixed node constraint (just clamp the top node)
            pred_pos[0] = self.init_state.pos0[0]

            # Inextensibility constraint for each edge
            # Constraint is difference between length and rest length
            p1_minus_p2 = pred_pos[i] - pred_pos[i + 1]
            distance = np.linalg.norm(p1_minus_p2, axis=1)
            constraint = distance - self.init_state.l_bar_edge[i]
            # Update lambda and position
            d_lambda = (-constraint) / (sum_mass + compliance)
            correction_vector = d_lambda[:, None] * p1_minus_p2 / (distance[:, None] + 1e-8)
            lambdas[i] += d_lambda
            pred_pos[i] += inv_mass[:, None] * correction_vector
            pred_pos[i + 1] -= inv_mass2[:, None] * correction_vector

        return pred_pos.reshape(-1, 3)
