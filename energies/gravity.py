import cupy as cp

from energies.energy import Energy
from rod.rod import RodState, InitialRodState, RodParams


class Gravity(Energy):
    @staticmethod
    def compute_energy(pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams) -> float:
        energy = cp.sum(rod_params.mass * rod_params.g * pos[:, 2])
        return energy

    @staticmethod
    def d_energy_d_theta(grad: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams) -> cp.ndarray:
        return grad  # No theta dependence

    @staticmethod
    def d_energy_d_pos(grad: cp.ndarray, pos: cp.ndarray, theta: cp.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState, rod_params: RodParams) -> cp.ndarray:
        grad[:, 2] += rod_params.mass * rod_params.g
        return grad

