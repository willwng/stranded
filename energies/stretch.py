import jax.numpy as np

from energies.energy import Energy
from rod.rod import RodState, InitialRodState, RodParams
from rod.rod_util import RodUtil


class Stretch(Energy):
    """
    Represents the stretching energy of a rod (based on inextensible edges)
        - Note! For inextensible rods we use XPBD to solve the constraint
            rather than this specific energy
    """

    @staticmethod
    def compute_energy(pos: np.ndarray, theta: np.ndarray, rod_state: RodState, init_rod_state: InitialRodState,
                       rod_params: RodParams):
        edge_lengths = RodUtil.compute_edge_lengths(pos=pos)
        rest_edge_lengths = init_rod_state.l_bar_edge
        energy = 0.5 * rod_params.k * np.sum((edge_lengths - rest_edge_lengths) ** 2)
        return energy

    @staticmethod
    def d_energy_d_theta(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                         init_rod_state: InitialRodState, rod_params: RodParams):
        return grad  # No theta dependence

    @staticmethod
    def d_energy_d_pos(grad: np.ndarray, pos: np.ndarray, theta: np.ndarray, rod_state: RodState,
                       init_rod_state: InitialRodState,
                       rod_params: RodParams):
        raise NotImplementedError
