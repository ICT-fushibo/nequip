"""Opt4 fixed-cell energy/force/stress VJP; the Opt2 force-only path is unchanged."""

from __future__ import annotations

import torch
from nequip.data import AtomicDataDict


class ForceStressEnergyVJP(torch.nn.Module):
    """Same affine strain convention as ForceStressOutput, with static scratch.

    Strain stays zero. Both positions and cell are strained so periodic shifts
    participate in the derivative. No training/double-backward support is added.
    """

    def __init__(self, energy_model, positions, cell):
        super().__init__()
        self.energy_model = energy_model
        self.register_buffer(
            "strain",
            torch.zeros(
                (3, 3),
                device=positions.device,
                dtype=positions.dtype,
                requires_grad=True,
            ),
        )
        self.register_buffer(
            "volume", torch.linalg.det(cell.reshape(3, 3)).abs().detach()
        )

    def forward(self, inputs):
        positions = inputs[AtomicDataDict.POSITIONS_KEY]
        cell = inputs[AtomicDataDict.CELL_KEY].reshape(3, 3)
        data = dict(inputs)
        strain = 0.5 * (self.strain + self.strain.T)
        data[AtomicDataDict.POSITIONS_KEY] = positions + positions @ strain
        data[AtomicDataDict.CELL_KEY] = (cell + cell @ strain).reshape_as(
            inputs[AtomicDataDict.CELL_KEY]
        )
        # No cached edge vectors from an earlier strain/geometry may survive.
        data.pop(AtomicDataDict.EDGE_VECTORS_KEY, None)
        data.pop(AtomicDataDict.EDGE_LENGTH_KEY, None)
        output = self.energy_model(data)
        energy = output[AtomicDataDict.TOTAL_ENERGY_KEY]
        grad_positions, grad_strain = torch.autograd.grad(
            energy.sum(),
            (positions, self.strain),
            create_graph=False,
            retain_graph=False,
        )
        self.last_stress = (grad_strain / self.volume).detach()
        return energy.reshape(-1), -grad_positions
