from __future__ import annotations
from typing import Tuple
import torch
import torch.nn as nn

ENVIRONMENTS = ("void", "sheet", "filament", "cluster")

_AMP_FLOOR = 1e-06


def marked_field(mark, delta, normalize=False, amplitude_floor=_AMP_FLOOR):
    field = mark * (1.0 + delta)
    A = field.mean(dim=(-3, -2, -1), keepdim=True)
    if normalize:
        denom = torch.where(
            A >= 0, A.clamp(min=amplitude_floor), A.clamp(max=-amplitude_floor)
        )
        return (field / denom - 1.0, A)
    return (field - A, A)


class CanonicalMark(nn.Module):
    mark_normalize: bool = False

    def _marked_field_and_amplitude(
        self, mark: torch.Tensor, delta: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return marked_field(mark, delta, self.mark_normalize)

    def _make_final_field(
        self, mark: torch.Tensor, delta: torch.Tensor
    ) -> torch.Tensor:
        return self._marked_field_and_amplitude(mark, delta)[0]
