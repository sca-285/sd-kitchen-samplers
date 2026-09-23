# SPDX-License-Identifier: GPL-3.0-or-later
# From MisterChief95 / sd-forge-extra-samplers.
# See NOTICE.md for authors of individual samplers.

"""Helpers from MisterChief95's Extra Samplers (lib_es/utils.py). Dy/SMEA steps come from lib_kitchen.canon.smea."""
import math
from enum import Enum

import torch

from lib_kitchen.canon.common import to_d
from lib_kitchen.canon.smea import Rescaler as _Rescaler, dy_sampling_step, smea_sampling_step  # noqa: F401


@torch.no_grad()
def overall_sampling_step(x, model, dt, sigma_hat, **extra_args):
    original_shape = x.shape
    batch_size, channels, m, n = original_shape[0], original_shape[1], original_shape[2] // 2, original_shape[3] // 2
    extra_row = x.shape[2] % 2 == 1
    extra_col = x.shape[3] % 2 == 1

    if extra_row:
        extra_row_content = x[:, :, -1:, :]
        x = x[:, :, :-1, :]

    if extra_col:
        extra_col_content = x[:, :, :, -1:]
        x = x[:, :, :, :-1]

    a_list = x.unfold(2, 2, 2).unfold(3, 2, 2).contiguous().view(batch_size, channels, m * n, 2, 2)
    c = a_list[:, :, :, 1, 1].view(batch_size, channels, m, n)

    denoised = model(c, sigma_hat * c.new_ones([c.shape[0]]), **extra_args)
    d = to_d(c, sigma_hat, denoised)
    c = c + d * dt

    d_list = denoised.view(batch_size, channels, m * n, 1, 1)
    a_list[:, :, :, 1, 1] = d_list[:, :, :, 0, 0]

    x = (
        a_list.view(batch_size, channels, m, n, 2, 2)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(batch_size, channels, 2 * m, 2 * n)
    )

    if extra_row or extra_col:
        x_expanded = torch.zeros(original_shape, dtype=x.dtype, device=x.device)
        x_expanded[:, :, : 2 * m, : 2 * n] = x

        if extra_row:
            x_expanded[:, :, -1:, : 2 * n + 1] = extra_row_content

        if extra_col:
            x_expanded[:, :, : 2 * m, -1:] = extra_col_content

        if extra_row and extra_col:
            x_expanded[:, :, -1:, -1:] = extra_col_content[:, :, -1:, :]

        x = x_expanded

    return x


def sampler_metadata(name: str, extra_params: dict = {}, sampler_aliases: list[str] = []):
    def decorator(func):
        func.sampler_extra_params = extra_params
        func.sampler_name = name
        func.sampler_k_names = [name.replace(" ", "_").lower(), *sampler_aliases]
        return func

    return decorator


class Interpolator(Enum):
    LINEAR = (lambda x: x,)  # noqa: E731
    COSINE = (lambda x: torch.sin(x * math.pi / 2),)  # noqa: E731
    SINE = (lambda x: 1 - torch.cos(x * math.pi / 2),)


def extend_sigmas(
    sigmas: torch.Tensor,
    steps: int,
    start_at_sigma: float,
    end_at_sigma: float,
    interpolator: Interpolator = Interpolator.LINEAR,
) -> torch.FloatTensor:
    if start_at_sigma < 0:
        start_at_sigma = float("inf")

    # linear space for our interpolation function
    x = torch.linspace(0, 1, steps + 1, device=sigmas.device)[1:-1]
    computed_spacing: torch.Tensor = interpolator.value[0](x)

    extended_sigmas: list[torch.Tensor] = []
    for i in range(len(sigmas) - 1):
        sigma_current = sigmas[i]
        sigma_next = sigmas[i + 1]

        extended_sigmas.append(sigma_current)

        if end_at_sigma <= sigma_current <= start_at_sigma:
            interpolated_steps: torch.Tensor = computed_spacing * (sigma_next - sigma_current) + sigma_current
            extended_sigmas.extend(interpolated_steps.tolist())

    # Add the last sigma value
    if len(sigmas) > 0:
        extended_sigmas.append(sigmas[-1])

    return torch.FloatTensor(extended_sigmas).to(sigmas.device)
